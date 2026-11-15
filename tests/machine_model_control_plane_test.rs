//! End-to-end integration tests for Wave 3 Machine-Model Control Plane (`LMA1980/jero-manager#43`–`#51`).

use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use arrow_array::{BooleanArray, Float32Array, Float64Array, Int8Array, Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use duckdb::Connection;
use jero_core::profile::CONSTRAINED_TEST_N_CTX;
use jero_manager::{
    AdminCommand, CapabilityFilter, CasCatalog, ContentAddressedStore, ControlPlaneCapacityConfig,
    ControlPlaneRuntime, DuckDbPersistenceStore, HysteresisEventTranslator, MachineEventRule,
    MachineFeatureSchema, MachineInferRequest, MachineInferResponse, MachineModelManifest,
    MockOsAffinityBackend, ModelManifest, OpenAiChatRequest, OpenAiMessage, OsAffinityBackend,
    PromptOrchestrator, PromptPreparationRequest, ProtocolServer, ProtocolServerConfig, Sha256,
    SimpleTokenizer, SystemEventPayload, extract_feature_matrix_from_record_batch,
    format_symbolic_machine_state, handle_machine_infer,
};

static TEST_DIR_SEQ: AtomicU64 = AtomicU64::new(1);

struct TempDirGuard {
    path: PathBuf,
}

impl TempDirGuard {
    fn new(prefix: &str) -> Self {
        let seq = TEST_DIR_SEQ.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("{prefix}_{}_{seq}", std::process::id()));
        fs::create_dir_all(&path).expect("create temp dir");
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn response_body_string(resp: jero_manager::protocol::HttpResponse) -> String {
    match resp.body {
        jero_manager::protocol::HttpResponseBody::Bytes(bytes) => {
            String::from_utf8(bytes).expect("utf-8 response body")
        }
        jero_manager::protocol::HttpResponseBody::Stream(_) => {
            panic!("expected buffered bytes HTTP response")
        }
    }
}

fn build_machine_test_runtime() -> Arc<ControlPlaneRuntime> {
    let capacity = ControlPlaneCapacityConfig::default()
        .resolve_default()
        .expect("valid default control capacity");
    let mock_affinity: Arc<dyn OsAffinityBackend> = Arc::new(MockOsAffinityBackend::default());
    Arc::new(
        ControlPlaneRuntime::with_affinity_backend(capacity, mock_affinity)
            .expect("create runtime"),
    )
}

fn register_test_model(runtime: &ControlPlaneRuntime, model_id: &str, cluster: &str) {
    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: model_id.to_owned(),
            weights_path: PathBuf::from(format!("/models/{model_id}.gguf")),
            cluster: cluster.to_owned(),
        })
        .unwrap_or_else(|e| panic!("failed to register test model '{model_id}': {e}"));
}

fn sample_schema(
    target_subsystem: &str,
    event_code: &str,
    enter: f32,
    exit: f32,
    cooldown_ms: u64,
) -> MachineFeatureSchema {
    let mut mappings = BTreeMap::new();
    mappings.insert(
        "anomaly".to_owned(),
        MachineEventRule {
            event_code: event_code.to_owned(),
            target_subsystem: target_subsystem.to_owned(),
            enter_threshold: Some(enter),
            exit_threshold: Some(exit),
            cooldown_ms: Some(cooldown_ms),
        },
    );
    MachineFeatureSchema {
        feature_names: vec![
            "cpu_util".to_owned(),
            "queue_depth".to_owned(),
            "mem_pressure".to_owned(),
        ],
        feature_types: vec!["f32".to_owned(), "i32".to_owned(), "f64".to_owned()],
        zscore_mean: vec![0.40, 4.0, 0.50],
        zscore_std: vec![0.10, 2.0, 0.25],
        event_mappings: mappings,
    }
}

fn ingest_synthetic_gguf_blob(store: &ContentAddressedStore) -> (String, u64) {
    let mut bytes = Vec::with_capacity(32);
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(b"MACHINE_TEST");
    let digest = Sha256::digest(&bytes);
    let len = u64::try_from(bytes.len()).expect("len fits u64");
    let (staging_path, mut file) = store.create_staging_file().expect("create staging file");
    file.write_all(&bytes).expect("write synthetic gguf");
    file.flush().expect("flush synthetic gguf");
    drop(file);
    store
        .promote_staged_blob(&staging_path, &digest, "gguf")
        .expect("promote synthetic gguf blob");
    (digest, len)
}

async fn assert_machine_infer_fail_closed_http_codes(
    runtime: Arc<ControlPlaneRuntime>,
    valid_req: &MachineInferRequest,
) {
    let get_resp = handle_machine_infer(
        jero_manager::protocol::HttpRequest::new("GET", "/v1/machine/infer", Vec::new()),
        Arc::clone(&runtime),
    )
    .await;
    assert_eq!(get_resp.status, 405);

    let nan_json = serde_json::json!({
        "model": "ttm-telemetry",
        "features": [0.4, 4.0, 0.5],
        "clip_sigma": -1.0
    });
    let bad_sigma_resp = handle_machine_infer(
        jero_manager::protocol::HttpRequest::new(
            "POST",
            "/v1/machine/infer",
            serde_json::to_vec(&nan_json).unwrap(),
        ),
        Arc::clone(&runtime),
    )
    .await;
    assert_eq!(bad_sigma_resp.status, 400);

    let wrong_cols_req = MachineInferRequest {
        model: "ttm-telemetry".to_owned(),
        features: vec![0.1, 0.2],
        rows: None,
        num_rows: None,
        num_cols: Some(2),
        clip_sigma: None,
        timestamp_ms: None,
        already_normalized: false,
    };
    let wrong_cols_resp = handle_machine_infer(
        jero_manager::protocol::HttpRequest::new(
            "POST",
            "/v1/machine/infer",
            serde_json::to_vec(&wrong_cols_req).unwrap(),
        ),
        Arc::clone(&runtime),
    )
    .await;
    assert_eq!(wrong_cols_resp.status, 400);

    let missing_req = MachineInferRequest {
        model: "missing-machine-model".to_owned(),
        features: vec![1.0],
        rows: None,
        num_rows: None,
        num_cols: None,
        clip_sigma: None,
        timestamp_ms: None,
        already_normalized: true,
    };
    let missing_resp = handle_machine_infer(
        jero_manager::protocol::HttpRequest::new(
            "POST",
            "/v1/machine/infer",
            serde_json::to_vec(&missing_req).unwrap(),
        ),
        Arc::clone(&runtime),
    )
    .await;
    assert_eq!(missing_resp.status, 404);

    runtime.update_worker_health("ttm-telemetry", false, 1);
    let dead_resp = handle_machine_infer(
        jero_manager::protocol::HttpRequest::new(
            "POST",
            "/v1/machine/infer",
            serde_json::to_vec(valid_req).unwrap(),
        ),
        runtime,
    )
    .await;
    assert_eq!(dead_resp.status, 503);
}

#[tokio::test]
async fn test_post_v1_machine_infer_and_flight_batch() {
    let runtime = build_machine_test_runtime();
    let server = ProtocolServer::new(ProtocolServerConfig::default(), Arc::clone(&runtime));
    register_test_model(&runtime, "ttm-telemetry", "x100");
    runtime
        .register_machine_schema(
            "ttm-telemetry",
            sample_schema("ActionRunner", "STAGE_RESCHEDULE", 0.80, 0.35, 5_000),
        )
        .expect("register schema");

    let observed_normalized: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    let captured = Arc::clone(&observed_normalized);
    runtime.set_machine_infer_evaluator(Arc::new(move |_req, normalized| {
        *captured.lock().expect("lock") = normalized.to_vec();
        Ok(vec![0.88])
    }));

    // 1. Valid POST /v1/machine/infer via HTTP router triggers z-score scaling + event emission
    // raw: [0.40, 12.0, 0.75] -> z = [(0.4-0.4)/0.1, (12-4)/2, (0.75-0.5)/0.25] = [0.0, 4.0, 1.0]
    let infer_req = MachineInferRequest {
        model: "ttm-telemetry".to_owned(),
        features: vec![0.40, 12.0, 0.75],
        rows: None,
        num_rows: None,
        num_cols: None,
        clip_sigma: Some(8.0),
        timestamp_ms: Some(1_000),
        already_normalized: false,
    };
    let http_resp = server
        .route_request(jero_manager::protocol::HttpRequest::new(
            "POST",
            "/v1/machine/infer",
            serde_json::to_vec(&infer_req).expect("serialize request"),
        ))
        .await;
    assert_eq!(http_resp.status, 200);
    let parsed: MachineInferResponse =
        serde_json::from_str(&response_body_string(http_resp)).expect("parse response");
    assert_eq!(parsed.model, "ttm-telemetry");
    assert_eq!(parsed.num_rows, 1);
    assert_eq!(parsed.num_cols, 3);
    assert_eq!(parsed.outputs, vec![0.88]);
    let norm = observed_normalized.lock().expect("lock").clone();
    assert!((norm[0] - 0.0).abs() < 1e-5);
    assert!((norm[1] - 4.0).abs() < 1e-5);
    assert!((norm[2] - 1.0).abs() < 1e-5);

    let event = parsed.emitted_event.expect("emitted SystemEventPayload");
    event.validate().expect("valid payload");
    assert_eq!(event.event_code, "STAGE_RESCHEDULE");
    assert_eq!(event.target_subsystem, "ActionRunner");
    assert_eq!(event.top_causal_features[0].0, "queue_depth");
    assert!((event.top_causal_features[0].1 - 4.0).abs() < 1e-5);

    // 2. Arrow RecordBatch flight batch feeds straight into machine inference (outside cooldown)
    let arrow_schema = Arc::new(Schema::new(vec![
        Field::new("cpu_util", DataType::Float32, false),
        Field::new("queue_depth", DataType::Int32, false),
        Field::new("mem_pressure", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        arrow_schema,
        vec![
            Arc::new(Float32Array::from(vec![0.40, 0.70])),
            Arc::new(Int32Array::from(vec![4, 14])),
            Arc::new(Float64Array::from(vec![0.50, 0.75])),
        ],
    )
    .expect("construct record batch");
    runtime.set_machine_infer_evaluator(Arc::new(|_req, normalized| {
        assert_eq!(normalized.len(), 6);
        Ok(vec![0.20, 0.91])
    }));
    let batch_resp = runtime
        .evaluate_machine_record_batch("ttm-telemetry", &batch, None, Some(10_000))
        .expect("evaluate record batch");
    assert_eq!(batch_resp.num_rows, 2);
    assert_eq!(batch_resp.num_cols, 3);
    assert_eq!(batch_resp.outputs, vec![0.20, 0.91]);
    let batch_evt = batch_resp
        .emitted_event
        .expect("second row triggers after exit & cooldown");
    assert_eq!(batch_evt.top_causal_features[0].0, "queue_depth");
    assert_eq!(batch_evt.timestamp_ms, 10_000);

    assert_machine_infer_fail_closed_http_codes(Arc::clone(&runtime), &infer_req).await;
}

fn create_legacy_duckdb_without_machine_schema(db_path: &Path) {
    let raw_conn = Connection::open(db_path).expect("open legacy duckdb");
    raw_conn
        .execute_batch(
            "CREATE TABLE model_blobs (
                digest TEXT PRIMARY KEY NOT NULL,
                size_bytes BIGINT NOT NULL,
                blob_type TEXT NOT NULL,
                created_at BIGINT NOT NULL,
                last_accessed_at BIGINT NOT NULL,
                ref_count INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE model_manifests (
                manifest_digest TEXT PRIMARY KEY NOT NULL,
                weight_digest TEXT NOT NULL REFERENCES model_blobs(digest),
                format TEXT NOT NULL,
                architecture_id TEXT NOT NULL,
                recipe_version TEXT NOT NULL,
                quantization TEXT NOT NULL,
                context_length BIGINT NOT NULL,
                license TEXT,
                origin TEXT,
                manifest_json TEXT NOT NULL,
                created_at BIGINT NOT NULL
            );",
        )
        .expect("create legacy schema");
}

fn build_mixed_telemetry_schema() -> MachineFeatureSchema {
    let mut mappings = BTreeMap::new();
    mappings.insert(
        "default".to_owned(),
        MachineEventRule {
            event_code: "CANARY_ROLLBACK".to_owned(),
            target_subsystem: "MuConDe".to_owned(),
            enter_threshold: Some(0.78),
            exit_threshold: Some(0.32),
            cooldown_ms: Some(4_000),
        },
    );
    MachineFeatureSchema {
        feature_names: vec![
            "p99_latency_ms".to_owned(),
            "error_rate".to_owned(),
            "thermal_celsius".to_owned(),
            "throttle_active".to_owned(),
        ],
        feature_types: vec![
            "f32".to_owned(),
            "f64".to_owned(),
            "i8".to_owned(),
            "bool".to_owned(),
        ],
        zscore_mean: vec![12.0, 0.01, 55.0, 0.0],
        zscore_std: vec![4.0, 0.005, 5.0, 1.0],
        event_mappings: mappings,
    }
}

fn build_mixed_telemetry_batch() -> RecordBatch {
    let arrow_schema = Arc::new(Schema::new(vec![
        Field::new("p99_latency_ms", DataType::Float32, false),
        Field::new("error_rate", DataType::Float64, false),
        Field::new("thermal_celsius", DataType::Int8, false),
        Field::new("throttle_active", DataType::Boolean, false),
    ]));
    RecordBatch::try_new(
        arrow_schema,
        vec![
            Arc::new(Float32Array::from(vec![16.0])),
            Arc::new(Float64Array::from(vec![0.02])),
            Arc::new(Int8Array::from(vec![75])),
            Arc::new(BooleanArray::from(vec![true])),
        ],
    )
    .expect("build mixed batch")
}

#[tokio::test]
async fn test_machine_schema_manifest_and_http_flight_ingest() {
    let tmp = TempDirGuard::new("k3_machine_schema_e2e");
    let db_path = tmp.path().join("legacy_migrated_catalog.duckdb");
    create_legacy_duckdb_without_machine_schema(&db_path);

    let catalog = Arc::new(DuckDbPersistenceStore::open(&db_path).expect("open migrated catalog"));
    let store =
        ContentAddressedStore::open(tmp.path(), Arc::clone(&catalog) as Arc<dyn CasCatalog>)
            .expect("open cas");
    let (weight_digest, weight_size_bytes) = ingest_synthetic_gguf_blob(&store);

    let schema = build_mixed_telemetry_schema();

    let core = ModelManifest::new(
        "gguf",
        "ft-transformer",
        "1.0.0",
        "F32",
        512,
        weight_digest,
        weight_size_bytes,
    );
    let manifest = MachineModelManifest::new(core, Some(schema.clone())).expect("valid manifest");
    let rec = store
        .register_machine_manifest(&manifest)
        .expect("register manifest");
    store
        .create_alias(
            "local",
            "control",
            "ft-canary-guard",
            "v1",
            &rec.manifest_digest,
        )
        .expect("create alias");

    // Reopen DuckDB catalog and confirm schema loads identically from alias & machine_schema_json
    drop(store);
    drop(catalog);
    let catalog2 = Arc::new(DuckDbPersistenceStore::open(&db_path).expect("reopen migrated db"));
    assert!(
        catalog2
            .get_machine_schema_json(&rec.manifest_digest)
            .expect("query column")
            .is_some()
    );
    let store2 =
        ContentAddressedStore::open(tmp.path(), Arc::clone(&catalog2) as Arc<dyn CasCatalog>)
            .expect("reopen cas");
    let resolved_schema = store2
        .get_machine_schema("local/control/ft-canary-guard/v1")
        .expect("load schema by alias")
        .expect("schema present");
    assert_eq!(resolved_schema, schema);
    assert_ne!(resolved_schema.schema_hash(), 0);

    // Wire Arrow mixed-dtype batch + resolved schema into HTTP /v1/machine/infer runtime
    let runtime = build_machine_test_runtime();
    register_test_model(&runtime, "ft-canary-guard", "x100");
    runtime
        .register_machine_schema("ft-canary-guard", resolved_schema.clone())
        .expect("register resolved schema on runtime");
    runtime.set_machine_infer_evaluator(Arc::new(|_req, norm| {
        // Check clipped/normalized features from RecordBatch: thermal z = (75 - 55) / 5 = 4.0
        assert!(norm[2] > 3.5);
        Ok(vec![0.92])
    }));

    let batch = build_mixed_telemetry_batch();
    let expected_cols: Vec<&str> = schema.feature_names.iter().map(String::as_str).collect();
    let matrix = extract_feature_matrix_from_record_batch(&batch, Some(&expected_cols))
        .expect("extract feature matrix");

    let req = MachineInferRequest {
        model: "ft-canary-guard".to_owned(),
        features: matrix.values,
        rows: None,
        num_rows: Some(u32::try_from(matrix.num_rows).unwrap()),
        num_cols: Some(u32::try_from(matrix.num_cols).unwrap()),
        clip_sigma: None,
        timestamp_ms: Some(5_000),
        already_normalized: false,
    };
    let resp = handle_machine_infer(
        jero_manager::protocol::HttpRequest::new(
            "POST",
            "/v1/machine/infer",
            serde_json::to_vec(&req).unwrap(),
        ),
        Arc::clone(&runtime),
    )
    .await;
    assert_eq!(resp.status, 200);
    let parsed: MachineInferResponse =
        serde_json::from_str(&response_body_string(resp)).expect("parse response");
    let evt = parsed.emitted_event.expect("canary event emitted");
    assert_eq!(evt.event_code, "CANARY_ROLLBACK");
    assert_eq!(evt.target_subsystem, "MuConDe");
    assert_eq!(evt.top_causal_features[0].0, "thermal_celsius");
}

#[tokio::test]
async fn test_hysteresis_suppresses_single_tick_spike_and_prevents_llm_wakeup() {
    let runtime = build_machine_test_runtime();
    register_test_model(&runtime, "ttm-spike-guard", "x100");
    let translator = HysteresisEventTranslator::new(0.80, 0.35, 5_000, "MEM_CLAMP", "MuIsolation")
        .expect("valid translator")
        .with_persistence(2, 3)
        .expect("2-of-3 persistence")
        .with_feature_names(vec!["rss_ratio".to_owned(), "page_faults".to_owned()])
        .expect("feature names");
    runtime.set_machine_hysteresis_translator("ttm-spike-guard", translator);

    let llm_wakeups = Arc::new(AtomicUsize::new(0));
    runtime.set_machine_infer_evaluator(Arc::new(move |req, _norm| {
        let score = if req.timestamp_ms == Some(100) {
            0.96
        } else {
            0.20
        };
        Ok(vec![score])
    }));

    for (ts, features) in [(100_u64, vec![4.5, 1.0]), (200_u64, vec![0.2, 0.1])] {
        let infer = runtime
            .evaluate_machine_infer(&MachineInferRequest {
                model: "ttm-spike-guard".to_owned(),
                features,
                rows: None,
                num_rows: Some(1),
                num_cols: Some(2),
                clip_sigma: None,
                timestamp_ms: Some(ts),
                already_normalized: true,
            })
            .expect("infer succeeds");
        if let Some(evt) = infer.emitted_event {
            let _prompt = format_symbolic_machine_state(&evt, "Investigate alert");
            llm_wakeups.fetch_add(1, Ordering::SeqCst);
        }
    }

    assert_eq!(
        llm_wakeups.load(Ordering::SeqCst),
        0,
        "transient single-tick spike must never wake downstream LLM"
    );
}

#[tokio::test]
async fn test_sustained_anomaly_emits_attributed_event_and_triggers_llm_reasoning() {
    let runtime = build_machine_test_runtime();
    let server = ProtocolServer::new(ProtocolServerConfig::default(), Arc::clone(&runtime));
    register_test_model(&runtime, "ttm-monitor", "x100");
    register_test_model(&runtime, "qwen3-ops", "a100");

    let translator =
        HysteresisEventTranslator::new(0.80, 0.35, 5_000, "CANARY_ROLLBACK", "MuConDe")
            .expect("valid translator")
            .with_persistence(2, 3)
            .expect("2-of-3 persistence")
            .with_feature_stats(
                vec![
                    "p99_latency_ms".to_owned(),
                    "error_rate".to_owned(),
                    "cpu_temp_c".to_owned(),
                ],
                vec![20.0, 0.01, 55.0],
                vec![5.0, 0.01, 5.0],
            )
            .expect("feature stats");
    runtime.set_machine_hysteresis_translator("ttm-monitor", translator);
    runtime.set_machine_infer_evaluator(Arc::new(|req, _norm| {
        if req.timestamp_ms.unwrap_or(0) >= 1_000 {
            Ok(vec![0.93])
        } else {
            Ok(vec![0.10])
        }
    }));

    let mut llm_responses = Vec::new();
    for ts in [1_000_u64, 2_000_u64, 3_000_u64] {
        let resp = runtime
            .evaluate_machine_infer(&MachineInferRequest {
                model: "ttm-monitor".to_owned(),
                features: vec![30.0, 0.07, 65.0], // z = [2.0, 6.0, 2.0] -> error_rate leads at 6.0
                rows: None,
                num_rows: Some(1),
                num_cols: Some(3),
                clip_sigma: None,
                timestamp_ms: Some(ts),
                already_normalized: false,
            })
            .expect("machine infer");
        if let Some(event) = resp.emitted_event {
            let prompt = format_symbolic_machine_state(
                &event,
                "Generate concise {target_subsystem} action for {event_code} drivers={top_causal_features}.",
            );
            assert!(prompt.contains("MuConDe"));
            assert!(prompt.contains("CANARY_ROLLBACK"));
            assert!(prompt.contains("error_rate=6.000"));

            let chat_req = OpenAiChatRequest {
                model: "qwen3-ops".to_owned(),
                messages: vec![OpenAiMessage::user(prompt)],
                stream: Some(false),
                tools: None,
                tool_choice: None,
                max_tokens: Some(u32::try_from(CONSTRAINED_TEST_N_CTX).unwrap()),
                max_completion_tokens: None,
                temperature: None,
                top_p: None,
                stop: None,
            };
            let chat_http = server
                .route_request(jero_manager::protocol::HttpRequest::new(
                    "POST",
                    "/v1/chat/completions",
                    serde_json::to_vec(&chat_req).unwrap(),
                ))
                .await;
            assert_eq!(chat_http.status, 200);
            llm_responses.push(response_body_string(chat_http));
        }
    }

    assert_eq!(
        llm_responses.len(),
        1,
        "sustained 2-of-3 ticks emits once while remaining latched/cooled down"
    );
}

#[test]
fn test_symbolic_prompt_conditioning_injects_structured_machine_context() {
    let event = SystemEventPayload {
        event_code: "STAGE_RESCHEDULE".to_owned(),
        target_subsystem: "ActionRunner".to_owned(),
        confidence: 0.9125,
        top_causal_features: vec![
            ("worker_queue_depth".to_owned(), 5.5),
            ("cpu_temp\u{200B}_c".to_owned(), 3.25),
        ],
        timestamp_ms: 98_765,
    };

    let untemplated = format_symbolic_machine_state(&event, "Assess node-a worker load.\u{FEFF}");
    assert!(untemplated.contains("[MACHINE_STATE"));
    assert!(untemplated.contains("event_code=STAGE_RESCHEDULE"));
    assert!(untemplated.contains("target_subsystem=ActionRunner"));
    assert!(untemplated.contains("confidence=0.9125"));
    assert!(untemplated.contains("timestamp_ms=98765"));
    assert!(untemplated.contains("worker_queue_depth=5.500"));
    assert!(untemplated.contains("cpu_temp_c=3.250"));
    assert!(!untemplated.contains('\u{200B}'));
    assert!(!untemplated.contains('\u{FEFF}'));

    let mut tok = SimpleTokenizer::new();
    for (idx, word) in [
        "Assess",
        "ActionRunner",
        "STAGE_RESCHEDULE",
        "on",
        "node-a",
        "worker_queue_depth",
    ]
    .into_iter()
    .enumerate()
    {
        tok.add_token(word, u32::try_from(idx + 10).unwrap());
    }
    let orchestrator = PromptOrchestrator::new();
    orchestrator.register_tokenizer("qwen3-ops", Arc::new(tok));
    let prepared = orchestrator
        .prepare_prompt(
            "qwen3-ops",
            &PromptPreparationRequest::new(
                "Assess ActionRunner STAGE_RESCHEDULE on node-a",
                false,
                CONSTRAINED_TEST_N_CTX,
            ),
        )
        .expect("prepare conservative n_ctx=64 prompt");
    assert_eq!(prepared.token_ids.len(), 5);
}

fn validate_chained_recipe_manifests() {
    for arch in [
        "ttm",
        "ft-transformer",
        "tabpfn-v2",
        "decision-transformer",
        "qwen3",
    ] {
        let manifest = ModelManifest::new(
            "gguf",
            arch,
            "1.0.0",
            "F32",
            512,
            "sha256:chained_pipeline",
            128_000_000,
        );
        CapabilityFilter::validate_model_manifest(&manifest)
            .unwrap_or_else(|e| panic!("recipe '{arch}@1.0.0' must validate: {e}"));
    }
}

#[tokio::test]
async fn test_chained_pipeline_ttm_to_ft_transformer_to_llm() {
    validate_chained_recipe_manifests();

    // Wire Stage 1 (ttm) -> Stage 2 (ft-transformer) -> Stage 3 (qwen3 symbolic fallback)
    let runtime = build_machine_test_runtime();
    let server = ProtocolServer::new(ProtocolServerConfig::default(), Arc::clone(&runtime));
    register_test_model(&runtime, "ttm-forecaster", "x100");
    register_test_model(&runtime, "ft-risk-scorer", "x100");
    register_test_model(&runtime, "qwen3-planner", "a100");

    runtime
        .register_machine_schema(
            "ft-risk-scorer",
            sample_schema("ActionRunner", "STAGE_RESCHEDULE", 0.80, 0.35, 2_000),
        )
        .expect("register stage 2 schema");

    runtime.set_machine_infer_evaluator(Arc::new(|req, norm| {
        if req.model == "ttm-forecaster" {
            // Forecast future utilization/queue trajectory from current telemetry
            Ok(vec![norm[0] + 0.35, norm[1] + 6.0, norm[2] + 0.40])
        } else {
            // FT-Transformer evaluates combined forecasted state and emits high anomaly probability
            let max_abs_z = norm.iter().copied().map(f32::abs).fold(0.0_f32, f32::max);
            let score = if max_abs_z >= 2.5 { 0.94 } else { 0.15 };
            Ok(vec![score])
        }
    }));

    // Ingest worker-riscv64 telemetry batch via Arrow RecordBatch -> Stage 1 (TTM)
    let stage1_schema = Arc::new(Schema::new(vec![
        Field::new("cpu_util", DataType::Float32, false),
        Field::new("queue_depth", DataType::Int32, false),
        Field::new("mem_pressure", DataType::Float64, false),
    ]));
    let stage1_batch = RecordBatch::try_new(
        stage1_schema,
        vec![
            Arc::new(Float32Array::from(vec![0.40])),
            Arc::new(Int32Array::from(vec![4])),
            Arc::new(Float64Array::from(vec![0.50])),
        ],
    )
    .expect("stage 1 batch");

    let stage1_resp = runtime
        .evaluate_machine_record_batch("ttm-forecaster", &stage1_batch, None, Some(10_000))
        .expect("stage 1 ttm forecast");
    assert_eq!(stage1_resp.outputs.len(), 3);

    // Pipe Stage 1 forecast outputs into Stage 2 (ft-transformer) over POST /v1/machine/infer
    let stage2_http = server
        .route_request(jero_manager::protocol::HttpRequest::new(
            "POST",
            "/v1/machine/infer",
            serde_json::to_vec(&MachineInferRequest {
                model: "ft-risk-scorer".to_owned(),
                features: stage1_resp.outputs,
                rows: None,
                num_rows: Some(1),
                num_cols: Some(3),
                clip_sigma: None,
                timestamp_ms: Some(10_100),
                already_normalized: false,
            })
            .expect("serialize stage 2 request"),
        ))
        .await;
    assert_eq!(stage2_http.status, 200);
    let stage2_parsed: MachineInferResponse =
        serde_json::from_str(&response_body_string(stage2_http)).expect("parse stage 2");
    assert!((stage2_parsed.outputs[0] - 0.94).abs() < 1e-6);

    let event = stage2_parsed
        .emitted_event
        .expect("stage 2 emits ActionRunner reschedule event");
    assert_eq!(event.event_code, "STAGE_RESCHEDULE");
    assert_eq!(event.target_subsystem, "ActionRunner");
    assert_eq!(event.top_causal_features[0].0, "cpu_util");

    // Stage 3 symbolic prompt bridge invokes downstream qwen3-planner over /v1/chat/completions
    let llm_prompt = format_symbolic_machine_state(
        &event,
        "Coordinate {target_subsystem} execution shift after {event_code}.",
    );
    let stage3_http = server
        .route_request(jero_manager::protocol::HttpRequest::new(
            "POST",
            "/v1/chat/completions",
            serde_json::to_vec(&OpenAiChatRequest {
                model: "qwen3-planner".to_owned(),
                messages: vec![OpenAiMessage::user(llm_prompt)],
                stream: Some(false),
                tools: None,
                tool_choice: None,
                max_tokens: Some(u32::try_from(CONSTRAINED_TEST_N_CTX).unwrap()),
                max_completion_tokens: None,
                temperature: None,
                top_p: None,
                stop: None,
            })
            .expect("serialize stage 3 chat"),
        ))
        .await;
    assert_eq!(stage3_http.status, 200);
}

#[tokio::test]
async fn test_already_normalized_causal_attribution_avoids_double_zscore_normalization() {
    let runtime = build_machine_test_runtime();
    register_test_model(&runtime, "ttm-pre-norm", "x100");
    // mean = [0.40, 4.0, 0.50], std = [0.10, 2.0, 0.25]
    runtime
        .register_machine_schema(
            "ttm-pre-norm",
            sample_schema("ActionRunner", "STAGE_RESCHEDULE", 0.80, 0.35, 5_000),
        )
        .expect("register schema");
    runtime.set_machine_infer_evaluator(Arc::new(|_req, norm| {
        assert!((norm[1] - 4.0).abs() < 1e-6);
        Ok(vec![0.91])
    }));

    // With already_normalized: true, [0.0, 4.0, 0.0] represents pre-computed z-scores.
    // Re-applying zscore_mean/std would invert ranking to cpu_util=4.0, mem_pressure=2.0, queue_depth=0.0.
    let resp = runtime
        .evaluate_machine_infer(&MachineInferRequest {
            model: "ttm-pre-norm".to_owned(),
            features: vec![0.0, 4.0, 0.0],
            rows: None,
            num_rows: Some(1),
            num_cols: Some(3),
            clip_sigma: None,
            timestamp_ms: Some(10_000),
            already_normalized: true,
        })
        .expect("pre-normalized inference succeeds");

    let evt = resp.emitted_event.expect("system event emitted");
    assert_eq!(evt.top_causal_features.len(), 3);
    assert_eq!(evt.top_causal_features[0].0, "queue_depth");
    assert!(
        (evt.top_causal_features[0].1 - 4.0).abs() < 1e-6,
        "expected queue_depth magnitude 4.0 without double normalization, got {}",
        evt.top_causal_features[0].1
    );
    assert!(
        evt.top_causal_features[1].1.abs() < 1e-6 && evt.top_causal_features[2].1.abs() < 1e-6,
        "non-elevated pre-normalized features must remain 0.0: {:?}",
        evt.top_causal_features
    );
}

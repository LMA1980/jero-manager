//! Milestone M10 End-to-End Integration & Adversarial Protocol Tests (`jero-manager`).
//!
//! Covers M10 Epic `#100`:
//! - Feature `#101` (Task `#102`, `#103–#105`): `BUILTIN_RECIPES` registration (`nomic-bert`, `bge`,
//!   `jina-bert`, `qwen3-embedding`, `clip`, `siglip`, `whisper`, rerankers) & `ModelFormat::SafeTensors` admission/inspection
//! - Feature `#106` (Task `#107`, `#108–#111`): `POST /v1/embeddings` (`float` & `base64`, Matryoshka truncation) & `POST /v1/rerank`
//! - Feature `#112` (Task `#113`, `#114–#116`): `POST /v1/audio/transcriptions` (multipart WAV + JSON, 16 kHz resampler, fail-closed `400`/`413`/`415`/`422`)
//! - Feature `#112` (Task `#117`, `#118–#120`): OpenAI & Claude base64 VLM `mmproj` soft-prompt conditioning, SSRF fail-closed rejection, and zero-copy SHM `FeatureInputSlotHeader` staging

#![allow(
    clippy::pedantic,
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::similar_names,
    clippy::useless_vec
)]

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use flate2::Compression;
use flate2::write::ZlibEncoder;
use jero_core::affinity::ControlPlaneCapacityConfig;
use jero_core::ipc::shm::SharedMemoryArena;
use jero_core::profile::{ComputePool, PriorityClass, TuningProfile};
use jero_manager::domain::{MockOsAffinityBackend, OsAffinityBackend};
use jero_manager::ingestion::{ModelFormat, SafeTensorsHeaderPrefix, StagingPipeline};
use jero_manager::persistence::DuckDbPersistenceStore;
use jero_manager::protocol::{
    AudioTranscriptionResponse, ClaudeContent, ClaudeContentBlock, ClaudeMessage,
    ClaudeMessageRequest, ClaudeMessageResponse, HttpRequest, HttpResponseBody, OpenAiChatResponse,
    OpenAiEmbeddingResponse, OpenAiEmbeddingVector, ProtocolServer, ProtocolServerConfig,
    RerankResponse,
};
use jero_manager::runtime::{AdminCommand, ControlPlaneRuntime};
use jero_manager::store::{
    CapabilityFilter, ContentAddressedStore, ModelManifest, Sha256, inspect_safetensors_bytes,
    inspect_weight_file,
};
use jero_manager::{SOFT_PROMPT_SCHEMA_HASH, stage_soft_prompt_feature_slot};
use serde_json::json;

static TEST_DIR_COUNTER: AtomicU64 = AtomicU64::new(1);

struct TempDirGuard {
    path: PathBuf,
}

impl TempDirGuard {
    fn new(prefix: &str) -> Self {
        let seq = TEST_DIR_COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir =
            std::env::temp_dir().join(format!("jero_m10_{prefix}_{}_{seq}", std::process::id()));
        fs::create_dir_all(&dir).expect("create temp test dir");
        Self { path: dir }
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

fn build_test_runtime() -> Arc<ControlPlaneRuntime> {
    let capacity = ControlPlaneCapacityConfig::default()
        .resolve_default()
        .expect("valid topology");
    let mock_affinity: Arc<dyn OsAffinityBackend> = Arc::new(MockOsAffinityBackend::default());
    Arc::new(
        ControlPlaneRuntime::with_affinity_backend(capacity, mock_affinity)
            .expect("create ControlPlaneRuntime"),
    )
}

fn response_body_bytes(resp: jero_manager::protocol::HttpResponse) -> (u16, String, Vec<u8>) {
    let status = resp.status;
    let content_type = resp
        .headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("content-type"))
        .map(|(_, v)| v.clone())
        .unwrap_or_default();
    let bytes = match resp.body {
        HttpResponseBody::Bytes(b) => b,
        HttpResponseBody::Stream(_) => panic!("unexpected streaming response body"),
    };
    (status, content_type, bytes)
}

fn build_test_safetensors(
    arch: &str,
    tensor_name: &str,
    dtype: &str,
    shape: &[usize],
    payload: &[u8],
) -> Vec<u8> {
    let header_obj = json!({
        "__metadata__": {
            "architecture": arch
        },
        tensor_name: {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [0, payload.len()]
        }
    });
    let mut header_bytes = serde_json::to_vec(&header_obj).unwrap();
    while !header_bytes.len().is_multiple_of(8) {
        header_bytes.push(b' ');
    }
    let header_len = header_bytes.len() as u64;
    let mut out = Vec::with_capacity(8 + header_bytes.len() + payload.len());
    out.extend_from_slice(&header_len.to_le_bytes());
    out.extend_from_slice(&header_bytes);
    out.extend_from_slice(payload);
    out
}

fn build_test_wav_pcm16(sample_rate: u32, channels: u16, samples_per_channel: usize) -> Vec<u8> {
    let bits_per_sample: u16 = 16;
    let block_align = channels * (bits_per_sample / 8);
    let byte_rate = sample_rate * u32::from(block_align);
    let data_bytes_len = (samples_per_channel * usize::from(block_align)) as u32;
    let riff_len = 36 + data_bytes_len;

    let mut out = Vec::with_capacity((riff_len + 8) as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&riff_len.to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits_per_sample.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_bytes_len.to_le_bytes());

    for i in 0..samples_per_channel {
        let phase = 2.0_f32 * std::f32::consts::PI * 440.0_f32 * (i as f32) / (sample_rate as f32);
        let s = (phase.sin() * 12_000.0_f32) as i16;
        for _ in 0..channels {
            out.extend_from_slice(&s.to_le_bytes());
        }
    }
    out
}

fn build_test_wav_f32(sample_rate: u32, channels: u16, samples: &[f32]) -> Vec<u8> {
    let block_align = channels * 4;
    let byte_rate = sample_rate * u32::from(block_align);
    let frames = samples.len() / usize::from(channels);
    let data_bytes_len = (frames * usize::from(block_align)) as u32;
    let riff_len = 36 + data_bytes_len;

    let mut out = Vec::with_capacity((riff_len + 8) as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&riff_len.to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&3u16.to_le_bytes());
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&32u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_bytes_len.to_le_bytes());
    for &s in samples {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

fn png_chunk(tag: &[u8; 4], data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(12 + data.len());
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(tag);
    out.extend_from_slice(data);
    out.extend_from_slice(&0u32.to_be_bytes());
    out
}

fn build_test_png_rgb(width: u32, height: u32) -> Vec<u8> {
    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.push(8);
    ihdr.push(2);
    ihdr.push(0);
    ihdr.push(0);
    ihdr.push(0);
    png.extend_from_slice(&png_chunk(b"IHDR", &ihdr));

    let mut raw_scanlines = Vec::new();
    for y in 0..height {
        raw_scanlines.push(0);
        for x in 0..width {
            raw_scanlines.push(((x * 37) % 256) as u8);
            raw_scanlines.push(((y * 53) % 256) as u8);
            raw_scanlines.push((((x + y) * 29) % 256) as u8);
        }
    }
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::fast());
    encoder.write_all(&raw_scanlines).unwrap();
    let compressed = encoder.finish().unwrap();
    png.extend_from_slice(&png_chunk(b"IDAT", &compressed));
    png.extend_from_slice(&png_chunk(b"IEND", &[]));
    png
}

fn build_test_ppm_p6(width: u32, height: u32) -> Vec<u8> {
    let mut out = format!("P6\n{width} {height}\n255\n").into_bytes();
    for y in 0..height {
        for x in 0..width {
            out.push(((x * 19) % 256) as u8);
            out.push(((y * 41) % 256) as u8);
            out.push(200);
        }
    }
    out
}

fn build_test_bmp_24(width: u32, height: u32) -> Vec<u8> {
    let row_unpadded = (width as usize) * 3;
    let row_stride = (row_unpadded + 3) & !3;
    let pixel_size = row_stride * (height as usize);
    let file_size = 54 + pixel_size;

    let mut out = vec![0u8; file_size];
    out[0..2].copy_from_slice(b"BM");
    out[2..6].copy_from_slice(&(file_size as u32).to_le_bytes());
    out[10..14].copy_from_slice(&54u32.to_le_bytes());
    out[14..18].copy_from_slice(&40u32.to_le_bytes());
    out[18..22].copy_from_slice(&(width as i32).to_le_bytes());
    out[22..26].copy_from_slice(&(height as i32).to_le_bytes());
    out[26..28].copy_from_slice(&1u16.to_le_bytes());
    out[28..30].copy_from_slice(&24u16.to_le_bytes());
    for y in 0..height as usize {
        for x in 0..width as usize {
            let p = 54 + y * row_stride + x * 3;
            out[p] = 32;
            out[p + 1] = 160;
            out[p + 2] = 240;
        }
    }
    out
}

fn build_test_raw_rgb(width: u16, height: u16) -> Vec<u8> {
    let mut out = Vec::with_capacity(8 + usize::from(width) * usize::from(height) * 3);
    out.extend_from_slice(b"RGB\0");
    out.extend_from_slice(&width.to_le_bytes());
    out.extend_from_slice(&height.to_le_bytes());
    for _ in 0..usize::from(width) * usize::from(height) {
        out.extend_from_slice(&[255, 128, 64]);
    }
    out
}

fn build_multipart_wav_body(
    boundary: &str,
    model: &str,
    wav_bytes: &[u8],
    extra_fields: &[(&str, &str)],
) -> Vec<u8> {
    let mut body = Vec::new();
    body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    body.extend_from_slice(
        b"Content-Disposition: form-data; name=\"file\"; filename=\"speech.wav\"\r\nContent-Type: audio/wav\r\n\r\n",
    );
    body.extend_from_slice(wav_bytes);
    body.extend_from_slice(b"\r\n");

    body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    body.extend_from_slice(b"Content-Disposition: form-data; name=\"model\"\r\n\r\n");
    body.extend_from_slice(model.as_bytes());
    body.extend_from_slice(b"\r\n");

    for &(k, v) in extra_fields {
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(
            format!("Content-Disposition: form-data; name=\"{k}\"\r\n\r\n").as_bytes(),
        );
        body.extend_from_slice(v.as_bytes());
        body.extend_from_slice(b"\r\n");
    }

    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    body
}

#[test]
fn test_m10_multimodal_recipes_and_safetensors_admission() {
    let required = [
        "nomic-bert",
        "bge",
        "jina-bert",
        "qwen3-embedding",
        "clip",
        "siglip",
        "whisper",
        "cross-encoder",
        "reranker",
        "bge-reranker",
        "jina-reranker",
        "modernbert-reranker",
    ];
    let recipes = CapabilityFilter::supported_recipes();
    for arch in required {
        assert!(
            CapabilityFilter::is_supported(arch, "1.0.0"),
            "missing recipe for {arch}@1.0.0"
        );
        assert!(recipes.contains(&format!("{arch}@1.0.0")));

        let manifest = ModelManifest {
            schema_version: 1,
            format: "safetensors".to_owned(),
            architecture_id: arch.to_owned(),
            recipe_version: "1.0.0".to_owned(),
            quantization: "BF16".to_owned(),
            context_length: 8192,
            license: Some("Apache-2.0".to_owned()),
            origin: Some("hf:test/m10".to_owned()),
            weight_digest: "b".repeat(64),
            weight_size_bytes: 4096,
            created_at: 1_700_000_000,
        };
        let suit = CapabilityFilter::evaluate_candidate(&manifest);
        assert!(
            suit.executable,
            "SafeTensors candidate for {arch} should be executable: {:?}",
            suit.reason
        );
        assert!(CapabilityFilter::validate_model_manifest(&manifest).is_ok());
    }

    // Unsupported format must fail closed in both evaluate_candidate and validate_model_manifest
    let bad_format = ModelManifest {
        schema_version: 1,
        format: "onnx".to_owned(),
        architecture_id: "bge".to_owned(),
        recipe_version: "1.0.0".to_owned(),
        quantization: "F16".to_owned(),
        context_length: 512,
        license: None,
        origin: None,
        weight_digest: "c".repeat(64),
        weight_size_bytes: 1024,
        created_at: 1,
    };
    let eval_bad = CapabilityFilter::evaluate_candidate(&bad_format);
    assert!(!eval_bad.executable);
    assert!(
        eval_bad
            .reason
            .unwrap_or_default()
            .contains("unsupported model format")
    );
    assert!(CapabilityFilter::validate_model_manifest(&bad_format).is_err());

    // Real SafeTensors binary creation, staging header verification, CAS promotion & deep tensor inspection
    let dir = TempDirGuard::new("safetensors_admit");
    let catalog = DuckDbPersistenceStore::open_in_memory().expect("open duckdb catalog");
    let store = ContentAddressedStore::open(dir.path(), Arc::new(catalog)).expect("open CAS");

    let f32_weights = vec![0.25_f32; 8];
    let raw_bytes: Vec<u8> = f32_weights.iter().flat_map(|v| v.to_le_bytes()).collect();
    let st_bytes =
        build_test_safetensors("nomic-bert", "encoder.weight", "F32", &[2, 4], &raw_bytes);

    let prefix_ok = SafeTensorsHeaderPrefix {
        header_len: u64::from_le_bytes(st_bytes[0..8].try_into().unwrap()),
    };
    assert!(prefix_ok.header_len > 0);

    let (staging_path, mut file) = store.create_staging_file().expect("staging file");
    file.write_all(&st_bytes).expect("write SafeTensors");
    drop(file);
    StagingPipeline::verify_format_header(&staging_path, &ModelFormat::SafeTensors)
        .expect("staging prefix verification");

    let digest = Sha256::digest(&st_bytes);
    let blob_rec = store
        .promote_staged_blob(&staging_path, &digest, "safetensors")
        .expect("promote SafeTensors blob");
    let manifest = ModelManifest {
        schema_version: 1,
        format: "safetensors".to_owned(),
        architecture_id: "nomic-bert".to_owned(),
        recipe_version: "1.0.0".to_owned(),
        quantization: "F32".to_owned(),
        context_length: 2048,
        license: Some("Apache-2.0".to_owned()),
        origin: None,
        weight_digest: digest,
        weight_size_bytes: st_bytes.len() as u64,
        created_at: 2,
    };

    let (report, suit) =
        ContentAddressedStore::validate_safetensors_weights_and_admit(&blob_rec, &manifest)
            .expect("deep SafeTensors inspection and admission");
    assert!(suit.executable);
    assert_eq!(report.format, "safetensors");
    assert_eq!(report.architecture.as_deref(), Some("nomic-bert"));
    assert_eq!(report.tensor_count, 1);
    assert_eq!(report.total_weight_bytes, 32);

    // Unsupported SafeTensors dtype "I8" fails closed under deep inspection
    let bad_st = build_test_safetensors("nomic-bert", "bad.weight", "I8", &[4], &[1, 2, 3, 4]);
    assert!(inspect_safetensors_bytes(&bad_st).is_err());

    let corrupt_path = dir.path().join("corrupt.safetensors");
    fs::write(&corrupt_path, b"\x04\x00\x00\x00\x00\x00\x00\x00{bad").unwrap();
    assert!(inspect_weight_file(&corrupt_path, "safetensors").is_err());
}

#[tokio::test]
async fn test_m10_openai_embeddings_float_base64_matryoshka_and_adversarial() {
    let runtime = build_test_runtime();
    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: "nomic-embed-text".to_owned(),
            weights_path: PathBuf::from("/models/nomic-bert.safetensors"),
            cluster: "x100".to_owned(),
        })
        .expect("register embedding model");
    let server = ProtocolServer::new(ProtocolServerConfig::default(), Arc::clone(&runtime));

    // 1. Batch float embeddings + semantic cosine similarity check
    let float_req = HttpRequest::new(
        "POST",
        "/v1/embeddings",
        serde_json::to_vec(&json!({
            "model": "nomic-embed-text",
            "input": [
                "rust rvv vector kernel runtime",
                "rust vector kernel scheduling",
                "chocolate strawberry cake frosting"
            ],
            "encoding_format": "float"
        }))
        .unwrap(),
    );
    let (status, _, body) = response_body_bytes(server.route_request(float_req).await);
    assert_eq!(
        status,
        200,
        "unexpected error: {}",
        String::from_utf8_lossy(&body)
    );
    let float_resp: OpenAiEmbeddingResponse =
        serde_json::from_slice(&body).expect("parse float resp");
    assert_eq!(float_resp.object, "list");
    assert_eq!(float_resp.data.len(), 3);
    assert!(float_resp.usage.prompt_tokens > 0);

    let mut vecs = Vec::new();
    for (idx, item) in float_resp.data.iter().enumerate() {
        assert_eq!(item.index, idx as u32);
        assert_eq!(item.object, "embedding");
        let OpenAiEmbeddingVector::Float(ref v) = item.embedding else {
            panic!("expected Float vector");
        };
        assert_eq!(v.len(), 768);
        let norm_sq: f32 = v.iter().map(|&x| x * x).sum();
        assert!(
            (norm_sq - 1.0).abs() < 1e-3,
            "unit L2 norm violated: {norm_sq}"
        );
        vecs.push(v.clone());
    }
    let sim_close: f32 = vecs[0].iter().zip(&vecs[1]).map(|(&a, &b)| a * b).sum();
    let sim_far: f32 = vecs[0].iter().zip(&vecs[2]).map(|(&a, &b)| a * b).sum();
    assert!(
        sim_close > sim_far,
        "sim_close ({sim_close}) <= sim_far ({sim_far})"
    );

    // 2. Matryoshka truncation + base64 LE f32 encoding parity via `/embeddings` alias
    let trunc_float_req = HttpRequest::new(
        "POST",
        "/embeddings",
        serde_json::to_vec(&json!({
            "model": "nomic-embed-text",
            "input": "rust rvv vector kernel runtime",
            "dimensions": 128,
            "encoding_format": "float"
        }))
        .unwrap(),
    );
    let (s1, _, b1) = response_body_bytes(server.route_request(trunc_float_req).await);
    assert_eq!(s1, 200);
    let r1: OpenAiEmbeddingResponse = serde_json::from_slice(&b1).unwrap();
    let OpenAiEmbeddingVector::Float(ref expected_128) = r1.data[0].embedding else {
        panic!("expected float");
    };
    assert_eq!(expected_128.len(), 128);
    let norm128: f32 = expected_128.iter().map(|&x| x * x).sum();
    assert!((norm128 - 1.0).abs() < 1e-3);

    let trunc_b64_req = HttpRequest::new(
        "POST",
        "/v1/embeddings",
        serde_json::to_vec(&json!({
            "model": "nomic-embed-text",
            "input": "rust rvv vector kernel runtime",
            "dimensions": 128,
            "encoding_format": "base64"
        }))
        .unwrap(),
    );
    let (s2, _, b2) = response_body_bytes(server.route_request(trunc_b64_req).await);
    assert_eq!(s2, 200);
    let r2: OpenAiEmbeddingResponse = serde_json::from_slice(&b2).unwrap();
    let OpenAiEmbeddingVector::Base64(ref b64_str) = r2.data[0].embedding else {
        panic!("expected base64");
    };
    let raw_le = BASE64_STANDARD.decode(b64_str).expect("valid base64");
    assert_eq!(raw_le.len(), 128 * 4);
    let decoded_f32: Vec<f32> = raw_le
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&c| f32::from_le_bytes(c))
        .collect();
    assert_eq!(&decoded_f32, expected_128);

    // 3. Token IDs array input
    let tok_req = HttpRequest::new(
        "POST",
        "/v1/embeddings",
        serde_json::to_vec(&json!({
            "model": "nomic-embed-text",
            "input": [[11, 22, 33], [44, 55]],
            "dimensions": 64
        }))
        .unwrap(),
    );
    let (s_tok, _, b_tok) = response_body_bytes(server.route_request(tok_req).await);
    assert_eq!(s_tok, 200);
    let tok_resp: OpenAiEmbeddingResponse = serde_json::from_slice(&b_tok).unwrap();
    assert_eq!(tok_resp.usage.prompt_tokens, 5);
    assert_eq!(tok_resp.data.len(), 2);

    // 4. Adversarial HTTP error contracts: 400, 404, 405, 413, 422, 503
    let (s405, _, _) = response_body_bytes(
        server
            .route_request(HttpRequest::new("GET", "/v1/embeddings", Vec::new()))
            .await,
    );
    assert_eq!(s405, 405);

    let (s404, _, _) = response_body_bytes(
        server
            .route_request(HttpRequest::new(
                "POST",
                "/v1/embeddings",
                br#"{"model":"missing","input":"hi"}"#.to_vec(),
            ))
            .await,
    );
    assert_eq!(s404, 404);

    let (s400, _, _) = response_body_bytes(
        server
            .route_request(HttpRequest::new(
                "POST",
                "/v1/embeddings",
                br#"{"model":"nomic-embed-text","input":[]}"#.to_vec(),
            ))
            .await,
    );
    assert_eq!(s400, 400);

    let (s422, _, _) = response_body_bytes(
        server
            .route_request(HttpRequest::new(
                "POST",
                "/v1/embeddings",
                br#"{"model":"nomic-embed-text","input":"hi","dimensions":999999}"#.to_vec(),
            ))
            .await,
    );
    assert_eq!(s422, 422);

    let big_str = "x".repeat(300 * 1024);
    let (s413, _, _) = response_body_bytes(
        server
            .route_request(HttpRequest::new(
                "POST",
                "/v1/embeddings",
                serde_json::to_vec(&json!({ "model": "nomic-embed-text", "input": big_str }))
                    .unwrap(),
            ))
            .await,
    );
    assert_eq!(s413, 413);

    runtime.update_worker_health("nomic-embed-text", false, 1);
    let (s503, _, _) = response_body_bytes(
        server
            .route_request(HttpRequest::new(
                "POST",
                "/v1/embeddings",
                br#"{"model":"nomic-embed-text","input":"hi"}"#.to_vec(),
            ))
            .await,
    );
    assert_eq!(s503, 503);
}

#[tokio::test]
async fn test_m10_rerank_endpoint_sorting_top_n_and_errors() {
    let runtime = build_test_runtime();
    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: "bge-reranker-v2".to_owned(),
            weights_path: PathBuf::from("/models/bge-reranker.safetensors"),
            cluster: "x100".to_owned(),
        })
        .expect("register reranker model");
    let server = ProtocolServer::new(ProtocolServerConfig::default(), Arc::clone(&runtime));

    let req = HttpRequest::new(
        "POST",
        "/v1/rerank",
        serde_json::to_vec(&json!({
            "model": "bge-reranker-v2",
            "query": "spacemit riscv vector matrix kernel",
            "documents": [
                "baking sourdough bread overnight",
                "spacemit k3 riscv vector matrix execution engine kernel",
                { "text": "riscv vector intrinsics optimization" }
            ],
            "top_n": 2,
            "return_documents": true
        }))
        .unwrap(),
    );

    let (status, _, body) = response_body_bytes(server.route_request(req).await);
    assert_eq!(status, 200);
    let resp: RerankResponse = serde_json::from_slice(&body).expect("parse rerank response");
    assert_eq!(resp.object, "list");
    assert_eq!(resp.results.len(), 2);
    assert_eq!(resp.results[0].index, 1);
    assert!(resp.results[0].relevance_score >= resp.results[1].relevance_score);
    assert!(resp.results[0].relevance_score >= 0.0 && resp.results[0].relevance_score <= 1.0);
    assert!(resp.results[0].document.is_some());

    // top_n == 0 -> 422, empty documents -> 400
    let (s422, _, _) = response_body_bytes(
        server
            .route_request(HttpRequest::new(
                "POST",
                "/rerank",
                serde_json::to_vec(&json!({
                    "model": "bge-reranker-v2",
                    "query": "hello",
                    "documents": ["world"],
                    "top_n": 0
                }))
                .unwrap(),
            ))
            .await,
    );
    assert_eq!(s422, 422);

    let (s400, _, _) = response_body_bytes(
        server
            .route_request(HttpRequest::new(
                "POST",
                "/v1/rerank",
                serde_json::to_vec(&json!({
                    "model": "bge-reranker-v2",
                    "query": "hello",
                    "documents": []
                }))
                .unwrap(),
            ))
            .await,
    );
    assert_eq!(s400, 400);
}

#[tokio::test]
async fn test_m10_audio_transcriptions_multipart_json_resample_and_adversarial() {
    let runtime = build_test_runtime();
    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: "whisper-tiny".to_owned(),
            weights_path: PathBuf::from("/models/whisper-tiny.safetensors"),
            cluster: "x100".to_owned(),
        })
        .expect("register whisper model");
    let server = ProtocolServer::new(ProtocolServerConfig::default(), Arc::clone(&runtime));

    // 1. Multipart WAV (44.1 kHz stereo PCM16 -> 16 kHz mono resampled) with verbose_json
    let boundary = "----K3WhisperBoundary99";
    let stereo_44k = build_test_wav_pcm16(44_100, 2, 44_100 / 2);
    let mp_body = build_multipart_wav_body(
        boundary,
        "whisper-tiny",
        &stereo_44k,
        &[("language", "en"), ("response_format", "verbose_json")],
    );
    let mp_req = HttpRequest::new("POST", "/v1/audio/transcriptions", mp_body).with_header(
        "Content-Type",
        format!("multipart/form-data; boundary={boundary}"),
    );

    let (st_mp, ct_mp, b_mp) = response_body_bytes(server.route_request(mp_req).await);
    assert_eq!(st_mp, 200, "body: {}", String::from_utf8_lossy(&b_mp));
    assert!(ct_mp.contains("application/json"));
    let mp_resp: AudioTranscriptionResponse = serde_json::from_slice(&b_mp).unwrap();
    assert!(!mp_resp.text.is_empty());
    assert_eq!(mp_resp.task.as_deref(), Some("transcribe"));
    assert_eq!(mp_resp.language.as_deref(), Some("en"));
    assert!(mp_resp.duration.unwrap_or(0.0) > 0.45);
    assert!(!mp_resp.segments.unwrap_or_default().is_empty());

    // 2. JSON audio_base64 (IEEE_FLOAT32 WAV @ 24 kHz) + response_format="text"
    let float_samples: Vec<f32> = (0..6000)
        .map(|i| (2.0 * std::f32::consts::PI * 300.0 * (i as f32) / 24_000.0).sin() * 0.5)
        .collect();
    let wav_f32 = build_test_wav_f32(24_000, 1, &float_samples);
    let json_req = HttpRequest::new(
        "POST",
        "/audio/transcriptions",
        serde_json::to_vec(&json!({
            "model": "whisper-tiny",
            "audio_base64": BASE64_STANDARD.encode(&wav_f32),
            "response_format": "text"
        }))
        .unwrap(),
    )
    .with_header("Content-Type", "application/json");
    let (st_txt, ct_txt, b_txt) = response_body_bytes(server.route_request(json_req).await);
    assert_eq!(st_txt, 200);
    assert!(ct_txt.contains("text/plain"));
    assert!(!String::from_utf8_lossy(&b_txt).trim().is_empty());

    // 3. Adversarial checks:
    // 415 Unsupported Media Type
    let (s415, _, _) = response_body_bytes(
        server
            .route_request(
                HttpRequest::new("POST", "/v1/audio/transcriptions", b"hello".to_vec())
                    .with_header("Content-Type", "text/plain"),
            )
            .await,
    );
    assert_eq!(s415, 415);

    // 400 Corrupt WAV / non-finite IEEE float sample
    let mut nan_wav = build_test_wav_f32(16_000, 1, &[0.1, 0.2, 0.3, 0.4]);
    let last_offset = nan_wav.len() - 4;
    nan_wav[last_offset..].copy_from_slice(&f32::NAN.to_le_bytes());
    let (s400_nan, _, _) = response_body_bytes(
        server
            .route_request(
                HttpRequest::new(
                    "POST",
                    "/v1/audio/transcriptions",
                    serde_json::to_vec(&json!({
                        "model": "whisper-tiny",
                        "audio_base64": BASE64_STANDARD.encode(&nan_wav)
                    }))
                    .unwrap(),
                )
                .with_header("Content-Type", "application/json"),
            )
            .await,
    );
    assert_eq!(s400_nan, 400);

    // 422 Unsupported channel count (>2) in WAV header
    let bad_ch_wav = build_test_wav_pcm16(16_000, 6, 160);
    let (s422_ch, _, _) = response_body_bytes(
        server
            .route_request(
                HttpRequest::new(
                    "POST",
                    "/v1/audio/transcriptions",
                    build_multipart_wav_body(boundary, "whisper-tiny", &bad_ch_wav, &[]),
                )
                .with_header(
                    "Content-Type",
                    format!("multipart/form-data; boundary={boundary}"),
                ),
            )
            .await,
    );
    assert_eq!(s422_ch, 422);

    // 422 Audio exceeding 30s single-window cap
    let long_pcm = vec![0.05_f32; 16_000 * 31];
    let (s422_dur, _, _) = response_body_bytes(
        server
            .route_request(
                HttpRequest::new(
                    "POST",
                    "/v1/audio/transcriptions",
                    serde_json::to_vec(&json!({
                        "model": "whisper-tiny",
                        "pcm_f32": long_pcm,
                        "sample_rate": 16_000
                    }))
                    .unwrap(),
                )
                .with_header("Content-Type", "application/json"),
            )
            .await,
    );
    assert_eq!(s422_dur, 422);
}

#[tokio::test]
async fn test_m10_vision_mmproj_openai_claude_ssrf_and_shm_staging() {
    let runtime = build_test_runtime();
    let vlm_tuning = TuningProfile::conservative_default();
    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: "gemma3-4b-vlm".to_owned(),
            weights_path: PathBuf::from("/models/gemma3-4b-vlm.safetensors"),
            cluster: "x100".to_owned(),
        })
        .expect("register VLM model");
    runtime
        .execute_admin(AdminCommand::ApplyTuningProfile {
            model_id: "gemma3-4b-vlm".to_owned(),
            profile: vlm_tuning,
        })
        .expect("apply VLM tuning");

    let mut tiny_ctx_tuning = TuningProfile::conservative_default();
    tiny_ctx_tuning.apply_operator_tuning(
        ComputePool::X100,
        0,
        2,
        PriorityClass::Latency,
        16,
        false,
        false,
    );
    runtime
        .execute_admin(AdminCommand::RegisterModel {
            model_id: "clip-tiny-ctx".to_owned(),
            weights_path: PathBuf::from("/models/clip-tiny.safetensors"),
            cluster: "x100".to_owned(),
        })
        .expect("register tiny ctx model");
    runtime
        .execute_admin(AdminCommand::ApplyTuningProfile {
            model_id: "clip-tiny-ctx".to_owned(),
            profile: tiny_ctx_tuning,
        })
        .expect("apply tiny ctx tuning");

    let server = ProtocolServer::new(ProtocolServerConfig::default(), Arc::clone(&runtime));

    let png_data_uri = format!(
        "data:image/png;base64,{}",
        BASE64_STANDARD.encode(build_test_png_rgb(8, 8))
    );
    let ppm_data_uri = format!(
        "data:image/ppm;base64,{}",
        BASE64_STANDARD.encode(build_test_ppm_p6(4, 4))
    );
    let bmp_data_uri = format!(
        "data:image/bmp;base64,{}",
        BASE64_STANDARD.encode(build_test_bmp_24(4, 4))
    );

    // 1. OpenAI chat completions with base64 PNG + PPM data URIs
    let openai_payload = json!({
        "model": "gemma3-4b-vlm",
        "messages": [{
            "role": "user",
            "content": [
                { "type": "text", "text": "Describe these two test diagrams." },
                { "type": "image_url", "image_url": { "url": png_data_uri } },
                { "type": "image_url", "image_url": { "url": ppm_data_uri } }
            ]
        }],
        "max_tokens": 64,
        "stream": false
    });

    let (st_oai, _, body_oai) = response_body_bytes(
        server
            .route_request(HttpRequest::new(
                "POST",
                "/v1/chat/completions",
                serde_json::to_vec(&openai_payload).unwrap(),
            ))
            .await,
    );
    assert_eq!(st_oai, 200, "{}", String::from_utf8_lossy(&body_oai));
    let oai_resp: OpenAiChatResponse = serde_json::from_slice(&body_oai).unwrap();
    let content_str = oai_resp.choices[0]
        .message
        .content
        .as_deref()
        .unwrap_or_default();
    assert!(content_str.contains("vlm_mmproj images=2"));
    assert!(
        oai_resp.usage.prompt_tokens > 90,
        "expected soft prompt tokens counted in usage: {}",
        oai_resp.usage.prompt_tokens
    );

    // 2. Claude POST /v1/messages with base64 BMP & 1x1 PNG blocks
    let claude_vlm_req = ClaudeMessageRequest {
        model: "gemma3-4b-vlm".to_owned(),
        system: None,
        messages: vec![ClaudeMessage {
            role: "user".to_owned(),
            content: ClaudeContent::Blocks(vec![
                ClaudeContentBlock::text("Inspect diagram:"),
                ClaudeContentBlock {
                    r#type: "image".to_owned(),
                    text: None,
                    thinking: None,
                    id: None,
                    name: None,
                    input: None,
                    tool_use_id: None,
                    content: None,
                    is_error: None,
                    source: Some(json!({
                        "type": "base64",
                        "media_type": "image/bmp",
                        "data": bmp_data_uri.split_once(";base64,").unwrap().1
                    })),
                },
                ClaudeContentBlock {
                    r#type: "image".to_owned(),
                    text: None,
                    thinking: None,
                    id: None,
                    name: None,
                    input: None,
                    tool_use_id: None,
                    content: None,
                    is_error: None,
                    source: Some(json!({
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY44YAAAAASUVORK5CYII="
                    })),
                },
            ]),
        }],
        max_tokens: 64,
        thinking: None,
        tools: None,
        tool_choice: None,
        stop_sequences: None,
        stream: Some(false),
        temperature: None,
        top_p: None,
    };

    let (st_cl, _, body_cl) = response_body_bytes(
        server
            .route_request(HttpRequest::new(
                "POST",
                "/v1/messages",
                serde_json::to_vec(&claude_vlm_req).unwrap(),
            ))
            .await,
    );
    assert_eq!(st_cl, 200, "{}", String::from_utf8_lossy(&body_cl));
    let cl_resp: ClaudeMessageResponse = serde_json::from_slice(&body_cl).unwrap();
    assert!(cl_resp.usage.input_tokens > 90);

    // 3. Soft-prompt token count exceeding n_ctx triggers 400 context_length_exceeded
    let mut overflow_payload = openai_payload.clone();
    overflow_payload["model"] = json!("clip-tiny-ctx");
    let (st_ctx, _, body_ctx) = response_body_bytes(
        server
            .route_request(HttpRequest::new(
                "POST",
                "/v1/chat/completions",
                serde_json::to_vec(&overflow_payload).unwrap(),
            ))
            .await,
    );
    assert_eq!(st_ctx, 400);
    assert!(String::from_utf8_lossy(&body_ctx).contains("context_length_exceeded"));

    // 4. SSRF protection: remote URLs rejected fail closed (400) on BOTH OpenAI & Claude
    let oai_ssrf = json!({
        "model": "gemma3-4b-vlm",
        "messages": [{
            "role": "user",
            "content": [
                { "type": "text", "text": "fetch metadata" },
                { "type": "image_url", "image_url": { "url": "http://169.254.169.254/latest/meta-data" } }
            ]
        }]
    });
    let (st_ssrf1, _, _) = response_body_bytes(
        server
            .route_request(HttpRequest::new(
                "POST",
                "/v1/chat/completions",
                serde_json::to_vec(&oai_ssrf).unwrap(),
            ))
            .await,
    );
    assert_eq!(st_ssrf1, 400);

    let claude_ssrf = json!({
        "model": "gemma3-4b-vlm",
        "max_tokens": 64,
        "messages": [{
            "role": "user",
            "content": [{
                "type": "image",
                "source": { "type": "url", "url": "https://example.org/secret.png" }
            }]
        }]
    });
    let (st_ssrf2, _, _) = response_body_bytes(
        server
            .route_request(HttpRequest::new(
                "POST",
                "/v1/messages",
                serde_json::to_vec(&claude_ssrf).unwrap(),
            ))
            .await,
    );
    assert_eq!(st_ssrf2, 400);

    // 5. Huge declared PNG header dimensions rejected BEFORE allocation (422)
    let mut bomb_png = b"\x89PNG\r\n\x1a\n".to_vec();
    let mut big_ihdr = Vec::new();
    big_ihdr.extend_from_slice(&16_384u32.to_be_bytes());
    big_ihdr.extend_from_slice(&16_384u32.to_be_bytes());
    big_ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
    bomb_png.extend_from_slice(&png_chunk(b"IHDR", &big_ihdr));
    bomb_png.extend_from_slice(&png_chunk(b"IEND", &[]));
    let bomb_uri = format!("data:image/png;base64,{}", BASE64_STANDARD.encode(bomb_png));
    let (st_bomb, _, _) = response_body_bytes(
        server
            .route_request(HttpRequest::new(
                "POST",
                "/v1/chat/completions",
                serde_json::to_vec(&json!({
                    "model": "gemma3-4b-vlm",
                    "messages": [{
                        "role": "user",
                        "content": [{ "type": "image_url", "image_url": { "url": bomb_uri } }]
                    }]
                }))
                .unwrap(),
            ))
            .await,
    );
    assert_eq!(st_bomb, 422);

    // 6. Real VLM runtime soft_prompt_embeds + SharedMemoryArena FeatureInputSlotHeader staging
    let raw_img_b64 = BASE64_STANDARD.encode(build_test_raw_rgb(4, 4));
    let raw_img = jero_manager::protocol::decode_bounded_image_bytes(
        &BASE64_STANDARD.decode(&raw_img_b64).unwrap(),
        "image/x-raw-rgb",
    )
    .expect("decode raw RGB");
    let bundle = runtime
        .evaluate_vision_soft_prompts("gemma3-4b-vlm", &[raw_img])
        .expect("compute VLM soft prompts");
    assert_eq!(
        bundle.soft_prompt_embeds.len(),
        (bundle.num_tokens as usize) * (bundle.embed_dim as usize)
    );

    let shm_name = format!(
        "jero_m10_vlm_shm_{}_{}",
        std::process::id(),
        TEST_DIR_COUNTER.fetch_add(1, Ordering::SeqCst)
    );
    let mut arena = SharedMemoryArena::create(&shm_name, 64 * 1024).expect("create arena");
    let aligned_offset = 4096usize;
    let written = stage_soft_prompt_feature_slot(
        &mut arena,
        aligned_offset,
        77,
        &bundle.soft_prompt_embeds,
        bundle.num_tokens,
        bundle.embed_dim,
    )
    .expect("stage soft_prompt_embeds into FeatureInputSlotHeader");
    assert!(written > 64);

    let (hdr, roundtrip) = arena
        .read_feature_slot_as_f32(aligned_offset)
        .expect("read back soft prompt feature slot");
    assert_eq!(hdr.schema_hash, SOFT_PROMPT_SCHEMA_HASH);
    assert_eq!(hdr.num_rows, bundle.num_tokens);
    assert_eq!(hdr.num_cols, bundle.embed_dim);
    assert_eq!(roundtrip, bundle.soft_prompt_embeds);
}

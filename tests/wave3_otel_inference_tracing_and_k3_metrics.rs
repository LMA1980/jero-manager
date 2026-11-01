//! Integration tests for Milestone 3 Wave 3 (`#52`–`#55`):
//! OpenTelemetry inference tracing (`OtelSpanEntry`) and `SpacemiT` K3 cluster (`X100`/`A100`)
//! efficiency and bounded `RLIMIT_MEMLOCK` metrics (`OtelMetricEntry`) streamed to `MuObserve-collector`.

use std::sync::Arc;

use k3_core::affinity::ControlPlaneCapacityConfig;
use k3_core::profile::{PriorityClass, ProfileStatus, TuningProfile};
use k3_manager::domain::MockOsAffinityBackend;
use k3_manager::protocol::types::{HttpRequest, HttpResponseBody};
use k3_manager::{
    BOUNDED_MEMLOCK_CEILING_BYTES, CollectorEndpoint, ControlPlaneRuntime,
    MSG_TYPE_ARROW_OTEL_METRICS, MSG_TYPE_ARROW_OTEL_SPANS, MSG_TYPE_WAL_ACK,
    MuObserveTelemetryStreamer, ProtocolServer, ProtocolServerConfig, WalBatchAck,
    decode_metric_batch, decode_span_batch,
};
use muobserve_ipc::{IpcHeader, create_fory_registry, read_frame, write_frame};
use tokio::net::TcpListener;

fn create_test_runtime() -> Arc<ControlPlaneRuntime> {
    let cap = ControlPlaneCapacityConfig::default()
        .resolve_default()
        .expect("resolve default capacity");
    let backend = Arc::new(MockOsAffinityBackend::default());
    let runtime =
        ControlPlaneRuntime::with_affinity_backend(cap, backend).expect("create test runtime");

    // 1. Register an X100 model (Cluster 0, VLEN=256, conservative n_ctx=64)
    let mut x100_tuning = TuningProfile::conservative_default();
    "x100".clone_into(&mut x100_tuning.cluster);
    x100_tuning.x100_cores = 4;
    x100_tuning.a100_cores = 0;
    x100_tuning.n_ctx = 64;
    x100_tuning.unprofiled = false;
    x100_tuning.profile_status = ProfileStatus::Profiled;
    x100_tuning.priority_class = PriorityClass::Latency;
    runtime.record_model_placement_with_tuning(
        "qwen2.5-0.5b-x100".to_owned(),
        "x100".to_owned(),
        Some(2001),
        Some(x100_tuning),
    );

    // 2. Register an A100 model (Cluster 1, VLEN=1024, XSMTVDotII, n_ctx=128)
    let mut a100_tuning = TuningProfile::conservative_default();
    "a100".clone_into(&mut a100_tuning.cluster);
    a100_tuning.x100_cores = 0;
    a100_tuning.a100_cores = 8;
    a100_tuning.n_ctx = 128;
    a100_tuning.unprofiled = false;
    a100_tuning.profile_status = ProfileStatus::Profiled;
    a100_tuning.priority_class = PriorityClass::Throughput;
    runtime.record_model_placement_with_tuning(
        "qwen2.5-3b-a100".to_owned(),
        "a100".to_owned(),
        Some(2002),
        Some(a100_tuning),
    );

    Arc::new(runtime)
}

#[tokio::test]
async fn test_openai_and_claude_emit_w3c_spans_and_k3_cluster_metrics_with_graceful_fallback() {
    let runtime = create_test_runtime();

    // Point collector endpoint to an unreachable closed port to verify graceful non-blocking fallback
    let unreachable_addr: std::net::SocketAddr = "127.0.0.1:9"
        .parse()
        .expect("parse closed discard port address");
    runtime
        .telemetry()
        .set_collector_endpoint(Some(CollectorEndpoint::Tcp(unreachable_addr)));

    let server = ProtocolServer::new(ProtocolServerConfig::default(), Arc::clone(&runtime));

    // Verify local model catalog (/v1/models) remains intact
    let models_req = HttpRequest::new("GET", "/v1/models", Vec::new());
    let models_resp = server.route_request(models_req).await;
    assert_eq!(models_resp.status, 200);

    // 1. Dispatch OpenAI /v1/chat/completions to X100 model with W3C traceparent
    let traceparent_openai = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
    let openai_body = serde_json::json!({
        "model": "qwen2.5-0.5b-x100",
        "messages": [
            {"role": "user", "content": "Hello X100 cluster"}
        ],
        "stream": false
    });
    let openai_req = HttpRequest::new(
        "POST",
        "/v1/chat/completions",
        serde_json::to_vec(&openai_body).expect("serialize openai body"),
    )
    .with_header("traceparent", traceparent_openai);

    let openai_resp = server.route_request(openai_req).await;
    assert_eq!(openai_resp.status, 200);
    assert!(matches!(openai_resp.body, HttpResponseBody::Bytes(_)));

    // 2. Dispatch Claude /v1/messages to A100 model with W3C traceparent
    let traceparent_claude = "00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01";
    let claude_body = serde_json::json!({
        "model": "qwen2.5-3b-a100",
        "max_tokens": 32,
        "messages": [
            {"role": "user", "content": "Hello A100 matrix cluster"}
        ],
        "stream": false
    });
    let claude_req = HttpRequest::new(
        "POST",
        "/v1/messages",
        serde_json::to_vec(&claude_body).expect("serialize claude body"),
    )
    .with_header("traceparent", traceparent_claude);

    let claude_resp = server.route_request(claude_req).await;
    assert_eq!(claude_resp.status, 200);

    // Verify graceful fallback incremented buffered count and preserved all spans & metrics
    assert!(runtime.telemetry().fallback_buffered_count() >= 4);

    let spans = runtime.telemetry().recorded_spans();
    assert_eq!(spans.len(), 8);

    // First 4 spans belong to X100 /v1/chat/completions
    let x100_spans = &spans[0..4];
    assert_eq!(x100_spans[0].span_name, "k3.inference.queue");
    assert_eq!(x100_spans[1].span_name, "k3.inference.kv_alloc");
    assert_eq!(x100_spans[2].span_name, "k3.inference.prefill");
    assert_eq!(x100_spans[3].span_name, "k3.inference.decode");
    for s in x100_spans {
        assert_eq!(s.trace_id, "4bf92f3577b34da6a3ce929d0e0e4736");
        assert_eq!(s.parent_span_id.as_deref(), Some("00f067aa0ba902b7"));
        assert!(s.attributes_json.contains("\"cpu_cluster\":\"X100\""));
        assert!(s.attributes_json.contains("\"vlen_bits\":256"));
    }

    // Next 4 spans belong to A100 /v1/messages
    let a100_spans = &spans[4..8];
    assert_eq!(a100_spans[0].span_name, "k3.inference.queue");
    assert_eq!(a100_spans[1].span_name, "k3.inference.kv_alloc");
    assert_eq!(a100_spans[2].span_name, "k3.inference.prefill");
    assert_eq!(a100_spans[3].span_name, "k3.inference.decode");
    for s in a100_spans {
        assert_eq!(s.trace_id, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
        assert_eq!(s.parent_span_id.as_deref(), Some("bbbbbbbbbbbbbbbb"));
        assert!(s.attributes_json.contains("\"cpu_cluster\":\"A100\""));
        assert!(s.attributes_json.contains("\"vlen_bits\":1024"));
    }

    // Verify hardware-tagged metrics for X100 (VLEN=256, kv_memlock_kb=64) and A100 (VLEN=1024, kv_memlock_kb=128)
    let metrics = runtime.telemetry().recorded_metrics();
    assert_eq!(metrics.len(), 6);

    let x100_metrics = &metrics[0..3];
    assert_eq!(x100_metrics[0].metric_name, "prompt_tok_per_s");
    assert_eq!(x100_metrics[1].metric_name, "decode_tok_per_s");
    assert_eq!(x100_metrics[2].metric_name, "kv_memlock_kb");
    assert!((x100_metrics[2].value_f64 - 64.0).abs() < f64::EPSILON);
    for m in x100_metrics {
        assert_eq!(m.cpu_cluster.as_deref(), Some("X100"));
        assert_eq!(m.vlen_bits, Some(256));
        assert_eq!(m.memlock_used_bytes, Some(64 * 1024));
        assert_eq!(m.memlock_limit_bytes, Some(BOUNDED_MEMLOCK_CEILING_BYTES));
    }

    let a100_metrics = &metrics[3..6];
    assert_eq!(a100_metrics[0].metric_name, "prompt_tok_per_s");
    assert_eq!(a100_metrics[1].metric_name, "decode_tok_per_s");
    assert_eq!(a100_metrics[2].metric_name, "kv_memlock_kb");
    assert!((a100_metrics[2].value_f64 - 128.0).abs() < f64::EPSILON);
    for m in a100_metrics {
        assert_eq!(m.cpu_cluster.as_deref(), Some("A100"));
        assert_eq!(m.vlen_bits, Some(1024));
        assert_eq!(m.memlock_used_bytes, Some(128 * 1024));
        assert_eq!(m.memlock_limit_bytes, Some(BOUNDED_MEMLOCK_CEILING_BYTES));
    }
}

#[tokio::test]
async fn test_live_muob_arrow_ipc_streaming_and_wal_ack_with_collector_listener() {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind loopback collector listener");
    let collector_addr = listener.local_addr().expect("local_addr");

    let collector_task = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accept client");
        let (span_hdr, span_payload) = read_frame(&mut socket).await.expect("read span frame");
        assert_eq!(span_hdr.msg_type, MSG_TYPE_ARROW_OTEL_SPANS);
        let spans = decode_span_batch(&span_payload).expect("decode span arrow batch");

        let (metric_hdr, metric_payload) =
            read_frame(&mut socket).await.expect("read metric frame");
        assert_eq!(metric_hdr.msg_type, MSG_TYPE_ARROW_OTEL_METRICS);
        let metrics = decode_metric_batch(&metric_payload).expect("decode metric arrow batch");

        (spans, metrics)
    });

    let runtime = create_test_runtime();
    runtime
        .telemetry()
        .set_collector_endpoint(Some(CollectorEndpoint::Tcp(collector_addr)));
    let server = ProtocolServer::new(ProtocolServerConfig::default(), Arc::clone(&runtime));

    let claude_body = serde_json::json!({
        "model": "qwen2.5-3b-a100",
        "max_tokens": 16,
        "messages": [
            {"role": "user", "content": "Stream telemetry to live collector"}
        ],
        "stream": false
    });
    let req = HttpRequest::new(
        "POST",
        "/v1/messages",
        serde_json::to_vec(&claude_body).expect("serialize claude body"),
    );

    let resp = server.route_request(req).await;
    assert_eq!(resp.status, 200);

    let (received_spans, received_metrics) = collector_task.await.expect("collector task join");
    assert_eq!(received_spans.len(), 4);
    assert_eq!(received_metrics.len(), 3);
    assert_eq!(received_metrics[0].cpu_cluster.as_deref(), Some("A100"));
    assert_eq!(received_metrics[0].vlen_bits, Some(1024));
    assert_eq!(runtime.telemetry().exported_frames_count(), 2);

    // Also verify bidirectional `send_batches_with_ack` over duplex stream
    let batches = runtime.telemetry().recorded_framed_batches();
    assert_eq!(batches.len(), 2);
    let (mut client_stream, mut server_stream) = tokio::io::duplex(64 * 1024);

    let ack_server = tokio::spawn(async move {
        let fory = create_fory_registry().expect("fory");
        for expected_seq in 1..=2_u64 {
            let (hdr, _payload) = read_frame(&mut server_stream).await.expect("read frame");
            let ack = WalBatchAck {
                sequence: expected_seq,
                accepted_rows: 4,
                ok: true,
                error_message: None,
            };
            let ack_bytes = fory.serialize(&ack).expect("serialize ack");
            let ack_hdr = IpcHeader::new(
                MSG_TYPE_WAL_ACK,
                u32::try_from(ack_bytes.len()).unwrap_or(0),
                0,
                hdr.sequence,
            );
            write_frame(&mut server_stream, &ack_hdr, &ack_bytes)
                .await
                .expect("write ack");
        }
    });

    let acks = MuObserveTelemetryStreamer::send_batches_with_ack(&mut client_stream, &batches)
        .await
        .expect("send_batches_with_ack");
    ack_server.await.expect("ack server join");
    assert_eq!(acks.len(), 2);
    assert!(acks.iter().all(|a| a.ok));
}

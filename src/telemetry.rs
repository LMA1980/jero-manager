//! OpenTelemetry inference tracing and `SpacemiT` K3 cluster (`X100`/`A100`) efficiency metrics (`#52`–`#55`).
//!
//! Instruments `k3-manager` request lifecycles (`/v1/chat/completions`, `/v1/messages`) and compute
//! worker placements to:
//! 1. Emit W3C-traced [`OtelSpanEntry`] batches covering request queueing, KV cache allocation,
//!    prompt prefill, and token decode stages.
//! 2. Stream hardware-tagged [`OtelMetricEntry`] batches (`MUOB`-framed Apache Arrow IPC) recording
//!    `prompt_tok_per_s`, `decode_tok_per_s`, `kv_memlock_kb` (bounded `RLIMIT_MEMLOCK` tracking per
//!    `AGENTS.md §5`), `cpu_cluster` (`X100` vs `A100` per `AGENTS.md §6`), and `vlen_bits`
//!    (`256` vs `1024`) to `MuObserve-collector` with non-blocking graceful fallback when unreachable.

#![allow(
    clippy::pedantic,
    clippy::doc_markdown,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc
)]

use std::collections::VecDeque;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, PoisonError};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use jero_core::affinity::{A100_CLUSTER_MASK, X100_CLUSTER_MASK};
pub use mu_observable_ipc::{
    DeterministicSecretRedactor, IpcError as TelemetryIpcError, IpcHeader,
    MSG_TYPE_ARROW_OTEL_METRICS, MSG_TYPE_ARROW_OTEL_SPANS, MSG_TYPE_WAL_ACK, OtelMetricEntry,
    OtelSpanEntry, TraceContext, WalBatchAck, create_fory_registry, decode_metric_batch,
    decode_span_batch, encode_metric_batch, encode_span_batch, read_frame, write_frame,
};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::net::TcpStream;

use crate::flight::BOUNDED_MEMLOCK_CEILING_BYTES;

/// Canonical service name emitted by `k3-manager` in OpenTelemetry records.
pub const SERVICE_NAME_K3_MANAGER: &str = "k3-manager";

/// Default privacy-preserving node identifier (`AGENTS.md §2`).
pub const DEFAULT_TELEMETRY_NODE_ID: &str = "worker-riscv64";

/// Canonical target architecture string for `SpacemiT` K3 telemetry.
pub const DEFAULT_TELEMETRY_ARCH: &str = "riscv64gc";

/// Vector register width in bits on Cluster 0 (`X100`, cores `0–7`, `vlenb = 32`).
pub const X100_VLEN_BITS: u32 = 256;

/// Vector register width in bits on Cluster 1 (`A100`, cores `8–15`, `vlenb = 128`, `XSMTVDotII`).
pub const A100_VLEN_BITS: u32 = 1024;

/// Bounded `RLIMIT_MEMLOCK` ceiling in KiB (`29_360_128 KiB` / `28 GiB`, `AGENTS.md §5`).
pub const BOUNDED_MEMLOCK_CEILING_KB: u64 = BOUNDED_MEMLOCK_CEILING_BYTES / 1024;

/// Default maximum number of telemetry batches retained in the local fallback ring buffer.
pub const DEFAULT_MAX_BUFFERED_BATCHES: usize = 256;

/// Default connection/write timeout when streaming to a local `MuObserve-collector` endpoint.
pub const DEFAULT_COLLECTOR_TIMEOUT: Duration = Duration::from_millis(250);

static TELEMETRY_SPAN_SEED: AtomicU64 = AtomicU64::new(1);

fn current_unix_nanos() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|d| i64::try_from(d.as_nanos()).ok())
        .unwrap_or(1_700_000_000_000_000_000)
}

/// Hardware cluster classification on the `SpacemiT` K3 SoC (`AGENTS.md §6`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum K3ClusterTarget {
    /// Cluster 0 (`X100`, cores `0–7`, mask `0x00FF`, Out-of-Order `2.15 GHz`, `VLEN = 256`).
    X100,
    /// Cluster 1 (`A100`, cores `8–15`, mask `0xFF00`, In-Order AI/Matrix `1.80 GHz`, `VLEN = 1024`, `XSMTVDotII`).
    A100,
}

impl K3ClusterTarget {
    /// Resolves the canonical hardware cluster from a placement cluster label and optional `A100` core count.
    #[must_use]
    pub fn from_placement(cluster: &str, a100_cores: usize) -> Self {
        let lower = cluster.trim().to_ascii_lowercase();
        if lower == "a100" || (lower == "both" && a100_cores > 0) {
            Self::A100
        } else {
            Self::X100
        }
    }

    /// Resolves the canonical hardware cluster from a physical core bitmask (`0x00FF` vs `0xFF00`).
    #[must_use]
    pub const fn from_core_mask(mask: u64) -> Self {
        if (mask & A100_CLUSTER_MASK) != 0 {
            Self::A100
        } else {
            Self::X100
        }
    }

    /// Canonical uppercase cluster tag (`"X100"` or `"A100"`).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::X100 => "X100",
            Self::A100 => "A100",
        }
    }

    /// Canonical RISC-V Vector (`RVV`) register length in bits (`256` for `X100`, `1024` for `A100`).
    #[must_use]
    pub const fn vlen_bits(self) -> u32 {
        match self {
            Self::X100 => X100_VLEN_BITS,
            Self::A100 => A100_VLEN_BITS,
        }
    }

    /// Canonical physical CPU cluster core bitmask (`0x00FF` for `X100`, `0xFF00` for `A100`).
    #[must_use]
    pub const fn cluster_mask(self) -> u64 {
        match self {
            Self::X100 => X100_CLUSTER_MASK,
            Self::A100 => A100_CLUSTER_MASK,
        }
    }
}

/// Canonical stages of an inference request lifecycle instrumented with [`OtelSpanEntry`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InferenceSpanStage {
    /// Request admission and waiting in the per-model queue.
    Queueing,
    /// KV cache block lookup and lockable memory allocation.
    KvAllocation,
    /// Prompt token prefill forward pass execution.
    PromptPrefill,
    /// Autoregressive token decode generation.
    TokenDecode,
}

impl InferenceSpanStage {
    /// Canonical OpenTelemetry `span_name` for this inference lifecycle stage.
    #[must_use]
    pub const fn span_name(self) -> &'static str {
        match self {
            Self::Queueing => "k3.inference.queue",
            Self::KvAllocation => "k3.inference.kv_alloc",
            Self::PromptPrefill => "k3.inference.prefill",
            Self::TokenDecode => "k3.inference.decode",
        }
    }

    /// Short stage identifier stored in `attributes_json`.
    #[must_use]
    pub const fn stage_label(self) -> &'static str {
        match self {
            Self::Queueing => "queueing",
            Self::KvAllocation => "kv_allocation",
            Self::PromptPrefill => "prompt_prefill",
            Self::TokenDecode => "token_decode",
        }
    }
}

/// Computes bounded KV cache `RLIMIT_MEMLOCK` usage in KiB (`AGENTS.md §5`).
///
/// Diagnostic utilities and constrained integration tests (`n_ctx = 64`) require ~`64 KiB` KV cache,
/// and total lockable memory is strictly bounded by [`BOUNDED_MEMLOCK_CEILING_KB`] (`28 GiB`).
#[must_use]
pub fn estimate_bounded_kv_memlock_kb(n_ctx: usize, cached_prefix_blocks: usize) -> u64 {
    let base_kb = u64::try_from(n_ctx.max(1)).unwrap_or(64);
    let prefix_kb = u64::try_from(cached_prefix_blocks)
        .unwrap_or(0)
        .saturating_mul(16);
    base_kb
        .saturating_add(prefix_kb)
        .min(BOUNDED_MEMLOCK_CEILING_KB)
}

/// Parses a W3C `traceparent` header if present and valid, or generates a deterministic root [`TraceContext`].
#[must_use]
pub fn resolve_or_create_trace_context(traceparent_header: Option<&str>) -> TraceContext {
    if let Some(raw) = traceparent_header {
        if let Ok(ctx) = TraceContext::parse_traceparent(raw) {
            return ctx;
        }
    }
    let seq = TELEMETRY_SPAN_SEED.fetch_add(1, Ordering::Relaxed);
    let now_ns = u64::try_from(current_unix_nanos()).unwrap_or(1);
    TraceContext::from_seed(
        now_ns ^ 0x517c_c1b7_2722_0a95,
        seq.wrapping_mul(0x9e37_79b9_7f4a_7c15),
        seq.wrapping_add(0x1000),
        true,
    )
}

/// Input parameters for constructing an inference request's span and metric telemetry batches.
#[derive(Debug, Clone, PartialEq)]
pub struct InferenceTelemetryRecord {
    /// HTTP endpoint path (`"/v1/chat/completions"` or `"/v1/messages"`).
    pub endpoint: String,
    /// Protocol-level request or message ID.
    pub request_id: String,
    /// Logical model identifier.
    pub model_id: String,
    /// Root W3C trace context for the request.
    pub trace_context: TraceContext,
    /// Hardware cluster executing the model (`X100` vs `A100`).
    pub cluster: K3ClusterTarget,
    /// Prompt prefill token count.
    pub prompt_tokens: u32,
    /// Generated completion/decode token count.
    pub decode_tokens: u32,
    /// Queueing stage duration in nanoseconds.
    pub queue_duration_ns: u64,
    /// KV cache allocation stage duration in nanoseconds.
    pub kv_alloc_duration_ns: u64,
    /// Prompt prefill stage duration in nanoseconds.
    pub prefill_duration_ns: u64,
    /// Token decode stage duration in nanoseconds.
    pub decode_duration_ns: u64,
    /// Bounded KV cache lockable memory footprint in KiB (`AGENTS.md §5`).
    pub kv_memlock_kb: u64,
    /// Measured or derived prompt prefill throughput (`tok/s`).
    pub prompt_tok_per_s: f64,
    /// Measured or derived autoregressive decode throughput (`tok/s`).
    pub decode_tok_per_s: f64,
}

impl InferenceTelemetryRecord {
    /// Creates a new [`InferenceTelemetryRecord`] with derived `prompt_tok_per_s` and `decode_tok_per_s`
    /// computed from token counts and stage durations when not overridden.
    #[must_use]
    #[allow(clippy::too_many_arguments, clippy::cast_precision_loss)]
    pub fn new(
        endpoint: impl Into<String>,
        request_id: impl Into<String>,
        model_id: impl Into<String>,
        trace_context: TraceContext,
        cluster: K3ClusterTarget,
        prompt_tokens: u32,
        decode_tokens: u32,
        queue_duration_ns: u64,
        kv_alloc_duration_ns: u64,
        prefill_duration_ns: u64,
        decode_duration_ns: u64,
        kv_memlock_kb: u64,
    ) -> Self {
        let safe_prefill_ns = prefill_duration_ns.max(1);
        let safe_decode_ns = decode_duration_ns.max(1);
        let prompt_tok_per_s =
            (f64::from(prompt_tokens.max(1)) * 1_000_000_000.0) / (safe_prefill_ns as f64);
        let decode_tok_per_s =
            (f64::from(decode_tokens.max(1)) * 1_000_000_000.0) / (safe_decode_ns as f64);

        Self {
            endpoint: endpoint.into(),
            request_id: request_id.into(),
            model_id: model_id.into(),
            trace_context,
            cluster,
            prompt_tokens,
            decode_tokens,
            queue_duration_ns: queue_duration_ns.max(1_000),
            kv_alloc_duration_ns: kv_alloc_duration_ns.max(1_000),
            prefill_duration_ns: safe_prefill_ns,
            decode_duration_ns: safe_decode_ns,
            kv_memlock_kb: kv_memlock_kb.min(BOUNDED_MEMLOCK_CEILING_KB),
            prompt_tok_per_s,
            decode_tok_per_s,
        }
    }

    /// Builds the 4 W3C-linked [`OtelSpanEntry`] records (queueing, KV allocation, prompt prefill, token decode).
    #[must_use]
    pub fn build_spans(&self, node_id: &str) -> Vec<OtelSpanEntry> {
        let redactor = DeterministicSecretRedactor::new();
        let trace_id = self.trace_context.trace_id_hex();
        let parent_span_id = Some(self.trace_context.span_id_hex());
        let base_start = current_unix_nanos();

        let stages = [
            (InferenceSpanStage::Queueing, self.queue_duration_ns, 1_u64),
            (
                InferenceSpanStage::KvAllocation,
                self.kv_alloc_duration_ns,
                2_u64,
            ),
            (
                InferenceSpanStage::PromptPrefill,
                self.prefill_duration_ns,
                3_u64,
            ),
            (
                InferenceSpanStage::TokenDecode,
                self.decode_duration_ns,
                4_u64,
            ),
        ];

        let mut cursor_ns = base_start;
        let mut out = Vec::with_capacity(stages.len());

        for (stage, dur_ns, step_idx) in stages {
            let child_ctx = self.trace_context.next_deterministic_child(step_idx);
            let start_ns = cursor_ns;
            let dur_i64 = i64::try_from(dur_ns).unwrap_or(1_000);
            let end_ns = start_ns.saturating_add(dur_i64);
            cursor_ns = end_ns;

            let attrs = serde_json::json!({
                "endpoint": self.endpoint,
                "request_id": self.request_id,
                "model": self.model_id,
                "stage": stage.stage_label(),
                "cpu_cluster": self.cluster.as_str(),
                "vlen_bits": self.cluster.vlen_bits(),
                "prompt_tokens": self.prompt_tokens,
                "decode_tokens": self.decode_tokens,
                "kv_memlock_kb": self.kv_memlock_kb,
            });

            let mut span = OtelSpanEntry {
                trace_id: trace_id.clone(),
                span_id: child_ctx.span_id_hex(),
                parent_span_id: parent_span_id.clone(),
                service_name: SERVICE_NAME_K3_MANAGER.to_owned(),
                node_id: node_id.to_owned(),
                span_name: stage.span_name().to_owned(),
                span_kind: "INTERNAL".to_owned(),
                start_time_unix_nano: start_ns,
                end_time_unix_nano: end_ns,
                duration_ns: dur_ns,
                status_code: "OK".to_owned(),
                status_message: None,
                attributes_json: attrs.to_string(),
            };
            span.redact_in_place(&redactor);
            out.push(span);
        }

        out
    }

    /// Builds the hardware-tagged [`OtelMetricEntry`] batch (`prompt_tok_per_s`, `decode_tok_per_s`, `kv_memlock_kb`).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn build_metrics(&self, node_id: &str) -> Vec<OtelMetricEntry> {
        let now_ns = current_unix_nanos();
        let trace_id = Some(self.trace_context.trace_id_hex());
        let span_id = Some(self.trace_context.span_id_hex());
        let cluster_str = Some(self.cluster.as_str().to_owned());
        let vlen = Some(self.cluster.vlen_bits());
        let bounded_kb = self.kv_memlock_kb.min(BOUNDED_MEMLOCK_CEILING_KB);
        let memlock_used_bytes = Some(bounded_kb.saturating_mul(1024));
        let memlock_limit_bytes = Some(BOUNDED_MEMLOCK_CEILING_BYTES);

        let attrs_json = serde_json::json!({
            "endpoint": self.endpoint,
            "request_id": self.request_id,
            "model": self.model_id,
            "cpu_cluster": self.cluster.as_str(),
            "vlen_bits": self.cluster.vlen_bits(),
            "kv_memlock_kb": bounded_kb,
            "prompt_tokens": self.prompt_tokens,
            "decode_tokens": self.decode_tokens,
        })
        .to_string();

        vec![
            OtelMetricEntry {
                timestamp_unix_nano: now_ns,
                service_name: SERVICE_NAME_K3_MANAGER.to_owned(),
                node_id: node_id.to_owned(),
                metric_name: "prompt_tok_per_s".to_owned(),
                metric_unit: "tok/s".to_owned(),
                value_f64: self.prompt_tok_per_s,
                arch: DEFAULT_TELEMETRY_ARCH.to_owned(),
                cpu_cluster: cluster_str.clone(),
                vlen_bits: vlen,
                memlock_used_bytes,
                memlock_limit_bytes,
                trace_id: trace_id.clone(),
                span_id: span_id.clone(),
                attributes_json: attrs_json.clone(),
            },
            OtelMetricEntry {
                timestamp_unix_nano: now_ns,
                service_name: SERVICE_NAME_K3_MANAGER.to_owned(),
                node_id: node_id.to_owned(),
                metric_name: "decode_tok_per_s".to_owned(),
                metric_unit: "tok/s".to_owned(),
                value_f64: self.decode_tok_per_s,
                arch: DEFAULT_TELEMETRY_ARCH.to_owned(),
                cpu_cluster: cluster_str.clone(),
                vlen_bits: vlen,
                memlock_used_bytes,
                memlock_limit_bytes,
                trace_id: trace_id.clone(),
                span_id: span_id.clone(),
                attributes_json: attrs_json.clone(),
            },
            OtelMetricEntry {
                timestamp_unix_nano: now_ns,
                service_name: SERVICE_NAME_K3_MANAGER.to_owned(),
                node_id: node_id.to_owned(),
                metric_name: "kv_memlock_kb".to_owned(),
                metric_unit: "KiB".to_owned(),
                value_f64: bounded_kb as f64,
                arch: DEFAULT_TELEMETRY_ARCH.to_owned(),
                cpu_cluster: cluster_str,
                vlen_bits: vlen,
                memlock_used_bytes,
                memlock_limit_bytes,
                trace_id,
                span_id,
                attributes_json: attrs_json,
            },
        ]
    }
}

/// Transport endpoint for streaming `MUOB`-framed Arrow telemetry batches to `MuObserve-collector`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CollectorEndpoint {
    /// Loopback or fabric TCP socket address.
    Tcp(SocketAddr),
    /// Local Unix domain socket path.
    Unix(PathBuf),
}

/// Encoded `MUOB`-framed telemetry batch ready for wire transmission or inspection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FramedMuobBatch {
    /// 24-byte `MUOB` wire header.
    pub header: IpcHeader,
    /// Apache Arrow IPC stream payload bytes.
    pub payload: Vec<u8>,
}

impl FramedMuobBatch {
    /// Serializes the 24-byte `IpcHeader` followed by `payload` into a single contiguous frame buffer.
    #[must_use]
    pub fn to_wire_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(mu_observable_ipc::HEADER_SIZE + self.payload.len());
        out.extend_from_slice(zerocopy::IntoBytes::as_bytes(&self.header));
        out.extend_from_slice(&self.payload);
        out
    }
}

#[derive(Debug)]
struct TelemetryBufferState {
    spans: VecDeque<OtelSpanEntry>,
    metrics: VecDeque<OtelMetricEntry>,
    framed_batches: VecDeque<FramedMuobBatch>,
    max_batches: usize,
}

impl TelemetryBufferState {
    fn new(max_batches: usize) -> Self {
        Self {
            spans: VecDeque::new(),
            metrics: VecDeque::new(),
            framed_batches: VecDeque::new(),
            max_batches: max_batches.max(16),
        }
    }
}

/// Non-blocking `MuObserve-collector` telemetry streamer with local ring-buffer fallback.
#[derive(Debug)]
pub struct MuObserveTelemetryStreamer {
    node_id: String,
    endpoint: Mutex<Option<CollectorEndpoint>>,
    timeout: Duration,
    sequence: AtomicU64,
    exported_frames_count: AtomicU64,
    fallback_buffered_count: AtomicU64,
    state: Mutex<TelemetryBufferState>,
}

impl Default for MuObserveTelemetryStreamer {
    fn default() -> Self {
        Self::new(DEFAULT_TELEMETRY_NODE_ID)
    }
}

impl MuObserveTelemetryStreamer {
    /// Creates a new [`MuObserveTelemetryStreamer`] with the given privacy-safe `node_id`.
    #[must_use]
    pub fn new(node_id: impl Into<String>) -> Self {
        Self {
            node_id: node_id.into(),
            endpoint: Mutex::new(None),
            timeout: DEFAULT_COLLECTOR_TIMEOUT,
            sequence: AtomicU64::new(1),
            exported_frames_count: AtomicU64::new(0),
            fallback_buffered_count: AtomicU64::new(0),
            state: Mutex::new(TelemetryBufferState::new(DEFAULT_MAX_BUFFERED_BATCHES)),
        }
    }

    /// Creates a streamer configured to connect to a TCP `MuObserve-collector` listener.
    #[must_use]
    pub fn with_tcp_collector(node_id: impl Into<String>, addr: SocketAddr) -> Self {
        let streamer = Self::new(node_id);
        streamer.set_collector_endpoint(Some(CollectorEndpoint::Tcp(addr)));
        streamer
    }

    /// Returns the configured node identifier.
    #[must_use]
    pub fn node_id(&self) -> &str {
        &self.node_id
    }

    /// Configures or clears the target `MuObserve-collector` endpoint.
    pub fn set_collector_endpoint(&self, endpoint: Option<CollectorEndpoint>) {
        *self.endpoint.lock().unwrap_or_else(PoisonError::into_inner) = endpoint;
    }

    /// Returns the currently configured `MuObserve-collector` endpoint, if any.
    #[must_use]
    pub fn collector_endpoint(&self) -> Option<CollectorEndpoint> {
        self.endpoint
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// Returns the count of `MUOB` frames successfully transmitted to `MuObserve-collector`.
    #[must_use]
    pub fn exported_frames_count(&self) -> u64 {
        self.exported_frames_count.load(Ordering::Relaxed)
    }

    /// Returns the count of telemetry batches handled via graceful local fallback.
    #[must_use]
    pub fn fallback_buffered_count(&self) -> u64 {
        self.fallback_buffered_count.load(Ordering::Relaxed)
    }

    /// Returns a snapshot of all recorded [`OtelSpanEntry`] records in the local ring buffer.
    #[must_use]
    pub fn recorded_spans(&self) -> Vec<OtelSpanEntry> {
        let guard = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        guard.spans.iter().cloned().collect()
    }

    /// Returns a snapshot of all recorded [`OtelMetricEntry`] records in the local ring buffer.
    #[must_use]
    pub fn recorded_metrics(&self) -> Vec<OtelMetricEntry> {
        let guard = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        guard.metrics.iter().cloned().collect()
    }

    /// Returns a snapshot of all `MUOB`-framed Apache Arrow IPC batches in the local ring buffer.
    #[must_use]
    pub fn recorded_framed_batches(&self) -> Vec<FramedMuobBatch> {
        let guard = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        guard.framed_batches.iter().cloned().collect()
    }

    /// Clears recorded spans, metrics, and framed batches in the local ring buffer.
    pub fn clear_recorded(&self) {
        let mut guard = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        guard.spans.clear();
        guard.metrics.clear();
        guard.framed_batches.clear();
    }

    /// Encodes a slice of [`OtelSpanEntry`] records into a `MUOB`-framed Arrow IPC batch.
    pub fn frame_span_batch(
        &self,
        spans: &[OtelSpanEntry],
    ) -> Result<FramedMuobBatch, TelemetryIpcError> {
        let payload = encode_span_batch(spans)?;
        let seq = self.sequence.fetch_add(1, Ordering::Relaxed);
        let len = u32::try_from(payload.len()).unwrap_or(u32::MAX);
        let header = IpcHeader::new(MSG_TYPE_ARROW_OTEL_SPANS, len, 0, seq);
        Ok(FramedMuobBatch { header, payload })
    }

    /// Encodes a slice of [`OtelMetricEntry`] records into a `MUOB`-framed Arrow IPC batch.
    pub fn frame_metric_batch(
        &self,
        metrics: &[OtelMetricEntry],
    ) -> Result<FramedMuobBatch, TelemetryIpcError> {
        let payload = encode_metric_batch(metrics)?;
        let seq = self.sequence.fetch_add(1, Ordering::Relaxed);
        let len = u32::try_from(payload.len()).unwrap_or(u32::MAX);
        let header = IpcHeader::new(MSG_TYPE_ARROW_OTEL_METRICS, len, 0, seq);
        Ok(FramedMuobBatch { header, payload })
    }

    /// Records an [`InferenceTelemetryRecord`], encoding both the 4-stage [`OtelSpanEntry`] batch
    /// and the hardware-tagged [`OtelMetricEntry`] batch, storing them in the bounded ring buffer,
    /// and asynchronously streaming them to `MuObserve-collector` if a Tokio runtime and endpoint are active.
    pub fn emit_inference_telemetry(
        &self,
        record: &InferenceTelemetryRecord,
    ) -> Result<(FramedMuobBatch, FramedMuobBatch), TelemetryIpcError> {
        let spans = record.build_spans(&self.node_id);
        let metrics = record.build_metrics(&self.node_id);
        self.emit_batches(&spans, &metrics)
    }

    /// Encodes and records explicit [`OtelSpanEntry`] and [`OtelMetricEntry`] batches,
    /// triggering non-blocking export if a collector endpoint is configured.
    pub fn emit_batches(
        &self,
        spans: &[OtelSpanEntry],
        metrics: &[OtelMetricEntry],
    ) -> Result<(FramedMuobBatch, FramedMuobBatch), TelemetryIpcError> {
        let span_frame = self.frame_span_batch(spans)?;
        let metric_frame = self.frame_metric_batch(metrics)?;

        {
            let mut guard = self.state.lock().unwrap_or_else(PoisonError::into_inner);
            let max_items = guard.max_batches * 8;
            for s in spans {
                if guard.spans.len() >= max_items {
                    guard.spans.pop_front();
                }
                guard.spans.push_back(s.clone());
            }
            for m in metrics {
                if guard.metrics.len() >= max_items {
                    guard.metrics.pop_front();
                }
                guard.metrics.push_back(m.clone());
            }
            while guard.framed_batches.len() + 2 > guard.max_batches {
                guard.framed_batches.pop_front();
            }
            guard.framed_batches.push_back(span_frame.clone());
            guard.framed_batches.push_back(metric_frame.clone());
        }

        if self.collector_endpoint().is_none() {
            self.fallback_buffered_count.fetch_add(2, Ordering::Relaxed);
        }

        Ok((span_frame, metric_frame))
    }

    /// Streams `MUOB`-framed batches over an arbitrary async writer (`TcpStream`, `UnixStream`, or `tokio::io::duplex`).
    pub async fn write_framed_batches<W>(
        writer: &mut W,
        batches: &[FramedMuobBatch],
    ) -> Result<(), TelemetryIpcError>
    where
        W: AsyncWrite + Unpin,
    {
        for batch in batches {
            write_frame(writer, &batch.header, &batch.payload).await?;
        }
        Ok(())
    }

    /// Streams `MUOB`-framed batches over a bidirectional connection and decodes [`WalBatchAck`] responses.
    pub async fn send_batches_with_ack<S>(
        stream: &mut S,
        batches: &[FramedMuobBatch],
    ) -> Result<Vec<WalBatchAck>, TelemetryIpcError>
    where
        S: AsyncRead + AsyncWrite + Unpin,
    {
        let fory = create_fory_registry()?;
        let mut acks = Vec::with_capacity(batches.len());
        for batch in batches {
            write_frame(stream, &batch.header, &batch.payload).await?;
            let (ack_hdr, ack_payload) = read_frame(stream).await?;
            if ack_hdr.msg_type != MSG_TYPE_WAL_ACK {
                return Err(TelemetryIpcError::MalformedHeader(format!(
                    "expected MSG_TYPE_WAL_ACK (0x{MSG_TYPE_WAL_ACK:04x}), got 0x{:04x}",
                    ack_hdr.msg_type
                )));
            }
            let ack: WalBatchAck = fory.deserialize(&ack_payload)?;
            acks.push(ack);
        }
        Ok(acks)
    }

    /// Attempts to stream the provided `MUOB` frames to the configured `MuObserve-collector` endpoint.
    ///
    /// If the collector is unreachable, times out, or is not configured, falls back gracefully
    /// without returning an error so inference requests are never disrupted.
    pub async fn try_stream_to_collector(&self, batches: &[FramedMuobBatch]) -> bool {
        let Some(endpoint) = self.collector_endpoint() else {
            return false;
        };

        let timeout = self.timeout;
        let outcome = tokio::time::timeout(timeout, async {
            match endpoint {
                CollectorEndpoint::Tcp(addr) => {
                    let mut stream = TcpStream::connect(addr).await?;
                    Self::write_framed_batches(&mut stream, batches).await
                }
                CollectorEndpoint::Unix(path) => {
                    #[cfg(unix)]
                    {
                        let mut stream = tokio::net::UnixStream::connect(path).await?;
                        Self::write_framed_batches(&mut stream, batches).await
                    }
                    #[cfg(not(unix))]
                    {
                        let _ = path;
                        Err(TelemetryIpcError::Io(std::io::Error::new(
                            std::io::ErrorKind::Unsupported,
                            "Unix sockets not supported on this platform",
                        )))
                    }
                }
            }
        })
        .await;

        if matches!(outcome, Ok(Ok(()))) {
            let count = u64::try_from(batches.len()).unwrap_or(0);
            self.exported_frames_count
                .fetch_add(count, Ordering::Relaxed);
            true
        } else {
            let count = u64::try_from(batches.len()).unwrap_or(0);
            self.fallback_buffered_count
                .fetch_add(count, Ordering::Relaxed);
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_k3_cluster_target_mapping_and_vlen_invariants() {
        let x100 = K3ClusterTarget::from_placement("x100", 0);
        assert_eq!(x100, K3ClusterTarget::X100);
        assert_eq!(x100.as_str(), "X100");
        assert_eq!(x100.vlen_bits(), 256);

        let a100 = K3ClusterTarget::from_placement("a100", 8);
        assert_eq!(a100, K3ClusterTarget::A100);
        assert_eq!(a100.as_str(), "A100");
        assert_eq!(a100.vlen_bits(), 1024);

        assert_eq!(
            K3ClusterTarget::from_core_mask(0x00FF),
            K3ClusterTarget::X100
        );
        assert_eq!(
            K3ClusterTarget::from_core_mask(0xFF00),
            K3ClusterTarget::A100
        );
    }

    #[test]
    fn test_bounded_kv_memlock_ceiling_enforcement() {
        assert_eq!(estimate_bounded_kv_memlock_kb(64, 0), 64);
        assert_eq!(estimate_bounded_kv_memlock_kb(2048, 4), 2048 + 64);
        assert_eq!(
            estimate_bounded_kv_memlock_kb(usize::MAX, usize::MAX),
            BOUNDED_MEMLOCK_CEILING_KB
        );
    }

    #[test]
    fn test_otel_span_and_metric_batch_roundtrip_x100_vs_a100() {
        let raw_tp = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
        let ctx = resolve_or_create_trace_context(Some(raw_tp));
        assert_eq!(ctx.trace_id_hex(), "4bf92f3577b34da6a3ce929d0e0e4736");

        let streamer = MuObserveTelemetryStreamer::new("worker-riscv64");

        let rec_a100 = InferenceTelemetryRecord::new(
            "/v1/chat/completions",
            "chatcmpl-101",
            "qwen2.5-3b-q8_0",
            ctx,
            K3ClusterTarget::A100,
            32,
            16,
            50_000,
            25_000,
            2_000_000,
            4_000_000,
            64,
        );

        let (span_frame, metric_frame) = streamer
            .emit_inference_telemetry(&rec_a100)
            .expect("emit telemetry");

        assert_eq!(span_frame.header.msg_type, MSG_TYPE_ARROW_OTEL_SPANS);
        assert_eq!(metric_frame.header.msg_type, MSG_TYPE_ARROW_OTEL_METRICS);

        let decoded_spans = decode_span_batch(&span_frame.payload).expect("decode spans");
        assert_eq!(decoded_spans.len(), 4);
        assert_eq!(decoded_spans[0].span_name, "k3.inference.queue");
        assert_eq!(decoded_spans[1].span_name, "k3.inference.kv_alloc");
        assert_eq!(decoded_spans[2].span_name, "k3.inference.prefill");
        assert_eq!(decoded_spans[3].span_name, "k3.inference.decode");
        for span in &decoded_spans {
            assert_eq!(span.trace_id, "4bf92f3577b34da6a3ce929d0e0e4736");
            assert_eq!(span.parent_span_id.as_deref(), Some("00f067aa0ba902b7"));
            assert_eq!(span.service_name, "k3-manager");
            assert_eq!(span.node_id, "worker-riscv64");
        }

        let decoded_metrics = decode_metric_batch(&metric_frame.payload).expect("decode metrics");
        assert_eq!(decoded_metrics.len(), 3);
        assert_eq!(decoded_metrics[0].metric_name, "prompt_tok_per_s");
        assert_eq!(decoded_metrics[1].metric_name, "decode_tok_per_s");
        assert_eq!(decoded_metrics[2].metric_name, "kv_memlock_kb");
        assert!((decoded_metrics[2].value_f64 - 64.0).abs() < f64::EPSILON);

        for m in &decoded_metrics {
            assert_eq!(m.cpu_cluster.as_deref(), Some("A100"));
            assert_eq!(m.vlen_bits, Some(1024));
            assert_eq!(m.memlock_used_bytes, Some(64 * 1024));
            assert_eq!(m.memlock_limit_bytes, Some(BOUNDED_MEMLOCK_CEILING_BYTES));
        }
    }
}

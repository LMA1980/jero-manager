//! Zero-copy forward pass dispatch and Apache Arrow tensor exposure.
//!
//! Enforces zero bulk tensor payload bytes across Unix Domain Sockets:
//! input tokens are placed in-place into shared-memory slots (`wrap_tokens_as_uint32_array`),
//! compact `ForwardPassCommand` descriptors are sent over UDS, and output activations/logits
//! are exposed directly from shared memory as Apache Arrow `Float32Array` columnar views.

use std::collections::VecDeque;
use std::os::unix::net::UnixStream;
use std::sync::Arc;

use arrow_array::{Float32Array, UInt32Array};
use jero_core::K3Error;
use jero_core::ipc::arrow::{wrap_floats_as_float32_array, wrap_tokens_as_uint32_array};
use jero_core::ipc::layout::{
    OutputSlabHeader, SLAB_STATUS_IDLE, SLAB_STATUS_READ, SLAB_STATUS_WRITTEN, TokenInputSlotHeader,
};
use jero_core::ipc::protocol::{
    ForwardPassCommand, ForwardPassCompleted, IpcMessage, recv_msg_frame, send_msg_frame,
};
use jero_core::ipc::shm::SharedMemoryArena;
use zerocopy::FromBytes;

use crate::error::ManagerError;

/// Input parameters for dispatching a forward pass to a compute worker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForwardPassRequest {
    /// Unique request identifier.
    pub request_id: u64,
    /// Sequence identifier.
    pub sequence_id: u64,
    /// Input prompt or decode token IDs.
    pub tokens: Vec<u32>,
    /// Starting sequence position for this forward pass.
    pub start_position: u32,
    /// 16-token KV cache block indices assigned to this sequence.
    pub block_indices: Vec<u32>,
    /// `true` for prompt prefill, `false` for single-token autoregressive decode.
    pub is_prefill: bool,
}

impl ForwardPassRequest {
    /// Constructs a new prefill request.
    #[must_use]
    pub fn new_prefill(
        request_id: u64,
        sequence_id: u64,
        tokens: Vec<u32>,
        block_indices: Vec<u32>,
    ) -> Self {
        Self {
            request_id,
            sequence_id,
            tokens,
            start_position: 0,
            block_indices,
            is_prefill: true,
        }
    }

    /// Constructs a new decode step request.
    #[must_use]
    pub fn new_decode(
        request_id: u64,
        sequence_id: u64,
        next_token: u32,
        position: u32,
        block_indices: Vec<u32>,
    ) -> Self {
        Self {
            request_id,
            sequence_id,
            tokens: vec![next_token],
            start_position: position,
            block_indices,
            is_prefill: false,
        }
    }
}

/// Result returned from executing a forward pass.
#[derive(Debug, Clone)]
pub struct ForwardPassResult {
    /// Execution status, checkpointed committed tokens, and measured latencies.
    pub completed: ForwardPassCompleted,
    /// Zero-copy Apache Arrow view over output logits in the shared-memory slab.
    pub logits: Float32Array,
}

impl ForwardPassResult {
    /// Returns the sampled or top-logit next token ID.
    #[must_use]
    pub const fn next_token(&self) -> u32 {
        self.completed.next_token
    }

    /// Returns the Apache Arrow logits array.
    #[must_use]
    pub fn logits(&self) -> &Float32Array {
        &self.logits
    }

    /// Returns measured prefill latency in milliseconds.
    #[must_use]
    pub const fn prefill_latency_ms(&self) -> f64 {
        self.completed.prefill_latency_ms
    }

    /// Returns measured decode latency in milliseconds.
    #[must_use]
    pub const fn decode_latency_ms(&self) -> f64 {
        self.completed.decode_latency_ms
    }
}

/// Dispatches a forward pass to a connected compute worker with zero bulk tensor bytes over UDS.
///
/// # Errors
/// Returns [`ManagerError`] on shared memory, IPC communication, or worker execution failure.
#[allow(clippy::too_many_lines, clippy::cast_ptr_alignment)]
pub fn dispatch_forward_pass(
    stream: &UnixStream,
    arena: &mut SharedMemoryArena,
    req: &ForwardPassRequest,
) -> Result<ForwardPassResult, ManagerError> {
    if req.tokens.is_empty() {
        return Err(ManagerError::Protocol(
            "cannot dispatch forward pass with 0 tokens".to_owned(),
        ));
    }

    // 1. Read header geometry and validate context length bounds
    let hdr = *arena.header()?;
    let req_tokens_len = u32::try_from(req.tokens.len())
        .map_err(|_| ManagerError::Protocol("token count exceeds u32".to_owned()))?;

    if req.start_position.saturating_add(req_tokens_len) > hdr.n_ctx {
        return Err(ManagerError::Protocol(format!(
            "sequence position {} + token count {} exceeds arena n_ctx {}",
            req.start_position, req_tokens_len, hdr.n_ctx
        )));
    }

    let slot_offset = usize::try_from(hdr.token_slots_offset)
        .map_err(|e| ManagerError::Protocol(format!("invalid token_slots_offset: {e}")))?;

    // 2. Write TokenInputSlotHeader (16 bytes) into shared memory
    let slot_hdr_slice = arena.slice_mut(slot_offset, 16)?;
    let slot_hdr = TokenInputSlotHeader::mut_from_bytes(slot_hdr_slice)
        .map_err(|e| ManagerError::Protocol(format!("failed to cast TokenInputSlotHeader: {e}")))?;
    slot_hdr.slot_seq_id = req.sequence_id;
    slot_hdr.token_len = req_tokens_len;
    slot_hdr.start_position = req.start_position;

    // 3. Write tokens array in place into shared memory slot
    let tokens_data_offset = slot_offset + 16;
    let tokens_slice = arena.slice_mut(tokens_data_offset, req.tokens.len() * 4)?;
    for (i, &tok) in req.tokens.iter().enumerate() {
        tokens_slice[i * 4..(i + 1) * 4].copy_from_slice(&tok.to_ne_bytes());
    }

    // 4. Mark output slab as IDLE prior to pass execution
    let slab_offset = usize::try_from(hdr.output_slabs_offset)
        .map_err(|e| ManagerError::Protocol(format!("invalid output_slabs_offset: {e}")))?;
    let slab_hdr_slice = arena.slice_mut(slab_offset, 64)?;
    let slab_hdr = OutputSlabHeader::mut_from_bytes(slab_hdr_slice)
        .map_err(|e| ManagerError::Protocol(format!("failed to cast OutputSlabHeader: {e}")))?;
    slab_hdr.status = SLAB_STATUS_IDLE;

    // 5. Send compact ForwardPassCommand descriptor over UDS (zero tensor bytes over socket)
    let cmd = IpcMessage::ForwardPassCommand(ForwardPassCommand {
        request_id: req.request_id,
        sequence_id: req.sequence_id,
        token_offset: req.start_position,
        token_count: req_tokens_len,
        block_indices: req.block_indices.clone(),
        is_prefill: req.is_prefill,
    });

    send_msg_frame(stream, &cmd, None).map_err(|e| {
        let is_timeout = e.kind() == std::io::ErrorKind::WouldBlock
            || e.kind() == std::io::ErrorKind::TimedOut
            || e.raw_os_error() == Some(libc::EAGAIN)
            || e.raw_os_error() == Some(libc::ETIMEDOUT);
        if is_timeout {
            ManagerError::Io(e)
        } else {
            ManagerError::WorkerIpc(format!("failed to send ForwardPassCommand to worker: {e}"))
        }
    })?;

    // 6. Receive ForwardPassCompleted from worker
    let (reply, _) = recv_msg_frame(stream).map_err(|e| {
        let is_timeout = e.kind() == std::io::ErrorKind::WouldBlock
            || e.kind() == std::io::ErrorKind::TimedOut
            || e.raw_os_error() == Some(libc::EAGAIN)
            || e.raw_os_error() == Some(libc::ETIMEDOUT);
        if is_timeout {
            ManagerError::Io(e)
        } else {
            ManagerError::WorkerIpc(format!(
                "failed to receive ForwardPassCompleted from worker: {e}"
            ))
        }
    })?;

    let completed = match reply {
        IpcMessage::ForwardPassCompleted(c) => {
            if !c.status.is_success() {
                return Err(ManagerError::Status {
                    status: c.status,
                    message: c.error_message.unwrap_or_else(|| {
                        format!("forward pass failed with status {:?}", c.status)
                    }),
                });
            }
            if c.request_id != req.request_id {
                return Err(ManagerError::Protocol(format!(
                    "request ID mismatch: sent {}, received {}",
                    req.request_id, c.request_id
                )));
            }
            c
        }
        other => {
            return Err(ManagerError::Protocol(format!(
                "expected ForwardPassCompleted from worker, got {other:?}"
            )));
        }
    };

    // 7. Verify slab status is WRITTEN and wrap output logits into zero-copy Apache Arrow Float32Array
    let slab_slice = arena.slice(slab_offset, 64)?;
    let slab_hdr = OutputSlabHeader::ref_from_bytes(slab_slice)
        .map_err(|e| ManagerError::Protocol(format!("failed to cast OutputSlabHeader: {e}")))?;

    if slab_hdr.status != SLAB_STATUS_WRITTEN {
        return Err(ManagerError::Protocol(format!(
            "output slab status is not WRITTEN (observed {})",
            slab_hdr.status
        )));
    }

    let logits = wrap_logits_as_float32_array(arena)?;

    // 8. Mark slab as READ to signal consumer completed inspection (ADR-018, ADR-042)
    let slab_slice_mut = arena.slice_mut(slab_offset, 64)?;
    if let Ok(slab_hdr_mut) = OutputSlabHeader::mut_from_bytes(slab_slice_mut) {
        slab_hdr_mut.status = SLAB_STATUS_READ;
    }

    Ok(ForwardPassResult { completed, logits })
}

/// Exposes the output activations/logits in place from `OutputSlabHeader` via Apache Arrow zero-copy [`Float32Array`] view.
///
/// # Errors
/// Returns [`ManagerError`] on bounds, layout, or casting failure.
#[allow(clippy::cast_ptr_alignment)]
pub fn wrap_logits_as_float32_array(
    arena: &SharedMemoryArena,
) -> Result<Float32Array, ManagerError> {
    let hdr = *arena.header()?;
    let slab_offset = usize::try_from(hdr.output_slabs_offset)
        .map_err(|e| ManagerError::Protocol(format!("invalid output_slabs_offset: {e}")))?;

    let slab_slice = arena.slice(slab_offset, 64)?;
    let slab_hdr = OutputSlabHeader::ref_from_bytes(slab_slice)
        .map_err(|e| ManagerError::Protocol(format!("failed to cast OutputSlabHeader: {e}")))?;

    let vocab_size = if slab_hdr.vocab_size > 0 {
        slab_hdr.vocab_size as usize
    } else if hdr.vocab_size > 0 {
        hdr.vocab_size as usize
    } else {
        10
    };

    let logits_offset = slab_offset + 64;
    let logits_slice = arena.slice(logits_offset, vocab_size * 4)?;

    let arena_fd_owner = Arc::new(arena.duplicate_fd()?);
    // SAFETY: logits_slice points to vocab_size valid f32 values in memory backed by arena_fd_owner.
    unsafe {
        wrap_floats_as_float32_array(
            arena_fd_owner,
            "output_logits",
            logits_slice.as_ptr().cast::<f32>(),
            vocab_size,
        )
        .map_err(ManagerError::from)
    }
}

/// Exposes the input tokens in the shared-memory slot as a zero-copy Apache Arrow [`UInt32Array`].
///
/// # Errors
/// Returns [`ManagerError`] on bounds or layout failure.
#[allow(clippy::cast_ptr_alignment)]
pub fn wrap_input_tokens_as_arrow(arena: &SharedMemoryArena) -> Result<UInt32Array, ManagerError> {
    let hdr = *arena.header()?;
    let slot_offset = usize::try_from(hdr.token_slots_offset)
        .map_err(|e| ManagerError::Protocol(format!("invalid token_slots_offset: {e}")))?;

    let slot_hdr_slice = arena.slice(slot_offset, 16)?;
    let slot_hdr = TokenInputSlotHeader::ref_from_bytes(slot_hdr_slice)
        .map_err(|e| ManagerError::Protocol(format!("failed to cast TokenInputSlotHeader: {e}")))?;
    let token_count = slot_hdr.token_len as usize;

    let tokens_data_offset = slot_offset + 16;
    let tokens_slice = arena.slice(tokens_data_offset, token_count * 4)?;

    let arena_fd_owner = Arc::new(arena.duplicate_fd()?);
    // SAFETY: tokens_slice points to token_count valid u32 values backed by arena_fd_owner.
    unsafe {
        wrap_tokens_as_uint32_array(
            arena_fd_owner,
            tokens_slice.as_ptr().cast::<u32>(),
            token_count,
        )
        .map_err(ManagerError::from)
    }
}

/// Canonical destination subsystems for deterministic [`SystemEventPayload`] sinks.
pub const VALID_EVENT_TARGET_SUBSYSTEMS: [&str; 3] = ["ActionRunner", "MuConDe", "MuIsolation"];

/// Default number of highest-magnitude causal features attached to emitted system events.
pub const DEFAULT_TOP_CAUSAL_FEATURES: usize = 3;

/// Validates that `target_subsystem` is one of the allowed control-plane destinations.
///
/// # Errors
/// Returns [`K3Error::CapabilityMismatch`] if `target_subsystem` is not `"ActionRunner"`,
/// `"MuConDe"`, or `"MuIsolation"`.
pub fn validate_target_subsystem(target_subsystem: &str) -> Result<(), K3Error> {
    if VALID_EVENT_TARGET_SUBSYSTEMS.contains(&target_subsystem) {
        Ok(())
    } else {
        Err(K3Error::CapabilityMismatch {
            architecture: "machine_event_sink".to_owned(),
            reason: format!(
                "unsupported target_subsystem '{target_subsystem}' (expected one of {VALID_EVENT_TARGET_SUBSYSTEMS:?})"
            ),
        })
    }
}

/// Deterministic control-plane event emitted by Sink A from machine-model inference outputs.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SystemEventPayload {
    /// Deterministic event code identifying the requested action (e.g., `"STAGE_RESCHEDULE"`).
    pub event_code: String,
    /// Target subsystem identifier: `"ActionRunner"`, `"MuConDe"`, or `"MuIsolation"`.
    pub target_subsystem: String,
    /// Finite score/confidence emitted by the machine model at trigger time.
    pub confidence: f32,
    /// Top-K causal features ranked descending by normalized z-score magnitude.
    pub top_causal_features: Vec<(String, f32)>,
    /// Event timestamp in milliseconds.
    pub timestamp_ms: u64,
}

impl SystemEventPayload {
    /// Validates structural invariants of this system event payload.
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] if fields violate schema bounds, non-finite values
    /// appear, or `target_subsystem` is unsupported.
    pub fn validate(&self) -> Result<(), K3Error> {
        if self.event_code.trim().is_empty() {
            return Err(K3Error::CapabilityMismatch {
                architecture: "machine_event_sink".to_owned(),
                reason: "SystemEventPayload.event_code must not be empty".to_owned(),
            });
        }
        validate_target_subsystem(&self.target_subsystem)?;
        if !self.confidence.is_finite() {
            return Err(K3Error::CapabilityMismatch {
                architecture: "machine_event_sink".to_owned(),
                reason: format!(
                    "SystemEventPayload.confidence must be finite, got {}",
                    self.confidence
                ),
            });
        }
        for (name, mag) in &self.top_causal_features {
            if name.trim().is_empty() || !mag.is_finite() {
                return Err(K3Error::CapabilityMismatch {
                    architecture: "machine_event_sink".to_owned(),
                    reason: format!("invalid causal feature entry ('{name}', {mag})"),
                });
            }
        }
        Ok(())
    }
}

/// Stateful hysteresis and cooldown translator converting machine-model scores into
/// deterministic [`SystemEventPayload`] notifications without event storms.
#[derive(Debug, Clone)]
pub struct HysteresisEventTranslator {
    /// Threshold at or above which the translator enters active event state.
    pub enter_threshold: f32,
    /// Threshold at or below which the translator clears active state (`exit_threshold < enter_threshold`).
    pub exit_threshold: f32,
    /// Minimum cooldown period in milliseconds between consecutive emitted events.
    pub cooldown_ms: u64,
    /// Minimum number of hits above `enter_threshold` (`k`) required inside `persistence_window` (`m`).
    pub min_persistence_ticks: u32,
    /// Rolling tick window length (`m >= min_persistence_ticks`) evaluated before triggering.
    pub persistence_window: u32,
    /// Number of highest-magnitude causal features attached to emitted events.
    pub top_k_features: usize,
    /// Canonical event code attached to emitted [`SystemEventPayload`]s.
    pub event_code: String,
    /// Canonical target subsystem (`"ActionRunner"`, `"MuConDe"`, or `"MuIsolation"`).
    pub target_subsystem: String,
    feature_names: Vec<String>,
    zscore_mean: Vec<f32>,
    zscore_std: Vec<f32>,
    active: bool,
    last_triggered_ms: Option<u64>,
    recent_hits: VecDeque<bool>,
}

impl HysteresisEventTranslator {
    /// Constructs a validated [`HysteresisEventTranslator`] with single-tick triggering (`k=1, m=1`)
    /// and [`DEFAULT_TOP_CAUSAL_FEATURES`] attribution.
    ///
    /// # Errors
    /// Returns [`ManagerError::TensorBounds`] if thresholds are non-finite or `enter_threshold <= exit_threshold`,
    /// `event_code` is empty, or `target_subsystem` is invalid.
    pub fn new(
        enter_threshold: f32,
        exit_threshold: f32,
        cooldown_ms: u64,
        event_code: impl Into<String>,
        target_subsystem: impl Into<String>,
    ) -> Result<Self, ManagerError> {
        let event_code = event_code.into();
        let target_subsystem = target_subsystem.into();
        if !enter_threshold.is_finite()
            || !exit_threshold.is_finite()
            || enter_threshold <= exit_threshold
        {
            return Err(ManagerError::TensorBounds(K3Error::CapabilityMismatch {
                architecture: "machine_event_sink".to_owned(),
                reason: format!(
                    "hysteresis thresholds must be finite with enter_threshold ({enter_threshold}) > exit_threshold ({exit_threshold})"
                ),
            }));
        }
        if event_code.trim().is_empty() {
            return Err(ManagerError::TensorBounds(K3Error::CapabilityMismatch {
                architecture: "machine_event_sink".to_owned(),
                reason: "hysteresis event_code must not be empty".to_owned(),
            }));
        }
        validate_target_subsystem(&target_subsystem).map_err(ManagerError::TensorBounds)?;

        Ok(Self {
            enter_threshold,
            exit_threshold,
            cooldown_ms,
            min_persistence_ticks: 1,
            persistence_window: 1,
            top_k_features: DEFAULT_TOP_CAUSAL_FEATURES,
            event_code,
            target_subsystem,
            feature_names: Vec::new(),
            zscore_mean: Vec::new(),
            zscore_std: Vec::new(),
            active: false,
            last_triggered_ms: None,
            recent_hits: VecDeque::with_capacity(1),
        })
    }

    /// Configures `k`-of-`m` persistence (`min_persistence_ticks` out of `persistence_window` ticks)
    /// to suppress single-tick spikes.
    ///
    /// # Errors
    /// Returns [`ManagerError::TensorBounds`] if `min_persistence_ticks == 0` or
    /// `persistence_window < min_persistence_ticks`.
    pub fn with_persistence(
        mut self,
        min_persistence_ticks: u32,
        persistence_window: u32,
    ) -> Result<Self, ManagerError> {
        if min_persistence_ticks == 0 || persistence_window < min_persistence_ticks {
            return Err(ManagerError::TensorBounds(K3Error::CapabilityMismatch {
                architecture: "machine_event_sink".to_owned(),
                reason: format!(
                    "invalid k-of-m persistence ({min_persistence_ticks} of {persistence_window}): require k >= 1 and m >= k"
                ),
            }));
        }
        self.min_persistence_ticks = min_persistence_ticks;
        self.persistence_window = persistence_window;
        self.recent_hits.clear();
        Ok(self)
    }

    /// Sets the maximum number of causal features included in emitted events.
    #[must_use]
    pub fn with_top_k(mut self, top_k_features: usize) -> Self {
        self.top_k_features = top_k_features;
        self
    }

    /// Binds feature column names for causal attribution when inputs are already z-score normalized.
    ///
    /// # Errors
    /// Returns [`ManagerError::TensorBounds`] if any feature name is empty.
    pub fn with_feature_names(mut self, feature_names: Vec<String>) -> Result<Self, ManagerError> {
        if feature_names.iter().any(|name| name.trim().is_empty()) {
            return Err(ManagerError::TensorBounds(K3Error::CapabilityMismatch {
                architecture: "machine_event_sink".to_owned(),
                reason: "feature_names must not contain empty column names".to_owned(),
            }));
        }
        self.feature_names = feature_names;
        Ok(self)
    }

    /// Binds feature names together with baseline `zscore_mean` and `zscore_std` vectors.
    ///
    /// # Errors
    /// Returns [`ManagerError::TensorBounds`] if vector lengths disagree, names are empty,
    /// `zscore_mean` has non-finite values, or `zscore_std` has non-finite/non-positive values.
    pub fn with_feature_stats(
        mut self,
        feature_names: Vec<String>,
        zscore_mean: Vec<f32>,
        zscore_std: Vec<f32>,
    ) -> Result<Self, ManagerError> {
        if feature_names.len() != zscore_mean.len() || feature_names.len() != zscore_std.len() {
            return Err(ManagerError::TensorBounds(K3Error::CapabilityMismatch {
                architecture: "machine_event_sink".to_owned(),
                reason: format!(
                    "feature_names ({}), zscore_mean ({}), and zscore_std ({}) lengths must match",
                    feature_names.len(),
                    zscore_mean.len(),
                    zscore_std.len()
                ),
            }));
        }
        if feature_names.iter().any(|name| name.trim().is_empty()) {
            return Err(ManagerError::TensorBounds(K3Error::CapabilityMismatch {
                architecture: "machine_event_sink".to_owned(),
                reason: "feature_names must not contain empty column names".to_owned(),
            }));
        }
        for (idx, (&mean, &std)) in zscore_mean.iter().zip(&zscore_std).enumerate() {
            if !mean.is_finite() || !std.is_finite() || std <= 0.0 {
                return Err(ManagerError::TensorBounds(K3Error::CapabilityMismatch {
                    architecture: "machine_event_sink".to_owned(),
                    reason: format!(
                        "invalid z-score stats at feature index {idx}: mean={mean}, std={std}"
                    ),
                }));
            }
        }
        self.feature_names = feature_names;
        self.zscore_mean = zscore_mean;
        self.zscore_std = zscore_std;
        Ok(self)
    }

    /// Returns whether baseline z-score mean/std stats are bound on this translator.
    #[must_use]
    pub const fn has_zscore_stats(&self) -> bool {
        !self.zscore_mean.is_empty() && !self.zscore_std.is_empty()
    }

    /// Returns whether the translator is currently latched in its active state.
    #[must_use]
    pub const fn is_active(&self) -> bool {
        self.active
    }

    /// Returns the timestamp in milliseconds of the last emitted event, if any.
    #[must_use]
    pub const fn last_triggered_ms(&self) -> Option<u64> {
        self.last_triggered_ms
    }

    /// Resets active state, hit window, and cooldown tracking.
    pub fn reset(&mut self) {
        self.active = false;
        self.last_triggered_ms = None;
        self.recent_hits.clear();
    }

    /// Evaluates a scalar anomaly score at `now_ms`, returning `true` only when hysteresis
    /// transitions into an emitted event outside the cooldown window.
    pub fn evaluate_score(&mut self, score: f32, now_ms: u64) -> bool {
        if !self.enter_threshold.is_finite()
            || !self.exit_threshold.is_finite()
            || self.enter_threshold <= self.exit_threshold
            || !score.is_finite()
        {
            return false;
        }

        if self.active {
            if score <= self.exit_threshold {
                self.active = false;
                self.recent_hits.clear();
            }
            return false;
        }

        let window = usize::try_from(self.persistence_window.max(1)).unwrap_or(1);
        if self.recent_hits.len() == window {
            self.recent_hits.pop_front();
        }
        self.recent_hits.push_back(score >= self.enter_threshold);

        let hits = u32::try_from(self.recent_hits.iter().filter(|&&h| h).count()).unwrap_or(0);
        if hits < self.min_persistence_ticks.max(1) {
            return false;
        }

        let cooldown_ready = self
            .last_triggered_ms
            .is_none_or(|prev| now_ms.saturating_sub(prev) >= self.cooldown_ms);
        if !cooldown_ready {
            return false;
        }

        self.active = true;
        self.last_triggered_ms = Some(now_ms);
        true
    }

    /// Ranks the top-`k` features deterministically by descending normalized z-score magnitude.
    ///
    /// Returns `None` if any feature value or provided normalization parameter is invalid.
    #[must_use]
    pub fn rank_top_causal_features(
        features: &[f32],
        feature_names: &[String],
        zscore_mean: &[f32],
        zscore_std: &[f32],
        top_k: usize,
    ) -> Option<Vec<(String, f32)>> {
        if !features.iter().all(|v| v.is_finite()) {
            return None;
        }
        let use_stats = !zscore_mean.is_empty()
            && zscore_mean.len() == features.len()
            && zscore_std.len() == features.len();
        if !use_stats && (!zscore_mean.is_empty() || !zscore_std.is_empty()) {
            return None;
        }

        let mut ranked: Vec<(usize, String, f32)> = Vec::with_capacity(features.len());
        for (idx, &val) in features.iter().enumerate() {
            let mag = if use_stats {
                let mean = zscore_mean[idx];
                let std = zscore_std[idx];
                if !mean.is_finite() || !std.is_finite() || std <= 0.0 {
                    return None;
                }
                let z = (val - mean) / std;
                if !z.is_finite() {
                    return None;
                }
                z.abs()
            } else {
                val.abs()
            };
            let name = feature_names
                .get(idx)
                .cloned()
                .unwrap_or_else(|| format!("feature_{idx}"));
            ranked.push((idx, name, mag));
        }

        ranked.sort_by(|a, b| b.2.total_cmp(&a.2).then_with(|| a.0.cmp(&b.0)));
        ranked.truncate(top_k);
        Some(
            ranked
                .into_iter()
                .map(|(_, name, mag)| (name, mag))
                .collect(),
        )
    }

    /// Evaluates model output scores and input features at `now_ms`, emitting a
    /// [`SystemEventPayload`] with top-K causal feature attribution when hysteresis triggers.
    ///
    /// Fails closed (returns `None` without mutating translator state) if inputs contain
    /// non-finite values, `outputs` is empty, or bound normalization stats mismatch `features`.
    pub fn evaluate_event(
        &mut self,
        outputs: &[f32],
        features: &[f32],
        now_ms: u64,
    ) -> Option<SystemEventPayload> {
        if outputs.is_empty()
            || !outputs.iter().all(|v| v.is_finite())
            || self.event_code.trim().is_empty()
            || validate_target_subsystem(&self.target_subsystem).is_err()
        {
            return None;
        }

        let top_causal_features = Self::rank_top_causal_features(
            features,
            &self.feature_names,
            &self.zscore_mean,
            &self.zscore_std,
            self.top_k_features,
        )?;

        let confidence = outputs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if !self.evaluate_score(confidence, now_ms) {
            return None;
        }

        Some(SystemEventPayload {
            event_code: self.event_code.clone(),
            target_subsystem: self.target_subsystem.clone(),
            confidence,
            top_causal_features,
            timestamp_ms: now_ms,
        })
    }

    /// Explicit error-returning variant of [`Self::evaluate_event`] using caller-supplied
    /// schema names and z-score statistics (ideal for strict HTTP/IPC request validation).
    ///
    /// # Errors
    /// Returns [`ManagerError::TensorBounds`] if `outputs` is empty or non-finite, `features` has
    /// non-finite values or length mismatches, or translator configuration is invalid.
    pub fn evaluate_event_with_stats(
        &mut self,
        outputs: &[f32],
        features: &[f32],
        feature_names: &[String],
        zscore_mean: &[f32],
        zscore_std: &[f32],
        now_ms: u64,
    ) -> Result<Option<SystemEventPayload>, ManagerError> {
        if outputs.is_empty() || !outputs.iter().all(|v| v.is_finite()) {
            return Err(ManagerError::TensorBounds(K3Error::CapabilityMismatch {
                architecture: "machine_event_sink".to_owned(),
                reason: "model outputs must be non-empty and finite".to_owned(),
            }));
        }
        validate_target_subsystem(&self.target_subsystem).map_err(ManagerError::TensorBounds)?;

        let top_causal_features = Self::rank_top_causal_features(
            features,
            feature_names,
            zscore_mean,
            zscore_std,
            self.top_k_features,
        )
        .ok_or_else(|| {
            ManagerError::TensorBounds(K3Error::CapabilityMismatch {
                architecture: "machine_event_sink".to_owned(),
                reason: "features or normalization stats contain non-finite or mismatched values"
                    .to_owned(),
            })
        })?;

        let confidence = outputs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if !self.evaluate_score(confidence, now_ms) {
            return Ok(None);
        }

        Ok(Some(SystemEventPayload {
            event_code: self.event_code.clone(),
            target_subsystem: self.target_subsystem.clone(),
            confidence,
            top_causal_features,
            timestamp_ms: now_ms,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hysteresis_enter_exit_cooldown() {
        let mut tr =
            HysteresisEventTranslator::new(0.80, 0.40, 1000, "STAGE_RESCHEDULE", "ActionRunner")
                .expect("valid translator");

        // Rising above enter threshold triggers once
        assert!(!tr.evaluate_score(0.79, 100));
        assert!(tr.evaluate_score(0.85, 200));
        assert!(tr.is_active());

        // Inside deadband or staying above enter never double-fires
        assert!(!tr.evaluate_score(0.92, 300));
        assert!(!tr.evaluate_score(0.55, 500));
        assert!(tr.is_active());
        assert!(!tr.evaluate_score(0.88, 700));

        // Dropping to or below exit threshold disarms active state
        assert!(!tr.evaluate_score(0.40, 800));
        assert!(!tr.is_active());

        // Re-crossing enter threshold inside cooldown (900 - 200 = 700 < 1000) is suppressed
        assert!(!tr.evaluate_score(0.90, 900));
        assert!(!tr.is_active());

        // Once cooldown elapses (1200 - 200 >= 1000), crossing enter emits again
        assert!(tr.evaluate_score(0.90, 1200));
        assert_eq!(tr.last_triggered_ms(), Some(1200));

        // Non-finite score fails closed without mutating active state
        assert!(!tr.evaluate_score(f32::NAN, 1300));
        assert!(tr.is_active());
        assert!(!tr.evaluate_score(f32::INFINITY, 1400));
    }

    #[test]
    fn test_hysteresis_cooldown_event_suppression() {
        let mut tr = HysteresisEventTranslator::new(0.75, 0.35, 5000, "CANARY_ROLLBACK", "MuConDe")
            .expect("valid translator")
            .with_feature_names(vec![
                "p99_latency_z".to_owned(),
                "error_rate_z".to_owned(),
                "mem_rss_z".to_owned(),
            ])
            .expect("valid names");

        let ev1 = tr
            .evaluate_event(&[0.82], &[1.2, 4.1, 0.5], 10_000)
            .expect("first anomaly crossing triggers");
        assert_eq!(ev1.target_subsystem, "MuConDe");
        assert_eq!(ev1.event_code, "CANARY_ROLLBACK");
        assert_eq!(ev1.timestamp_ms, 10_000);

        // Exit below 0.35 then spike again within cooldown_ms (11_500 - 10_000 < 5000)
        assert!(
            tr.evaluate_event(&[0.20], &[0.1, 0.2, 0.1], 11_000)
                .is_none()
        );
        assert!(
            tr.evaluate_event(&[0.95], &[3.0, 5.0, 1.0], 11_500)
                .is_none()
        );

        // After cooldown expires (15_000 - 10_000 >= 5000), anomaly triggers cleanly
        let ev2 = tr
            .evaluate_event(&[0.91], &[2.5, 4.8, 0.9], 15_000)
            .expect("triggers once cooldown elapsed");
        assert_eq!(ev2.timestamp_ms, 15_000);
    }

    #[test]
    fn test_system_event_payload_causal_attribution() {
        let mut tr = HysteresisEventTranslator::new(0.70, 0.30, 500, "MEM_CLAMP", "MuIsolation")
            .expect("valid translator")
            .with_top_k(2)
            .with_feature_stats(
                vec![
                    "cpu_util".to_owned(),
                    "queue_depth".to_owned(),
                    "rss_mb".to_owned(),
                ],
                vec![50.0, 10.0, 1024.0],
                vec![10.0, 2.0, 128.0],
            )
            .expect("valid stats");

        // Z-scores: cpu = |55-50|/10 = 0.5, queue_depth = |2-10|/2 = 4.0, rss_mb = |1408-1024|/128 = 3.0
        let ev = tr
            .evaluate_event(&[0.12, 0.89], &[55.0, 2.0, 1408.0], 2000)
            .expect("emits attributed system event");
        ev.validate().expect("payload valid");
        assert_eq!(ev.target_subsystem, "MuIsolation");
        assert!((ev.confidence - 0.89).abs() < 1e-6);
        assert_eq!(ev.top_causal_features.len(), 2);
        assert_eq!(ev.top_causal_features[0].0, "queue_depth");
        assert!((ev.top_causal_features[0].1 - 4.0).abs() < 1e-5);
        assert_eq!(ev.top_causal_features[1].0, "rss_mb");
        assert!((ev.top_causal_features[1].1 - 3.0).abs() < 1e-5);

        // NaN feature must fail closed without latching active
        assert!(
            tr.evaluate_event(&[0.95], &[55.0, f32::NAN, 1408.0], 3000)
                .is_none()
        );
    }

    #[test]
    fn test_hysteresis_system_event_translator() {
        // Reject invalid target subsystem and invalid threshold ordering upfront
        assert!(
            HysteresisEventTranslator::new(0.5, 0.6, 100, "EVT", "ActionRunner").is_err(),
            "enter <= exit must be rejected"
        );
        assert!(
            HysteresisEventTranslator::new(0.8, 0.4, 100, "EVT", "UnknownSink").is_err(),
            "unsupported target subsystem must be rejected"
        );

        // k=2 of m=3 persistence suppresses single-tick transient spike
        let mut persistent =
            HysteresisEventTranslator::new(0.8, 0.3, 1000, "STAGE_RESCHEDULE", "ActionRunner")
                .expect("valid translator")
                .with_persistence(2, 3)
                .expect("valid persistence");

        assert!(persistent.evaluate_event(&[0.95], &[3.0], 100).is_none());
        assert!(persistent.evaluate_event(&[0.20], &[0.1], 200).is_none());
        // Single spike cooled off; second elevated tick within 3-tick window reaches 2-of-3
        let emitted = persistent
            .evaluate_event(&[0.91], &[4.2], 300)
            .expect("sustained 2-of-3 ticks emits event");
        assert_eq!(emitted.target_subsystem, "ActionRunner");
        emitted.validate().expect("payload valid");
    }
}

//! `k3-manager`: Control plane and IPC daemon for the `SpacemiT` K3 AI platform.
//!
//! Supervises isolated `k3-worker` execution domains (ADR-018, ADR-041),
//! communicates with `k3-kv` context memory holder (ADR-042),
//! and orchestrates zero-copy forward pass dispatch and crash recovery (US-05).

pub mod capacity;
pub mod client;
pub mod dispatch;
pub mod domain;
pub mod error;
pub mod flight;
pub mod health;
pub mod ingestion;
pub mod persistence;
pub mod prefix;
pub mod prompt;
pub mod protocol;
pub mod queue;
pub mod router;
pub mod runtime;
pub mod scheduler;
pub mod session;
pub mod store;
pub mod supervisor;
pub mod telemetry;

pub use flight::{
    BOUNDED_MEMLOCK_CEILING_BYTES, FlightError, FlightReceiver, FlightStream, FlightSummary,
    FlightTensorBatchStreamer, IME_TILE_ELEMENTS, MemfdArena, TYPE_ID_FLIGHT_ARENA,
    flight_tensor_schema,
};
pub use ingestion::{
    AuthStatus, GgufHeaderPrefix, HuggingFaceAdapter, IngestionError, LocalFileAdapter,
    ModelFormat, ModelRevision, ModelVersionListing, OllamaAdapter, PullArtifactSummary,
    PullProgress, PullRequest, RemoteModelSummary, SafeTensorsHeaderPrefix, SearchFilter,
    SourceAdapter, SourceAdapterRegistry, SourceCredentials, StagingDownload, StagingPipeline,
};

pub use store::{
    AlignedPromptEnvironment, BlobUsageInfo, CandidateSuitability, CapabilityFilter,
    ContentAddressedStore, DEFAULT_STORE_ROOT, DerivedBlobManager, DerivedResolution,
    ENV_STORE_ROOT, GcReport, HighWaterMarkWarning, MAX_PATH_COMPONENT_LEN, MockProcMapsScanner,
    ModelManifest, ProcessMappingInfo, ProcessMapsScanner, RealProcMapsScanner, RecipeDescriptor,
    ResolvedAlias, RetentionPolicy, Sha256, StaleRecipeOutcome, StorageGarbageCollector,
    check_derived_memory_admission, hex_encode, parse_alias_id, parse_llamaguard_hazard,
    repack_phi3_fused_tensor_ime_8x16, validate_path_component,
};

pub use protocol::{
    ClaudeAdapter, ClaudeContent, ClaudeContentBlock, ClaudeErrorDetails, ClaudeErrorResponse,
    ClaudeMessage, ClaudeMessageRequest, ClaudeMessageResponse, ClaudeSystem, ClaudeThinkingConfig,
    ClaudeTool, ClaudeUsage, GuardrailEvaluateRequest, GuardrailEvaluateResponse, GuardrailPolicy,
    OpenAiAdapter, OpenAiChatRequest, OpenAiChatResponse, OpenAiChoice, OpenAiChunkChoice,
    OpenAiChunkDelta, OpenAiChunkResponse, OpenAiErrorDetails, OpenAiErrorResponse, OpenAiFunction,
    OpenAiFunctionCall, OpenAiMessage, OpenAiResponseMessage, OpenAiStop, OpenAiTool,
    OpenAiToolCall, OpenAiUsage, ProtocolServer, ProtocolServerConfig, ThinkingTagParser,
    audit_generated_tool_call, handle_guardrail_evaluate,
};

pub use capacity::{
    A100_CLUSTER_MASK, A100_DEFAULT_MAX_COMPUTE_THREADS, ActiveControlRotator,
    ControlPlaneCapacityConfig, ControlPlaneScheduler, DEFAULT_A100_MAX_COMPUTE_CORES,
    DEFAULT_ACTIVE_CONTROL_MASK, DEFAULT_MAX_A100_COMPUTE_CORES, DEFAULT_MAX_X100_COMPUTE_CORES,
    DEFAULT_SPARE_RECOVERY_MASK, DEFAULT_X100_COMPUTE_CAP_MASK, DEFAULT_X100_MAX_COMPUTE_CORES,
    ResolvedControlCapacity, ResolvedControlCapacityExt, SOC_CLUSTER_MASK, X100_CLUSTER_MASK,
    X100_DEFAULT_MAX_COMPUTE_THREADS,
};
pub use client::KvClient;
pub use dispatch::{
    ForwardPassRequest, ForwardPassResult, dispatch_forward_pass, wrap_input_tokens_as_arrow,
    wrap_logits_as_float32_array,
};
pub use domain::{
    CgroupFs, CgroupV2DomainController, DomainPlacementMode, IsolationDomainController,
    MockCgroupFs, MockOsAffinityBackend, OsAffinityBackend, PlacementOutcome, RealCgroupFs,
    RealOsAffinityBackend, format_cpuset_list, parse_cpuset_list, resolve_worker_compute_mask,
    validate_model_id, validate_tuning_profile_dory_overrides,
};
pub use error::ManagerError;
pub use health::{
    HealthSupervisor, HealthSupervisorConfig, RecoveryAction, RecoveryAuditEvent, RecoveryTarget,
};
pub use jero_core::profile::{
    CONSERVATIVE_DEFAULT_A100_CORES, CONSERVATIVE_DEFAULT_N_CTX, CONSERVATIVE_DEFAULT_X100_CORES,
    CONSERVATIVE_TUNING_ADVISORY, CONSTRAINED_TEST_N_CTX, ComputePool, CostProfile, ModelEntry,
    ModelListResponse, ModelLoadAuditEvent, ModelProfile, PriorityClass, ProfileStatus,
    TuningProfile,
};
pub use persistence::{
    BlobReferenceRecord, CasCatalog, DerivedBlobRecord, DiagnosticReplayResult,
    DuckDbPersistenceStore, ModelAliasRecord, ModelBlobRecord, ModelManifestRecord, OptionalExt,
    OptionalExtension, PersistenceError, PersistenceStore, Project, SessionLastTurn,
    SqlitePersistenceStore, Tenant,
};
pub use prefix::{
    BLOCK_SIZE_TOKENS, BlockTokens, ModelBuildScope, PrefixCacheStats, PrefixKey,
    PrefixMatchResult, PrefixRadixIndex,
};
pub use prompt::{
    ChatTemplate, DORY_BOS_TOKEN, DORY_EOS_TOKEN, DORY_MEDIUM_REASONING_SUFFIX, DetokenizeStep,
    FUNCTION_GEMMA_CALL_END, FUNCTION_GEMMA_CALL_START, GgufTokenizer, IncrementalStopDetector,
    OCTOPUS_NEXA_END_TOKEN, PreparedPrompt, PromptOrchestrator, PromptPreparationRequest,
    SimpleTokenizer, SpecialTokens, StreamingDetokenizer, TokenType, Tokenizer,
    XLAM_TOOL_CALL_TOKEN, evaluate_stop_token, format_chat_prompt, format_dory_chatml_prompt,
    format_functiongemma_prompt, format_octopus_prompt, format_xlam_prompt,
    is_functional_action_token, normalize_guardrail_input, resolve_dory_loop_budget,
    validate_and_tokenize,
};
pub use queue::{
    AdmissionQueueConfig, AdmissionQueueManager, DEFAULT_ESTIMATED_PASS_LATENCY_MS,
    DEFAULT_KEEP_ALIVE_INTERVAL, DEFAULT_MAX_QUEUE_DEPTH, DEFAULT_MIN_RETRY_AFTER_SECS,
    DEFAULT_PASSES_PER_REQUEST, ModelAdmissionQueue, QueueMetrics, QueuePermit, QueueSseFrame,
    QueuedRequest,
};
pub use router::{
    DecisionRouter, ExecutionAssessment, FunctionalTokenMap, ModelCapability, ModelTarget,
    RouterConfig, RoutingDecision, RoutingRequest, StructuralConstraint,
    resolve_functional_token_action,
};
pub use runtime::{
    ActiveTaskRecord, AdminCommand, AdminResponse, ControlPlaneRuntime, DEFAULT_CONTROL_TASK_LEASE,
    GuardrailEvaluatorFn, ModelAuditEvent, ModelAuditEventType, ModelPlacementRecord,
    TaskLeaseSnapshot, WorkerHealthSnapshot,
};
pub use scheduler::{
    DEFAULT_MAX_ACTIVE_MODELS, ModelExecutionMetrics, ModelSchedulerEntry, ModelSchedulingState,
    MultiModelScheduler, ScheduledPass,
};
pub use session::{
    DEFAULT_BROADCAST_CAPACITY, DEFAULT_RETENTION_WINDOW, SessionError, SessionRegistry,
    SessionState,
};
pub use supervisor::{
    DEFAULT_HEARTBEAT_TIMEOUT, DEFAULT_PASS_TIMEOUT, ManagedWorker, ResumptionSnapshot,
    WorkerPlacement, WorkerSupervisor, find_worker_binary,
};
pub use telemetry::{
    A100_VLEN_BITS, BOUNDED_MEMLOCK_CEILING_KB, CollectorEndpoint, DEFAULT_COLLECTOR_TIMEOUT,
    DEFAULT_MAX_BUFFERED_BATCHES, DEFAULT_TELEMETRY_ARCH, DEFAULT_TELEMETRY_NODE_ID,
    FramedMuobBatch, InferenceSpanStage, InferenceTelemetryRecord, K3ClusterTarget,
    MSG_TYPE_ARROW_OTEL_METRICS, MSG_TYPE_ARROW_OTEL_SPANS, MSG_TYPE_WAL_ACK,
    MuObserveTelemetryStreamer, OtelMetricEntry, OtelSpanEntry, SERVICE_NAME_K3_MANAGER,
    TelemetryIpcError, TraceContext, WalBatchAck, X100_VLEN_BITS, decode_metric_batch,
    decode_span_batch, encode_metric_batch, encode_span_batch, estimate_bounded_kv_memlock_kb,
    resolve_or_create_trace_context,
};

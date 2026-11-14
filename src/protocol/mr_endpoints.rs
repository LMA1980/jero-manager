//! Role-gated `Model-Runner` telemetry and connection endpoints (`#74`–`#78`).
//!
//! Exposes read-only control-plane endpoints directly on [`crate::protocol::ProtocolServer`]:
//! - `GET /api/mr/v1/status` (`folio:mr:read`: `mr:models.read` + `mr:queue.read`)
//! - `GET /api/mr/v1/connection` (`folio:mr:connect`: `mr:openai_api.read` + `mr:wireguard_vpn.read`)
//!
//! Uses `mu-auth`'s framework-agnostic header-slice verifier ([`GithubEdgeAuthAdapter`]) and
//! fail-closed session evaluation ([`AuthAuthority::evaluate_session`]) without `axum` or
//! `CAP_NET_ADMIN`.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::net::IpAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, PoisonError, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use mu_auth::{
    AuthAuthority, EdgeAuthConfig, FolioCatalog, FolioError, GithubEdgeAuthAdapter, Principal,
    RoleCatalog, RoleFolioIndex, RoleId, RoleLeases, RoleSpec, SessionEvaluation,
    resolve_secret_value,
};
use serde::{Deserialize, Serialize};

use crate::flight::BOUNDED_MEMLOCK_CEILING_BYTES;
use crate::protocol::types::{HttpRequest, HttpResponse};
use crate::runtime::ControlPlaneRuntime;
use crate::telemetry::estimate_bounded_kv_memlock_kb;

/// Canonical policy ID granting read access to resident models and memlock telemetry.
pub const POLICY_MR_MODELS_READ: &str = "mr:models.read";

/// Canonical policy ID granting read access to inference admission queue depths.
pub const POLICY_MR_QUEUE_READ: &str = "mr:queue.read";

/// Canonical policy ID granting read access to generic OpenAI-compatible API endpoint parameters.
pub const POLICY_MR_OPENAI_API_READ: &str = "mr:openai_api.read";

/// Canonical policy ID granting read access to sanitized WireGuard peer connection configuration.
pub const POLICY_MR_WIREGUARD_VPN_READ: &str = "mr:wireguard_vpn.read";

/// Environment variable supplying shared proxy secret literal or secret file path.
pub const ENV_PROXY_SECRET: &str = "JERO_MANAGER_PROXY_SECRET";

/// Environment variable supplying path to shared proxy secret file.
pub const ENV_PROXY_SECRET_FILE: &str = "JERO_MANAGER_PROXY_SECRET_FILE";

/// Environment variable opting into trusted loopback without `X-Proxy-Secret` (`true`/`1`).
pub const ENV_ALLOW_LOOPBACK_NO_SECRET: &str = "JERO_MANAGER_ALLOW_LOOPBACK_WITHOUT_SECRET";

/// Environment variable pointing to a JSON-encoded [`ModelRunnerAuthBundle`] policy store.
pub const ENV_AUTH_POLICY_FILE: &str = "JERO_MANAGER_AUTH_POLICY_FILE";

/// Environment variable overriding reported bounded `RLIMIT_MEMLOCK` ceiling in bytes.
pub const ENV_MEMLOCK_CEILING_BYTES: &str = "JERO_MANAGER_MEMLOCK_CEILING_BYTES";

/// Environment variable configuring advertised OpenAI-compatible base URL.
pub const ENV_OPENAI_BASE_URL: &str = "JERO_MANAGER_OPENAI_BASE_URL";

/// Environment variable supplying comma-separated OpenAI-compatible routes.
pub const ENV_OPENAI_ROUTES: &str = "JERO_MANAGER_OPENAI_ROUTES";

/// Environment variable supplying client auth header usage instructions.
pub const ENV_AUTH_HEADER_GUIDE: &str = "JERO_MANAGER_AUTH_HEADER_GUIDE";

/// Environment variable configuring generic WireGuard peer endpoint (`host:port`).
pub const ENV_WIREGUARD_ENDPOINT: &str = "JERO_MANAGER_WIREGUARD_ENDPOINT";

/// Environment variable configuring generic WireGuard tunnel interface CIDR.
pub const ENV_WIREGUARD_TUNNEL_CIDR: &str = "JERO_MANAGER_WIREGUARD_TUNNEL_CIDR";

/// Environment variable configuring fallback WireGuard interface state when no file is present.
pub const ENV_WIREGUARD_INTERFACE_STATE: &str = "JERO_MANAGER_WIREGUARD_INTERFACE_STATE";

/// Environment variable pointing to the rootless WireGuard status JSON file.
pub const ENV_WIREGUARD_STATUS_FILE: &str = "JERO_MANAGER_WIREGUARD_STATUS_FILE";

const PUBLIC_KEY_PREFIX_CHARS: usize = 8;

/// Resident model memory and queue status (`GET /api/mr/v1/status`, `mu-console-protocol` aligned).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResidentModelStatus {
    /// Logical model identifier.
    pub model_id: String,
    /// Placement cluster (`"x100"`, `"a100"`, or `"both"`).
    pub cluster: String,
    /// Whether the supervised worker is alive.
    pub is_alive: bool,
    /// Active context window limit in tokens.
    pub n_ctx: u32,
    /// Estimated or reported memory-locked bytes for this placement (`0` when offline).
    pub mlock_bytes: u64,
    /// Bounded memlock ceiling enforced per placement/host in bytes.
    pub memlock_ceiling_bytes: u64,
    /// Active queued request depth for this model.
    pub queue_depth: u32,
}

/// Aggregate Model-Runner runtime telemetry (`GET /api/mr/v1/status`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelRunnerStatusResponse {
    /// Aggregate prompt queue depth across all registered model queues.
    pub total_queue_depth: u32,
    /// Whether any model queue is currently executing a forward pass.
    pub is_executing: bool,
    /// Worst-case estimated queue wait time across queues in milliseconds.
    pub estimated_wait_ms: u64,
    /// Sum of locked memory bytes across resident placements.
    pub mlock_used_bytes: u64,
    /// Bounded host `RLIMIT_MEMLOCK` ceiling in bytes (`28 GiB` default on 32 GiB target).
    pub memlock_ceiling_bytes: u64,
    /// Per-model status breakdown sorted deterministically by `model_id`.
    pub models: Vec<ResidentModelStatus>,
}

/// Redacted WireGuard peer status summary (`GET /api/mr/v1/connection`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WireGuardPeerSummary {
    /// Truncated peer public key prefix (`<8 chars>...`, never full key or private material).
    pub public_key_prefix: String,
    /// Peer AllowedIPs CIDR list.
    pub allowed_ips: String,
    /// Latest handshake UNIX timestamp in milliseconds (`None` if no handshake recorded).
    pub latest_handshake_epoch_ms: Option<u64>,
    /// Received bytes reported by the rootless status exporter.
    pub rx_bytes: u64,
    /// Transmitted bytes reported by the rootless status exporter.
    pub tx_bytes: u64,
}

/// Generic OpenAI API & WireGuard connection response (`GET /api/mr/v1/connection`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelRunnerConnectionResponse {
    /// Generic OpenAI-compatible API base URL configured for clients.
    pub openai_base_url: String,
    /// Available OpenAI-compatible HTTP routes.
    pub openai_routes: Vec<String>,
    /// Human-readable client authentication header guidance.
    pub auth_header_guide: String,
    /// WireGuard endpoint (`host:port`, configurable; empty/unconfigured if unset).
    pub wireguard_endpoint: String,
    /// Tunnel interface CIDR (configurable; empty if unset).
    pub tunnel_cidr: String,
    /// WireGuard interface state (`"up"`, `"down"`, `"unconfigured"`, or `"unavailable"`).
    pub interface_state: String,
    /// Sanitized peer summaries parsed from the rootless WireGuard JSON file.
    pub peers: Vec<WireGuardPeerSummary>,
}

/// Server-side operator role bindings and active leases for a verified GitHub login.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct RegisteredOperator {
    /// Enrolled WebAuthn Passkeys (`0` fails closed via mandatory Passkey gate).
    pub passkey_count: usize,
    /// Persistent role bindings assignable to this principal (in addition to canonical baseline).
    pub bound_roles: BTreeSet<RoleId>,
    /// Active server-managed `Elective` toggles and `Timed` leases.
    pub leases: RoleLeases,
}

impl RegisteredOperator {
    /// Creates an operator record with Passkey enrollment count, bound roles, and leases.
    #[must_use]
    pub const fn new(
        passkey_count: usize,
        bound_roles: BTreeSet<RoleId>,
        leases: RoleLeases,
    ) -> Self {
        Self {
            passkey_count,
            bound_roles,
            leases,
        }
    }
}

/// Serializable authorization snapshot combining `mu-auth` authority and bound operators.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelRunnerAuthBundle {
    /// Validated folio, role, index, and subsystem suspension state.
    pub authority: AuthAuthority,
    /// Case-normalized (lowercase GitHub login) operator records.
    pub operators: BTreeMap<String, RegisteredOperator>,
}

impl ModelRunnerAuthBundle {
    /// Validates internal `mu-auth` invariants and normalizes operator keys to lowercase ASCII.
    ///
    /// # Errors
    /// Returns [`FolioError`] if `FolioCatalog`, `RoleCatalog`, or `RoleFolioIndex` invariants fail.
    pub fn into_validated(mut self) -> Result<Self, FolioError> {
        self.authority.folio_catalog.validate_graph()?;
        self.authority.role_catalog.validate()?;
        self.authority.role_folio_index.validate(
            &self.authority.folio_catalog,
            Some(&self.authority.role_catalog),
        )?;

        let mut normalized = BTreeMap::new();
        for (login, rec) in self.operators {
            normalized.insert(login.trim().to_ascii_lowercase(), rec);
        }
        self.operators = normalized;
        Ok(self)
    }

    /// Builds an empty catalog using canonical subsystem manifests and an arbitrary validated [`RoleCatalog`] / [`RoleFolioIndex`].
    ///
    /// # Errors
    /// Returns [`FolioError`] when canonical manifests or supplied catalogs fail validation.
    pub fn with_canonical_folios(
        role_catalog: RoleCatalog,
        role_folio_index: RoleFolioIndex,
    ) -> Result<Self, FolioError> {
        let folio_catalog = FolioCatalog::with_canonical_manifests();
        let authority = AuthAuthority::new(folio_catalog, role_catalog, role_folio_index)?;
        Ok(Self {
            authority,
            operators: BTreeMap::new(),
        })
    }
}

/// Deterministic wall-clock abstraction so unit/integration tests can test lease expiry without sleeps.
#[derive(Debug, Clone, Default)]
pub enum AuthClock {
    /// Uses system UNIX epoch time in milliseconds.
    #[default]
    System,
    /// Uses caller-controlled millisecond timestamp atomically.
    Fixed(Arc<AtomicU64>),
}

impl AuthClock {
    #[must_use]
    fn now_ms(&self) -> u64 {
        match self {
            Self::System => SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX)),
            Self::Fixed(clock) => clock.load(Ordering::SeqCst),
        }
    }
}

/// Edge + folio authorization policy enforced on `/api/mr/v1/*` routes.
#[derive(Clone)]
pub struct ModelRunnerAuthPolicy {
    edge_adapter: Option<GithubEdgeAuthAdapter>,
    bundle: Arc<RwLock<ModelRunnerAuthBundle>>,
    clock: AuthClock,
}

impl fmt::Debug for ModelRunnerAuthPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ModelRunnerAuthPolicy")
            .field("configured", &self.edge_adapter.is_some())
            .field("clock", &self.clock)
            .finish_non_exhaustive()
    }
}

impl ModelRunnerAuthPolicy {
    /// Builds a fail-closed verifier that rejects all `/api/mr/v1/*` requests with `401 Unauthorized`.
    ///
    /// # Panics
    /// Never panics for validated static canonical baseline constants.
    #[must_use]
    pub fn fail_closed() -> Self {
        let folio_catalog = FolioCatalog::with_canonical_manifests();
        // Minimal baseline-only catalog with zero folios granted to baseline or anonymous.
        let baseline_id = RoleId::new("baseline").expect("valid canonical baseline role id");
        let baseline_role =
            RoleSpec::pinned_baseline(baseline_id, Some("Baseline"), "Minimal canonical baseline")
                .expect("valid canonical baseline spec");
        let role_catalog =
            RoleCatalog::from_roles([baseline_role]).expect("valid minimal baseline catalog");
        let role_folio_index = RoleFolioIndex::default();
        let authority = AuthAuthority::new(folio_catalog, role_catalog, role_folio_index)
            .expect("valid minimal authority");
        Self {
            edge_adapter: None,
            bundle: Arc::new(RwLock::new(ModelRunnerAuthBundle {
                authority,
                operators: BTreeMap::new(),
            })),
            clock: AuthClock::System,
        }
    }

    /// Creates an active verifier backed by [`EdgeAuthConfig`] and a validated [`ModelRunnerAuthBundle`].
    ///
    /// # Errors
    /// Returns [`FolioError`] if `bundle` fails fail-closed validation.
    pub fn new(
        edge_config: EdgeAuthConfig,
        bundle: ModelRunnerAuthBundle,
    ) -> Result<Self, FolioError> {
        let validated = bundle.into_validated()?;
        Ok(Self {
            edge_adapter: Some(GithubEdgeAuthAdapter::new(edge_config)),
            bundle: Arc::new(RwLock::new(validated)),
            clock: AuthClock::System,
        })
    }

    /// Overrides the wall clock (for deterministic TTL / expiry verification).
    #[must_use]
    pub fn with_clock(mut self, clock: AuthClock) -> Self {
        self.clock = clock;
        self
    }

    /// Returns a shared handle to update operator role bindings or `AuthAuthority` state safely.
    #[must_use]
    pub fn bundle_handle(&self) -> Arc<RwLock<ModelRunnerAuthBundle>> {
        Arc::clone(&self.bundle)
    }

    /// Registers or updates a server-side operator record (lowercase-normalized GitHub login).
    pub fn upsert_operator(&self, github_login: &str, record: RegisteredOperator) {
        let mut guard = self.bundle.write().unwrap_or_else(PoisonError::into_inner);
        guard
            .operators
            .insert(github_login.trim().to_ascii_lowercase(), record);
    }

    /// Authorizes `req` against all `required_policies` via `mu-auth` header-slice verification
    /// and fail-closed `evaluate_session`.
    ///
    /// # Errors
    /// Returns an HTTP `401 Unauthorized` response on proxy/authentication failures or `403 Forbidden`
    /// when policy evaluation, passkey enrollment, or required folio grants deny access.
    #[allow(clippy::result_large_err)]
    pub fn authorize_request(
        &self,
        req: &HttpRequest,
        peer_ip: Option<IpAddr>,
        required_policies: &[&str],
    ) -> Result<SessionEvaluation, HttpResponse> {
        let Some(adapter) = &self.edge_adapter else {
            return Err(json_auth_error(
                401,
                "auth_unconfigured",
                "Edge authentication is not configured; failing closed",
            ));
        };

        let header_slices: Vec<(&str, &str)> = req
            .headers
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();

        let identity = adapter
            .authenticate_headers(&header_slices, peer_ip)
            .map_err(|err| json_auth_error(401, "unauthorized", &err.to_string()))?;

        let now_ms = self.clock.now_ms();
        let guard = self.bundle.read().unwrap_or_else(PoisonError::into_inner);

        let (principal, leases) = match &identity {
            None => (Principal::Anonymous, RoleLeases::default()),
            Some(id) => {
                let normalized_login = id.login.to_lowercase();
                let Some(operator) = guard.operators.get(&normalized_login) else {
                    return Err(json_auth_error(
                        403,
                        "forbidden",
                        "Authenticated user is not bound to any platform role",
                    ));
                };
                (
                    Principal::Authenticated {
                        login: normalized_login,
                        passkey_count: operator.passkey_count,
                        bound_roles: operator.bound_roles.clone(),
                    },
                    operator.leases.clone(),
                )
            }
        };

        let eval = guard
            .authority
            .evaluate_session(&principal, &leases, now_ms)
            .map_err(|err| json_auth_error(403, "policy_evaluation_failed", &err.to_string()))?;

        if identity.is_none()
            && !required_policies
                .iter()
                .all(|policy| eval.has_policy_str(policy))
        {
            return Err(json_auth_error(
                401,
                "authentication_required",
                "Authentication via trusted edge proxy is required",
            ));
        }

        if eval.passkey_enrollment_required {
            return Err(json_auth_error(
                403,
                "passkey_enrollment_required",
                "Passkey enrollment is required before Model-Runner policy folios become active",
            ));
        }

        for policy in required_policies {
            if !eval.has_policy_str(policy) {
                return Err(json_auth_error(
                    403,
                    "insufficient_policy",
                    &format!("Missing required authorization policy '{policy}'"),
                ));
            }
        }

        Ok(eval)
    }
}

/// Configuration for `/api/mr/v1/connection` OpenAI API hints and rootless WireGuard JSON reader.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelRunnerConnectionConfig {
    /// Generic OpenAI-compatible API base URL advertised to authorized operators.
    pub openai_base_url: String,
    /// OpenAI-compatible HTTP routes supported by [`crate::protocol::ProtocolServer`].
    pub openai_routes: Vec<String>,
    /// Human-readable authentication header guidance string.
    pub auth_header_guide: String,
    /// Generic WireGuard VPN endpoint (`host:port`, configurable; empty if unset).
    pub wireguard_endpoint: String,
    /// Generic WireGuard tunnel CIDR (configurable; empty if unset).
    pub tunnel_cidr: String,
    /// Fallback interface state reported when no status JSON file is configured.
    pub default_interface_state: String,
    /// Path to rootless WireGuard JSON status file (read without `CAP_NET_ADMIN`).
    pub wireguard_status_path: Option<PathBuf>,
}

impl Default for ModelRunnerConnectionConfig {
    fn default() -> Self {
        Self {
            openai_base_url: "http://127.0.0.1:8080/v1".to_owned(),
            openai_routes: vec![
                "/v1/models".to_owned(),
                "/v1/chat/completions".to_owned(),
                "/v1/messages".to_owned(),
            ],
            auth_header_guide:
                "Authorization: Bearer <token> or X-Proxy-Secret + X-Forwarded-User via trusted edge proxy"
                    .to_owned(),
            wireguard_endpoint: String::new(),
            tunnel_cidr: String::new(),
            default_interface_state: "unconfigured".to_owned(),
            wireguard_status_path: None,
        }
    }
}

/// Combined endpoint & authorization configuration for `/api/mr/v1/*` routes on `ProtocolServer`.
#[derive(Debug, Clone)]
pub struct ModelRunnerEndpointConfig {
    /// Authorization verifier policy (`fail_closed()` by default).
    pub auth: ModelRunnerAuthPolicy,
    /// Bounded host `RLIMIT_MEMLOCK` ceiling reported by `/api/mr/v1/status`.
    pub memlock_ceiling_bytes: u64,
    /// OpenAI API metadata and rootless WireGuard status file path.
    pub connection: ModelRunnerConnectionConfig,
}

impl Default for ModelRunnerEndpointConfig {
    fn default() -> Self {
        Self {
            auth: ModelRunnerAuthPolicy::fail_closed(),
            memlock_ceiling_bytes: BOUNDED_MEMLOCK_CEILING_BYTES,
            connection: ModelRunnerConnectionConfig::default(),
        }
    }
}

impl ModelRunnerEndpointConfig {
    /// Loads [`ModelRunnerEndpointConfig`] from standard `JERO_MANAGER_*` environment variables,
    /// failing closed on missing/invalid proxy secret or policy bundle files.
    #[must_use]
    pub fn from_env() -> Self {
        let mut cfg = Self::default();

        if let Some(parsed) = std::env::var(ENV_MEMLOCK_CEILING_BYTES)
            .ok()
            .and_then(|raw| raw.trim().parse::<u64>().ok())
            .filter(|&v| v > 0)
        {
            cfg.memlock_ceiling_bytes = parsed;
        }

        if let Ok(base_url) = std::env::var(ENV_OPENAI_BASE_URL) {
            let trimmed = base_url.trim();
            if !trimmed.is_empty() {
                trimmed.clone_into(&mut cfg.connection.openai_base_url);
            }
        }

        if let Ok(routes) = std::env::var(ENV_OPENAI_ROUTES) {
            let parsed_routes: Vec<String> = routes
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(ToOwned::to_owned)
                .collect();
            if !parsed_routes.is_empty() {
                cfg.connection.openai_routes = parsed_routes;
            }
        }

        if let Ok(guide) = std::env::var(ENV_AUTH_HEADER_GUIDE) {
            let trimmed = guide.trim();
            if !trimmed.is_empty() {
                trimmed.clone_into(&mut cfg.connection.auth_header_guide);
            }
        }

        if let Ok(endpoint) = std::env::var(ENV_WIREGUARD_ENDPOINT) {
            endpoint
                .trim()
                .clone_into(&mut cfg.connection.wireguard_endpoint);
        }
        if let Ok(cidr) = std::env::var(ENV_WIREGUARD_TUNNEL_CIDR) {
            cidr.trim().clone_into(&mut cfg.connection.tunnel_cidr);
        }
        if let Ok(state) = std::env::var(ENV_WIREGUARD_INTERFACE_STATE) {
            let trimmed = state.trim();
            if !trimmed.is_empty() {
                trimmed.clone_into(&mut cfg.connection.default_interface_state);
            }
        }
        if let Ok(path) = std::env::var(ENV_WIREGUARD_STATUS_FILE) {
            let trimmed = path.trim();
            if !trimmed.is_empty() {
                cfg.connection.wireguard_status_path = Some(PathBuf::from(trimmed));
            }
        }

        let allow_loopback = std::env::var(ENV_ALLOW_LOOPBACK_NO_SECRET)
            .is_ok_and(|v| matches!(v.trim(), "1" | "true" | "TRUE" | "yes" | "YES"));

        if let Some(policy) =
            EdgeAuthConfig::from_env(ENV_PROXY_SECRET, ENV_PROXY_SECRET_FILE, allow_loopback)
                .ok()
                .zip(load_auth_bundle_from_env())
                .and_then(|(edge_cfg, bundle)| ModelRunnerAuthPolicy::new(edge_cfg, bundle).ok())
        {
            cfg.auth = policy;
        }

        cfg
    }
}

fn load_auth_bundle_from_env() -> Option<ModelRunnerAuthBundle> {
    let raw_path = std::env::var(ENV_AUTH_POLICY_FILE).ok()?;
    let trimmed = raw_path.trim();
    if trimmed.is_empty() {
        return None;
    }
    let resolved_json = resolve_secret_value(trimmed)?;
    let parsed: ModelRunnerAuthBundle = serde_json::from_str(&resolved_json).ok()?;
    parsed.into_validated().ok()
}

/// Handles `GET /api/mr/v1/status` with `mu-auth` verification (`mr:models.read` + `mr:queue.read`).
#[must_use]
pub fn handle_mr_status(
    req: &HttpRequest,
    peer_ip: Option<IpAddr>,
    runtime: &ControlPlaneRuntime,
    cfg: &ModelRunnerEndpointConfig,
) -> HttpResponse {
    if let Err(resp) =
        cfg.auth
            .authorize_request(req, peer_ip, &[POLICY_MR_MODELS_READ, POLICY_MR_QUEUE_READ])
    {
        return resp;
    }
    if req.method != "GET" {
        return method_not_allowed_get();
    }

    let snapshot = build_status_snapshot(runtime, cfg.memlock_ceiling_bytes);
    serialize_json_ok(&snapshot)
}

/// Handles `GET /api/mr/v1/connection` with `mu-auth` verification (`mr:openai_api.read` + `mr:wireguard_vpn.read`).
#[must_use]
pub fn handle_mr_connection(
    req: &HttpRequest,
    peer_ip: Option<IpAddr>,
    cfg: &ModelRunnerEndpointConfig,
) -> HttpResponse {
    if let Err(resp) = cfg.auth.authorize_request(
        req,
        peer_ip,
        &[POLICY_MR_OPENAI_API_READ, POLICY_MR_WIREGUARD_VPN_READ],
    ) {
        return resp;
    }
    if req.method != "GET" {
        return method_not_allowed_get();
    }

    let payload = build_connection_response(&cfg.connection);
    serialize_json_ok(&payload)
}

/// Computes the deterministic [`ModelRunnerStatusResponse`] snapshot from [`ControlPlaneRuntime`].
#[must_use]
pub fn build_status_snapshot(
    runtime: &ControlPlaneRuntime,
    memlock_ceiling_bytes: u64,
) -> ModelRunnerStatusResponse {
    let placements = runtime.model_placement_snapshots();
    let queue_mgr = runtime.queue_mgr();
    let (total_depth, is_executing, estimated_wait_ms) = queue_mgr.aggregate_status();

    let mut models = Vec::with_capacity(placements.len());
    let mut mlock_used_bytes = 0_u64;

    for record in placements {
        let model_queue_depth =
            u32::try_from(queue_mgr.queue_depth(&record.model_id)).unwrap_or(u32::MAX);
        let mlock_bytes = if !record.is_alive {
            0
        } else if let Some(explicit) = runtime.model_mlock_bytes(&record.model_id) {
            explicit
        } else {
            let kv_bytes = estimate_bounded_kv_memlock_kb(record.n_ctx, 0).saturating_mul(1024);
            let weights_bytes = record
                .weights_path
                .as_ref()
                .and_then(|p| std::fs::metadata(p).ok())
                .filter(std::fs::Metadata::is_file)
                .map_or(0, |meta| meta.len());
            kv_bytes
                .saturating_add(weights_bytes)
                .min(memlock_ceiling_bytes)
        };

        mlock_used_bytes = mlock_used_bytes.saturating_add(mlock_bytes);
        models.push(ResidentModelStatus {
            model_id: record.model_id,
            cluster: record.cluster,
            is_alive: record.is_alive,
            n_ctx: u32::try_from(record.n_ctx).unwrap_or(u32::MAX),
            mlock_bytes,
            memlock_ceiling_bytes,
            queue_depth: model_queue_depth,
        });
    }

    ModelRunnerStatusResponse {
        total_queue_depth: u32::try_from(total_depth).unwrap_or(u32::MAX),
        is_executing,
        estimated_wait_ms,
        mlock_used_bytes,
        memlock_ceiling_bytes,
        models,
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct RawWireGuardStatusFile {
    wireguard_endpoint: Option<String>,
    endpoint: Option<String>,
    tunnel_cidr: Option<String>,
    interface_state: Option<String>,
    state: Option<String>,
    peers: Vec<RawWireGuardPeer>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct RawWireGuardPeer {
    public_key_prefix: Option<String>,
    public_key: Option<String>,
    allowed_ips: String,
    latest_handshake_epoch_ms: Option<u64>,
    latest_handshake_secs: Option<u64>,
    rx_bytes: u64,
    tx_bytes: u64,
}

/// Redacts a WireGuard peer public key to at most 8 printable characters followed by `"..."`.
#[must_use]
pub fn redact_public_key_prefix(raw: &str) -> String {
    let trimmed = raw
        .trim()
        .trim_end_matches("...")
        .chars()
        .filter(|c| !c.is_control() && !c.is_whitespace())
        .take(PUBLIC_KEY_PREFIX_CHARS)
        .collect::<String>();
    if trimmed.is_empty() {
        "(none)".to_owned()
    } else {
        format!("{trimmed}...")
    }
}

/// Builds a sanitized [`ModelRunnerConnectionResponse`] from generic OpenAI config and optional
/// rootless WireGuard JSON status file without requiring `CAP_NET_ADMIN`.
#[must_use]
pub fn build_connection_response(
    cfg: &ModelRunnerConnectionConfig,
) -> ModelRunnerConnectionResponse {
    let mut wireguard_endpoint = cfg.wireguard_endpoint.clone();
    let mut tunnel_cidr = cfg.tunnel_cidr.clone();
    let mut interface_state = cfg.default_interface_state.clone();
    let mut peers = Vec::new();

    if let Some(status_path) = &cfg.wireguard_status_path {
        match std::fs::read_to_string(status_path)
            .ok()
            .and_then(|raw| serde_json::from_str::<RawWireGuardStatusFile>(&raw).ok())
        {
            Some(parsed) => {
                if let Some(ep) = parsed
                    .wireguard_endpoint
                    .as_deref()
                    .or(parsed.endpoint.as_deref())
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                {
                    ep.clone_into(&mut wireguard_endpoint);
                }
                if let Some(cidr) = parsed
                    .tunnel_cidr
                    .as_deref()
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                {
                    cidr.clone_into(&mut tunnel_cidr);
                }
                interface_state = parsed
                    .interface_state
                    .as_deref()
                    .or(parsed.state.as_deref())
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map_or_else(|| "up".to_owned(), ToOwned::to_owned);

                peers = parsed
                    .peers
                    .into_iter()
                    .map(|p| {
                        let key_src = p
                            .public_key
                            .as_deref()
                            .map(str::trim)
                            .filter(|s| !s.is_empty())
                            .or_else(|| {
                                p.public_key_prefix
                                    .as_deref()
                                    .map(str::trim)
                                    .filter(|s| !s.is_empty())
                            })
                            .unwrap_or("");
                        let latest_handshake_epoch_ms = match p.latest_handshake_epoch_ms {
                            Some(ms) if ms > 0 => Some(ms),
                            _ => p
                                .latest_handshake_secs
                                .filter(|&s| s > 0)
                                .map(|s| s.saturating_mul(1000)),
                        };
                        WireGuardPeerSummary {
                            public_key_prefix: redact_public_key_prefix(key_src),
                            allowed_ips: p.allowed_ips.trim().to_owned(),
                            latest_handshake_epoch_ms,
                            rx_bytes: p.rx_bytes,
                            tx_bytes: p.tx_bytes,
                        }
                    })
                    .collect();
            }
            None => {
                "unavailable".clone_into(&mut interface_state);
            }
        }
    }

    ModelRunnerConnectionResponse {
        openai_base_url: cfg.openai_base_url.clone(),
        openai_routes: cfg.openai_routes.clone(),
        auth_header_guide: cfg.auth_header_guide.clone(),
        wireguard_endpoint,
        tunnel_cidr,
        interface_state,
        peers,
    }
}

fn serialize_json_ok<T: Serialize>(value: &T) -> HttpResponse {
    match serde_json::to_string_pretty(value) {
        Ok(json) => HttpResponse::json(200, json),
        Err(err) => json_auth_error(
            500,
            "serialization_error",
            &format!("Failed to serialize Model-Runner endpoint response: {err}"),
        ),
    }
}

fn json_auth_error(status: u16, code: &str, message: &str) -> HttpResponse {
    let body = serde_json::json!({
        "error": {
            "code": code,
            "message": message,
        }
    })
    .to_string();
    HttpResponse::error(status, body, Vec::new())
}

fn method_not_allowed_get() -> HttpResponse {
    let body = serde_json::json!({
        "error": {
            "code": "method_not_allowed",
            "message": "Only GET is supported on this endpoint",
        }
    })
    .to_string();
    HttpResponse::error(405, body, vec![("Allow".to_owned(), "GET".to_owned())])
}

#[cfg(test)]
mod tests {
    use super::*;
    use mu_auth::{FolioId, StepUpRequirement};
    use std::net::Ipv4Addr;

    fn test_auth_bundle() -> ModelRunnerAuthBundle {
        let mut role_catalog_roles = vec![
            RoleSpec::pinned_baseline(
                RoleId::new("baseline").expect("baseline id"),
                Some("Baseline"),
                "Minimal baseline role",
            )
            .expect("baseline"),
            RoleSpec::elective(
                RoleId::new("mr-viewer").expect("role id"),
                Some("MR Viewer"),
                "Read-only Model-Runner telemetry",
                StepUpRequirement::None,
            )
            .expect("viewer role"),
            RoleSpec::timed(
                RoleId::new("mr-connect").expect("role id"),
                Some("MR Connect"),
                "Timed Passkey-stepped Model-Runner connection access",
                900,
                300,
            )
            .expect("connect role"),
        ];
        // Make baseline itself grant nothing by default (fail-closed)
        let role_catalog = RoleCatalog::from_roles(role_catalog_roles.drain(..)).expect("catalog");
        let folio_catalog = FolioCatalog::with_canonical_manifests();
        let mut index = RoleFolioIndex::default();
        index
            .set_role_folios(
                RoleId::new("mr-viewer").expect("id"),
                BTreeSet::from([FolioId::new("folio:mr:read").expect("folio")]),
                &folio_catalog,
                &role_catalog,
            )
            .expect("bind mr-viewer");
        index
            .set_role_folios(
                RoleId::new("mr-connect").expect("id"),
                BTreeSet::from([FolioId::new("folio:mr:connect").expect("folio")]),
                &folio_catalog,
                &role_catalog,
            )
            .expect("bind mr-connect");

        let mut bundle =
            ModelRunnerAuthBundle::with_canonical_folios(role_catalog, index).expect("bundle");
        bundle.operators.insert(
            "viewer".to_owned(),
            RegisteredOperator::new(
                1,
                BTreeSet::from([RoleId::new("mr-viewer").expect("id")]),
                RoleLeases {
                    elective_roles: BTreeSet::from([RoleId::new("mr-viewer").expect("id")]),
                    ..RoleLeases::default()
                },
            ),
        );
        bundle
    }

    #[test]
    fn key_redaction_never_exposes_full_wireguard_key() {
        let full = "aBcDeFgHiJkLmNoPqRsTuVwXyZ0123456789abcdef=";
        let redacted = redact_public_key_prefix(full);
        assert_eq!(redacted, "aBcDeFgH...");
        assert!(!redacted.contains("iJkLmNoP"));
        assert_eq!(redact_public_key_prefix("   "), "(none)");
    }

    #[test]
    fn policy_enforces_401_on_fail_closed_and_403_on_insufficient_folio() {
        let fail_closed = ModelRunnerAuthPolicy::fail_closed();
        let req = HttpRequest::new("GET", "/api/mr/v1/status", Vec::new());
        let err_resp = fail_closed
            .authorize_request(&req, None, &[POLICY_MR_MODELS_READ, POLICY_MR_QUEUE_READ])
            .expect_err("unconfigured policy must fail closed");
        assert_eq!(err_resp.status, 401);

        let edge_cfg = EdgeAuthConfig::with_proxy_secret("unit-secret-xyz").expect("edge cfg");
        let policy = ModelRunnerAuthPolicy::new(edge_cfg, test_auth_bundle()).expect("policy");

        let spoofed_req = HttpRequest::new("GET", "/api/mr/v1/status", Vec::new())
            .with_header("X-Forwarded-User", "viewer");
        assert_eq!(
            policy
                .authorize_request(
                    &spoofed_req,
                    Some(IpAddr::V4(Ipv4Addr::LOCALHOST)),
                    &[POLICY_MR_MODELS_READ, POLICY_MR_QUEUE_READ],
                )
                .expect_err("missing proxy secret must fail 401")
                .status,
            401
        );

        let ok_status_req = HttpRequest::new("GET", "/api/mr/v1/status", Vec::new())
            .with_header("X-Proxy-Secret", "unit-secret-xyz")
            .with_header("X-Forwarded-User", "viewer");
        assert!(
            policy
                .authorize_request(
                    &ok_status_req,
                    None,
                    &[POLICY_MR_MODELS_READ, POLICY_MR_QUEUE_READ],
                )
                .is_ok()
        );

        let forbidden_connect = HttpRequest::new("GET", "/api/mr/v1/connection", Vec::new())
            .with_header("X-Proxy-Secret", "unit-secret-xyz")
            .with_header("X-Forwarded-User", "viewer");
        assert_eq!(
            policy
                .authorize_request(
                    &forbidden_connect,
                    None,
                    &[POLICY_MR_OPENAI_API_READ, POLICY_MR_WIREGUARD_VPN_READ],
                )
                .expect_err("viewer lacks folio:mr:connect")
                .status,
            403
        );
    }
}

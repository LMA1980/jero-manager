//! HTTP integration tests for role-gated `/api/mr/v1/status` and `/api/mr/v1/connection` (`#74`–`#78`).

#![allow(clippy::pedantic)]

use std::collections::BTreeSet;
use std::fs;
use std::net::{IpAddr, Ipv4Addr};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use jero_core::affinity::ControlPlaneCapacityConfig;
use jero_manager::domain::{MockOsAffinityBackend, OsAffinityBackend};
use jero_manager::flight::BOUNDED_MEMLOCK_CEILING_BYTES;
use jero_manager::protocol::{
    AuthClock, HttpRequest, HttpResponse, HttpResponseBody, ModelRunnerAuthBundle,
    ModelRunnerAuthPolicy, ModelRunnerConnectionConfig, ModelRunnerConnectionResponse,
    ModelRunnerEndpointConfig, ModelRunnerStatusResponse, ProtocolServer, ProtocolServerConfig,
    RegisteredOperator,
};
use jero_manager::queue::{AdmissionQueueConfig, AdmissionQueueManager};
use jero_manager::runtime::ControlPlaneRuntime;
use mu_auth::{
    EdgeAuthConfig, FolioCatalog, FolioId, Principal, RoleCatalog, RoleFolioIndex, RoleId,
    RoleLeases, RoleSpec, SubsystemId,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

static SEQ: AtomicU64 = AtomicU64::new(1);

struct TempWorkspace {
    path: PathBuf,
}

impl TempWorkspace {
    fn new(tag: &str) -> Self {
        let id = SEQ.fetch_add(1, Ordering::SeqCst);
        let dir =
            std::env::temp_dir().join(format!("mr_ep_test_{tag}_{}_{id}", std::process::id()));
        fs::create_dir_all(&dir).expect("create temp test dir");
        Self { path: dir }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempWorkspace {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn build_test_runtime() -> Arc<ControlPlaneRuntime> {
    let capacity = ControlPlaneCapacityConfig::default()
        .resolve_default()
        .expect("capacity resolves");
    let backend: Arc<dyn OsAffinityBackend> = Arc::new(MockOsAffinityBackend::default());
    let mut rt =
        ControlPlaneRuntime::with_affinity_backend(capacity, backend).expect("runtime initializes");
    let q_mgr = Arc::new(AdmissionQueueManager::new(AdmissionQueueConfig {
        max_depth: 8,
        estimated_pass_latency_ms: 25.0,
        estimated_passes_per_request: 2,
        ..AdmissionQueueConfig::default()
    }));
    rt.set_queue_mgr(q_mgr);
    Arc::new(rt)
}

fn extract_json_bytes(resp: HttpResponse) -> (u16, String) {
    let status = resp.status;
    let body = match resp.body {
        HttpResponseBody::Bytes(bytes) => String::from_utf8(bytes).expect("utf8 json"),
        HttpResponseBody::Stream(_) => panic!("unexpected streaming response"),
    };
    (status, body)
}

fn make_auth_fixture(
    edge_cfg: EdgeAuthConfig,
    clock_ms: Arc<AtomicU64>,
) -> (ModelRunnerAuthPolicy, RoleCatalog) {
    let roles = vec![
        RoleSpec::pinned_baseline(
            RoleId::new("baseline").expect("baseline id"),
            Some("Baseline Reader"),
            "Baseline read-only operator",
        )
        .expect("baseline"),
        RoleSpec::timed(
            RoleId::new("mr-connect-timed").expect("id"),
            Some("Model Runner Connect"),
            "Temporary WireGuard & OpenAI connection visibility",
            600,
            120,
        )
        .expect("timed role"),
    ];
    let role_catalog = RoleCatalog::from_roles(roles).expect("role catalog");
    let folio_catalog = FolioCatalog::with_canonical_manifests();

    let mut index = RoleFolioIndex::default();
    index
        .set_role_folios(
            RoleId::new("baseline").expect("id"),
            BTreeSet::from([FolioId::new("folio:mr:read").expect("folio")]),
            &folio_catalog,
            &role_catalog,
        )
        .expect("baseline folio binding");
    index
        .set_role_folios(
            RoleId::new("mr-connect-timed").expect("id"),
            BTreeSet::from([FolioId::new("folio:mr:connect").expect("folio")]),
            &folio_catalog,
            &role_catalog,
        )
        .expect("timed folio binding");

    let mut bundle =
        ModelRunnerAuthBundle::with_canonical_folios(role_catalog.clone(), index).expect("bundle");

    // Viewer with Passkey enrolled: holds canonical baseline (folio:mr:read only)
    bundle.operators.insert(
        "alice-viewer".to_owned(),
        RegisteredOperator::new(1, BTreeSet::new(), RoleLeases::default()),
    );

    // Operator with 0 Passkeys enrolled: mandatory Passkey gate denies even baseline folios
    bundle.operators.insert(
        "zero-passkey".to_owned(),
        RegisteredOperator::new(0, BTreeSet::new(), RoleLeases::default()),
    );

    // Operator holding active timed `mr-connect-timed` lease anchored at `now_ms`
    let now_ms = clock_ms.load(Ordering::SeqCst);
    let timed_role_id = RoleId::new("mr-connect-timed").expect("id");
    let bound_roles = BTreeSet::from([timed_role_id.clone()]);
    let bob_principal = Principal::Authenticated {
        login: "bob-connector".to_owned(),
        passkey_count: 1,
        bound_roles: bound_roles.clone(),
    };
    let mut connect_leases = RoleLeases::default();
    connect_leases
        .activate_timed(&role_catalog, &bob_principal, &timed_role_id, now_ms, true)
        .expect("activate timed lease");
    bundle.operators.insert(
        "bob-connector".to_owned(),
        RegisteredOperator::new(1, bound_roles, connect_leases),
    );

    let policy = ModelRunnerAuthPolicy::new(edge_cfg, bundle)
        .expect("valid auth policy")
        .with_clock(AuthClock::Fixed(clock_ms));
    (policy, role_catalog)
}

#[tokio::test]
async fn mr_status_reports_empty_queue_queued_prompts_and_mlock_accounting() {
    let runtime = build_test_runtime();
    runtime.record_model_placement("qwen2.5-7b".to_owned(), "x100".to_owned(), Some(1001));
    runtime.record_model_placement("deepseek-r1".to_owned(), "a100".to_owned(), Some(1002));
    runtime.record_model_mlock_bytes("deepseek-r1", 512 * 1024 * 1024);

    // Mark third model dead -> mlock_bytes must be 0 even if recorded offline
    runtime.record_model_placement("phi3-mini".to_owned(), "x100".to_owned(), None);
    runtime.update_worker_health("phi3-mini", false, 3);

    let clock = Arc::new(AtomicU64::new(1_700_000_000_000));
    let edge = EdgeAuthConfig::with_proxy_secret("proxy-secret-test").expect("edge config");
    let (auth, _) = make_auth_fixture(edge, clock);
    let mr_cfg = ModelRunnerEndpointConfig {
        auth,
        ..ModelRunnerEndpointConfig::default()
    };

    let server = ProtocolServer::with_model_runner(
        ProtocolServerConfig::default(),
        Arc::clone(&runtime),
        mr_cfg,
    );

    // 1. Empty queue state
    let req_empty = HttpRequest::new("GET", "/api/mr/v1/status", Vec::new())
        .with_header("X-Proxy-Secret", "proxy-secret-test")
        .with_header("X-Forwarded-User", "alice-viewer");
    let (status, json_body) = extract_json_bytes(server.route_request(req_empty).await);
    assert_eq!(status, 200);
    let parsed: ModelRunnerStatusResponse =
        serde_json::from_str(&json_body).expect("deserialize status dto");
    assert_eq!(parsed.total_queue_depth, 0);
    assert!(!parsed.is_executing);
    assert_eq!(parsed.estimated_wait_ms, 0);
    assert_eq!(parsed.memlock_ceiling_bytes, BOUNDED_MEMLOCK_CEILING_BYTES);
    assert_eq!(parsed.models.len(), 3);

    let ds = parsed
        .models
        .iter()
        .find(|m| m.model_id == "deepseek-r1")
        .expect("deepseek model present");
    assert!(ds.is_alive);
    assert_eq!(ds.mlock_bytes, 512 * 1024 * 1024);

    let qwen = parsed
        .models
        .iter()
        .find(|m| m.model_id == "qwen2.5-7b")
        .expect("qwen model present");
    assert!(qwen.is_alive);
    assert_eq!(qwen.mlock_bytes, u64::from(qwen.n_ctx) * 1024);

    let phi = parsed
        .models
        .iter()
        .find(|m| m.model_id == "phi3-mini")
        .expect("phi3 model present");
    assert!(!phi.is_alive);
    assert_eq!(phi.mlock_bytes, 0);
    assert_eq!(parsed.mlock_used_bytes, ds.mlock_bytes + qwen.mlock_bytes);

    // 2. Pop 1 request into active execution + leave 2 queued prompts on qwen2.5-7b
    runtime
        .queue_mgr()
        .admit_to_model(
            "qwen2.5-7b",
            "req-exec".to_owned(),
            false,
            None,
            None,
            Arc::new(AtomicBool::new(false)),
            None,
            None,
        )
        .expect("admit exec");
    let _permit = runtime
        .queue_mgr()
        .pop_next_for_model("qwen2.5-7b")
        .expect("dispatched");
    for idx in 1..=2 {
        runtime
            .queue_mgr()
            .admit_to_model(
                "qwen2.5-7b",
                format!("req-wait-{idx}"),
                false,
                None,
                None,
                Arc::new(AtomicBool::new(false)),
                None,
                None,
            )
            .expect("admit queued");
    }

    let req_busy = HttpRequest::new("GET", "/api/mr/v1/status?detail=full", Vec::new())
        .with_header("X-Proxy-Secret", "proxy-secret-test")
        .with_header("X-Forwarded-User", "Alice-Viewer");
    let (status_busy, body_busy) = extract_json_bytes(server.route_request(req_busy).await);
    assert_eq!(status_busy, 200);
    let busy: ModelRunnerStatusResponse = serde_json::from_str(&body_busy).expect("parse busy");
    assert_eq!(busy.total_queue_depth, 2);
    assert!(busy.is_executing);
    assert!(busy.estimated_wait_ms > 0);
    let qwen_busy = busy
        .models
        .iter()
        .find(|m| m.model_id == "qwen2.5-7b")
        .unwrap();
    assert_eq!(qwen_busy.queue_depth, 2);
}

#[tokio::test]
async fn mr_connection_parses_rootless_wireguard_json_redacts_keys_and_handles_missing_file() {
    let dir = TempWorkspace::new("wg");
    let wg_path = dir.path().join("wg-status.json");
    let full_peer_pub = "xYzAbCdEfGhIjKlMnOpQrStUvWxYz0123456789abcde=";
    let status_json = serde_json::json!({
        "wireguard_endpoint": "vpn.example.net:51820",
        "tunnel_cidr": "192.0.2.10/24",
        "interface_state": "up",
        "private_key": "DO_NOT_EXFILTRATE_WG_PRIV_KEY",
        "peers": [{
            "public_key": full_peer_pub,
            "preshared_key": "DO_NOT_EXFILTRATE_PSK",
            "allowed_ips": "192.0.2.11/32",
            "latest_handshake_secs": 1700000123_u64,
            "rx_bytes": 1048576_u64,
            "tx_bytes": 2097152_u64
        }]
    });
    fs::write(&wg_path, status_json.to_string()).expect("write wg status json");

    let clock = Arc::new(AtomicU64::new(1_700_000_000_000));
    let edge = EdgeAuthConfig::with_proxy_secret("proxy-secret-test").expect("edge config");
    let (auth, _) = make_auth_fixture(edge, Arc::clone(&clock));

    let mr_cfg = ModelRunnerEndpointConfig {
        auth,
        connection: ModelRunnerConnectionConfig {
            openai_base_url: "https://api.example.org/v1".to_owned(),
            openai_routes: vec!["/v1/models".to_owned(), "/v1/chat/completions".to_owned()],
            auth_header_guide: "Authorization: Bearer <token>".to_owned(),
            wireguard_endpoint: "fallback.example.net:51820".to_owned(),
            tunnel_cidr: "192.0.2.1/24".to_owned(),
            default_interface_state: "down".to_owned(),
            wireguard_status_path: Some(wg_path.clone()),
        },
        ..ModelRunnerEndpointConfig::default()
    };

    let server = ProtocolServer::with_model_runner(
        ProtocolServerConfig::default(),
        build_test_runtime(),
        mr_cfg,
    );

    let req = HttpRequest::new("GET", "/api/mr/v1/connection", Vec::new())
        .with_header("X-Proxy-Secret", "proxy-secret-test")
        .with_header("X-Forwarded-User", "bob-connector");
    let (status, raw_body) = extract_json_bytes(server.route_request(req.clone()).await);
    assert_eq!(status, 200);
    assert!(!raw_body.contains(full_peer_pub));
    assert!(!raw_body.contains("DO_NOT_EXFILTRATE_WG_PRIV_KEY"));
    assert!(!raw_body.contains("DO_NOT_EXFILTRATE_PSK"));

    let parsed: ModelRunnerConnectionResponse =
        serde_json::from_str(&raw_body).expect("connection dto");
    assert_eq!(parsed.openai_base_url, "https://api.example.org/v1");
    assert_eq!(parsed.wireguard_endpoint, "vpn.example.net:51820");
    assert_eq!(parsed.tunnel_cidr, "192.0.2.10/24");
    assert_eq!(parsed.interface_state, "up");
    assert_eq!(parsed.peers.len(), 1);
    assert_eq!(parsed.peers[0].public_key_prefix, "xYzAbCdE...");
    assert_eq!(parsed.peers[0].allowed_ips, "192.0.2.11/32");
    assert_eq!(
        parsed.peers[0].latest_handshake_epoch_ms,
        Some(1_700_000_123_000)
    );
    assert_eq!(parsed.peers[0].rx_bytes, 1_048_576);
    assert_eq!(parsed.peers[0].tx_bytes, 2_097_152);

    // Remove status file -> reports unavailable safely without leaking filesystem paths
    fs::remove_file(&wg_path).expect("remove wg json");
    let (status_missing, body_missing) = extract_json_bytes(server.route_request(req).await);
    assert_eq!(status_missing, 200);
    assert!(!body_missing.contains(&wg_path.to_string_lossy().to_string()));
    let missing_dto: ModelRunnerConnectionResponse =
        serde_json::from_str(&body_missing).expect("missing connection dto");
    assert_eq!(missing_dto.interface_state, "unavailable");
    assert!(missing_dto.peers.is_empty());
    assert_eq!(missing_dto.wireguard_endpoint, "fallback.example.net:51820");
}

#[tokio::test]
async fn mr_endpoints_fail_closed_with_401_and_403_across_auth_and_lease_boundaries() {
    let runtime = build_test_runtime();

    // 1. ProtocolServer::new fails closed on /api/mr/v1/* (401), keeping OpenAI /v1/models untouched
    let default_srv = ProtocolServer::new(ProtocolServerConfig::default(), Arc::clone(&runtime));
    assert_eq!(
        default_srv
            .route_request(HttpRequest::new("GET", "/v1/models", Vec::new()))
            .await
            .status,
        200
    );
    assert_eq!(
        default_srv
            .route_request(HttpRequest::new("GET", "/api/mr/v1/status", Vec::new()))
            .await
            .status,
        401
    );
    assert_eq!(
        default_srv
            .route_request(HttpRequest::new("GET", "/api/mr/v1/connection", Vec::new()))
            .await
            .status,
        401
    );

    // 2. Configured auth policy checks
    let clock = Arc::new(AtomicU64::new(1_700_000_000_000));
    let edge = EdgeAuthConfig::with_secret_and_loopback(None, true).expect("loopback trust");
    let (auth, _) = make_auth_fixture(edge, Arc::clone(&clock));
    let bundle_handle = auth.bundle_handle();
    let srv = ProtocolServer::with_model_runner(
        ProtocolServerConfig::default(),
        Arc::clone(&runtime),
        ModelRunnerEndpointConfig {
            auth,
            ..ModelRunnerEndpointConfig::default()
        },
    );

    let loopback_ip = Some(IpAddr::V4(Ipv4Addr::LOCALHOST));
    let remote_ip = Some(IpAddr::V4(Ipv4Addr::new(192, 0, 2, 42)));

    // No X-Forwarded-User -> 401
    let anon_req = HttpRequest::new("GET", "/api/mr/v1/status", Vec::new());
    assert_eq!(
        srv.route_request_with_peer(anon_req, loopback_ip)
            .await
            .status,
        401
    );

    // Non-loopback source spoofing X-Forwarded-User without secret -> 401
    let remote_spoof = HttpRequest::new("GET", "/api/mr/v1/status", Vec::new())
        .with_header("X-Forwarded-User", "bob-connector");
    assert_eq!(
        srv.route_request_with_peer(remote_spoof, remote_ip)
            .await
            .status,
        401
    );

    // Unbound authenticated GitHub login -> 403
    let unbound = HttpRequest::new("GET", "/api/mr/v1/status", Vec::new())
        .with_header("X-Forwarded-User", "stranger");
    assert_eq!(
        srv.route_request_with_peer(unbound, loopback_ip)
            .await
            .status,
        403
    );

    // Zero enrolled Passkeys -> 403
    let zero_pk = HttpRequest::new("GET", "/api/mr/v1/status", Vec::new())
        .with_header("X-Forwarded-User", "zero-passkey");
    assert_eq!(
        srv.route_request_with_peer(zero_pk, loopback_ip)
            .await
            .status,
        403
    );

    // alice-viewer holds folio:mr:read ONLY -> 200 on /status, 403 on /connection
    let alice_status = HttpRequest::new("GET", "/api/mr/v1/status", Vec::new())
        .with_header("X-Forwarded-User", "alice-viewer");
    assert_eq!(
        srv.route_request_with_peer(alice_status, loopback_ip)
            .await
            .status,
        200
    );
    let alice_conn = HttpRequest::new("GET", "/api/mr/v1/connection", Vec::new())
        .with_header("X-Forwarded-User", "alice-viewer");
    assert_eq!(
        srv.route_request_with_peer(alice_conn, loopback_ip)
            .await
            .status,
        403
    );

    // bob-connector holds active Timed lease -> 200 before expiry, 403 after idle expiry
    let bob_conn = HttpRequest::new("GET", "/api/mr/v1/connection", Vec::new())
        .with_header("X-Forwarded-User", "bob-connector");
    assert_eq!(
        srv.route_request_with_peer(bob_conn.clone(), loopback_ip)
            .await
            .status,
        200
    );
    clock.store(1_700_000_000_000 + 121_000, Ordering::SeqCst);
    assert_eq!(
        srv.route_request_with_peer(bob_conn, loopback_ip)
            .await
            .status,
        403
    );

    // Disabling ModelRunner subsystem suppresses alice-viewer's baseline /status grant -> 403
    {
        let mut guard = bundle_handle.write().expect("lock bundle");
        guard.authority.disable_subsystem(SubsystemId::ModelRunner);
    }
    let alice_disabled = HttpRequest::new("GET", "/api/mr/v1/status", Vec::new())
        .with_header("X-Forwarded-User", "alice-viewer");
    assert_eq!(
        srv.route_request_with_peer(alice_disabled, loopback_ip)
            .await
            .status,
        403
    );
}

#[tokio::test]
async fn tcp_wire_http_serves_401_403_and_200_over_real_socket() {
    let runtime = build_test_runtime();
    runtime.record_model_placement("qwen2.5-7b".to_owned(), "x100".to_owned(), Some(2001));

    let clock = Arc::new(AtomicU64::new(1_700_000_000_000));
    let edge = EdgeAuthConfig::with_proxy_secret("wire-secret").expect("edge");
    let (auth, _) = make_auth_fixture(edge, clock);
    let server = ProtocolServer::with_model_runner(
        ProtocolServerConfig::default(),
        runtime,
        ModelRunnerEndpointConfig {
            auth,
            ..ModelRunnerEndpointConfig::default()
        },
    );

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local addr");

    let srv_task = tokio::spawn(async move {
        for _ in 0..3 {
            let (sock, peer) = listener.accept().await.expect("accept");
            let s = server.clone();
            tokio::spawn(async move {
                let _ = s.handle_connection_with_peer(sock, Some(peer.ip())).await;
            });
        }
    });

    async fn send_raw(addr: std::net::SocketAddr, raw_req: &str) -> String {
        let mut stream = TcpStream::connect(addr).await.expect("connect");
        stream.write_all(raw_req.as_bytes()).await.expect("write");
        let mut buf = Vec::new();
        stream.read_to_end(&mut buf).await.expect("read");
        String::from_utf8(buf).expect("utf8")
    }

    let r401 = send_raw(
        addr,
        "GET /api/mr/v1/status HTTP/1.1\r\nHost: localhost\r\nX-Forwarded-User: alice-viewer\r\n\r\n",
    )
    .await;
    assert!(r401.starts_with("HTTP/1.1 401 Unauthorized\r\n"), "{r401}");

    let r403 = send_raw(
        addr,
        "GET /api/mr/v1/connection HTTP/1.1\r\nHost: localhost\r\nX-Proxy-Secret: wire-secret\r\nX-Forwarded-User: alice-viewer\r\n\r\n",
    )
    .await;
    assert!(r403.starts_with("HTTP/1.1 403 Forbidden\r\n"), "{r403}");

    let r200 = send_raw(
        addr,
        "GET /api/mr/v1/connection HTTP/1.1\r\nHost: localhost\r\nX-Proxy-Secret: wire-secret\r\nX-Forwarded-User: bob-connector\r\n\r\n",
    )
    .await;
    assert!(r200.starts_with("HTTP/1.1 200 OK\r\n"), "{r200}");
    assert!(r200.contains("\"openai_base_url\""), "{r200}");

    srv_task.await.expect("server tasks complete");
}

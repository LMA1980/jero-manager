//! `KvClient` unit and integration tests for memory arena management.

#![allow(
    clippy::pedantic,
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::doc_markdown
)]

use std::io::Read;
use std::os::unix::net::UnixStream;
use std::time::Duration;

use jero_core::ipc::layout::{ARENA_MAGIC, ARENA_VERSION};
use jero_core::ipc::protocol::{AllocateArenaRequest, KV_DTYPE_F32, StatusCode};
use jero_kv_cache::server::{KvServer, ServerConfig};
use jero_manager::client::KvClient;
use jero_manager::error::ManagerError;

#[test]
fn test_kv_client_lifecycle_and_arena_allocation() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");

    let pid = std::process::id();
    let socket_path = std::env::temp_dir().join(format!("kv_client_test_{pid}.sock"));

    let server_config = ServerConfig {
        socket_path: socket_path.clone(),
        max_memory_bytes: 1024 * 1024 * 1024,
    };
    let server = KvServer::new(server_config);

    let task = rt.spawn(async move {
        let _ = server.run().await;
    });

    // Wait for server to bind
    let mut client = None;
    for _ in 0..50 {
        if let Ok(c) = KvClient::connect(&socket_path) {
            client = Some(c);
            break;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    let mut client = client.expect("connect KvClient");

    // 1. Heartbeat
    let echo = client.heartbeat().expect("heartbeat");
    assert!(echo > 0);

    // 2. Allocate arena
    let alloc = client
        .allocate_arena("test_model", 65536, 64, 4)
        .expect("allocate arena");
    assert_eq!(alloc.status, StatusCode::Success);
    assert!(alloc.arena_bytes >= 65536);
    let arena_id = alloc.arena_id;

    // 3. Attach arena (fetch FD via SCM_RIGHTS and map)
    let arena = client
        .attach_arena(arena_id, alloc.arena_bytes as usize, "test_model_shm")
        .expect("attach arena");
    let hdr = *arena.header().expect("read header");
    assert_eq!(hdr.magic, ARENA_MAGIC);
    assert_eq!(hdr.version, ARENA_VERSION);
    assert_eq!(hdr.n_ctx, 64);

    // 4. Deallocate arena
    let dealloc = client.deallocate_arena(arena_id).expect("deallocate arena");
    assert_eq!(dealloc.status, StatusCode::Success);
    assert_eq!(dealloc.arena_id, arena_id);

    // 5. Querying deallocated arena returns error
    let query_err = client.get_arena_fd(arena_id);
    assert!(query_err.is_err());
    match query_err.unwrap_err() {
        ManagerError::Status { status, .. } => {
            assert_eq!(status, StatusCode::ArenaNotFound);
        }
        other => panic!("expected StatusCode::ArenaNotFound, got {other:?}"),
    }

    // Teardown
    task.abort();
    let _ = std::fs::remove_file(&socket_path);
}

#[test]
fn test_kv_client_with_request_hydrates_custom_geometry_header() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");

    let pid = std::process::id();
    let socket_path = std::env::temp_dir().join(format!("kv_client_custom_geom_{pid}.sock"));
    let _ = std::fs::remove_file(&socket_path);

    let server = KvServer::new(ServerConfig {
        socket_path: socket_path.clone(),
        max_memory_bytes: 1024 * 1024 * 1024,
    });
    let task = rt.spawn(async move {
        let _ = server.run().await;
    });

    let mut client = None;
    for _ in 0..50 {
        if let Ok(c) = KvClient::connect(&socket_path) {
            client = Some(c);
            break;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    let mut client = client.expect("connect KvClient");

    let req = AllocateArenaRequest {
        model_id: "custom_geom_model".to_owned(),
        arena_bytes: 2 * 1024 * 1024,
        n_ctx: 128,
        max_blocks: 8,
        num_layers: 12,
        head_count_kv: 4,
        head_dim: 128,
        vocab_size: 49_152,
        kv_dtype: KV_DTYPE_F32,
    };

    let (alloc, arena) = client
        .allocate_and_attach_with_request(req)
        .expect("allocate_and_attach_with_request custom geometry");
    assert_eq!(alloc.status, StatusCode::Success);

    let hdr = *arena.header().expect("read arena header");
    assert_eq!(hdr.magic, ARENA_MAGIC);
    assert_eq!(hdr.version, ARENA_VERSION);
    assert_eq!(hdr.n_ctx, 128);
    assert_eq!(hdr.num_layers, 12);
    assert_eq!(hdr.head_count_kv, 4);
    assert_eq!(hdr.head_dim, 128);
    assert_eq!(hdr.vocab_size, 49_152);

    let dealloc = client
        .deallocate_arena(alloc.arena_id)
        .expect("deallocate custom geometry arena");
    assert_eq!(dealloc.status, StatusCode::Success);

    task.abort();
    let _ = std::fs::remove_file(&socket_path);
}

#[test]
fn test_kv_client_with_request_rejects_zero_geometry_upfront() {
    let (client_stream, mut peer_stream) = UnixStream::pair().expect("unix stream pair");
    peer_stream
        .set_nonblocking(true)
        .expect("set peer nonblocking");
    let mut client = KvClient::from_stream(client_stream);

    let bad_layers_req = AllocateArenaRequest {
        model_id: "invalid_layers_model".to_owned(),
        arena_bytes: 65_536,
        n_ctx: 64,
        max_blocks: 4,
        num_layers: 0,
        head_count_kv: 2,
        head_dim: 64,
        vocab_size: 32_000,
        kv_dtype: KV_DTYPE_F32,
    };
    let err_layers = client
        .allocate_arena_with_request(bad_layers_req)
        .expect_err("num_layers == 0 must fail upfront");
    match err_layers {
        ManagerError::Kv(msg) => assert!(
            msg.contains("num_layers"),
            "unexpected validation message: {msg}"
        ),
        other => panic!("expected ManagerError::Kv for num_layers == 0, got {other:?}"),
    }

    let bad_vocab_req = AllocateArenaRequest {
        model_id: "invalid_vocab_model".to_owned(),
        arena_bytes: 65_536,
        n_ctx: 64,
        max_blocks: 4,
        num_layers: 2,
        head_count_kv: 2,
        head_dim: 64,
        vocab_size: 0,
        kv_dtype: KV_DTYPE_F32,
    };
    let err_vocab = client
        .allocate_and_attach_with_request(bad_vocab_req)
        .expect_err("vocab_size == 0 must fail upfront");
    match err_vocab {
        ManagerError::Kv(msg) => assert!(
            msg.contains("vocab_size"),
            "unexpected validation message: {msg}"
        ),
        other => panic!("expected ManagerError::Kv for vocab_size == 0, got {other:?}"),
    }

    // Confirm no bytes were serialized or sent onto the UDS stream.
    let mut buf = [0_u8; 16];
    match peer_stream.read(&mut buf) {
        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {}
        other => {
            panic!("expected zero UDS bytes transmitted before validation failure, got {other:?}")
        }
    }
}

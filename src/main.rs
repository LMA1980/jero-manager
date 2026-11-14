// SPDX-License-Identifier: AGPL-3.0-only
//! `jero-manager` daemon binary.
//!
//! Control plane and IPC daemon for the `SpacemiT` K3 AI platform, hosting
//! OpenAI-compatible and Claude-compatible HTTP endpoints (/v1/chat/completions, /v1/models, /v1/messages).

use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpListener;

use jero_core::affinity::ControlPlaneCapacityConfig;
use jero_manager::protocol::{ModelRunnerEndpointConfig, ProtocolServer, ProtocolServerConfig};
use jero_manager::runtime::{AdminCommand, ControlPlaneRuntime};

fn resolve_bind_addr() -> SocketAddr {
    // 1. Check JERO_MANAGER_BIND (e.g. "127.0.0.1:8080" or "8080")
    if let Ok(bind) = std::env::var("JERO_MANAGER_BIND") {
        if let Ok(addr) = bind.parse::<SocketAddr>() {
            return addr;
        }
        if let Ok(port) = bind.parse::<u16>() {
            return SocketAddr::from(([127, 0, 0, 1], port));
        }
    }
    // 2. Check PORT (e.g. "8080")
    if let Ok(port_str) = std::env::var("PORT") {
        if let Ok(port) = port_str.parse::<u16>() {
            return SocketAddr::from(([127, 0, 0, 1], port));
        }
    }
    // 3. Default to 127.0.0.1:8080
    SocketAddr::from(([127, 0, 0, 1], 8080))
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let bind_addr = resolve_bind_addr();

    // Initialize control plane capacity and runtime
    let capacity = ControlPlaneCapacityConfig::default().resolve_default()?;
    let runtime = Arc::new(ControlPlaneRuntime::new(capacity)?);

    // Register canonical default model ("jero") for OpenAI adapter resolution
    let _ = runtime.execute_admin(AdminCommand::RegisterModel {
        model_id: "jero".to_owned(),
        weights_path: std::path::PathBuf::from("/models/jero.gguf"),
        cluster: "x100".to_owned(),
    });

    if let Ok(extra_model) = std::env::var("JERO_MODEL_NAME") {
        let trimmed = extra_model.trim();
        if !trimmed.is_empty() && trimmed != "jero" {
            let _ = runtime.execute_admin(AdminCommand::RegisterModel {
                model_id: trimmed.to_owned(),
                weights_path: std::path::PathBuf::from("/models/custom.gguf"),
                cluster: "x100".to_owned(),
            });
        }
    }

    let config = ProtocolServerConfig {
        listen_addr: Some(bind_addr),
        ..Default::default()
    };

    let server =
        ProtocolServer::with_model_runner(config, runtime, ModelRunnerEndpointConfig::from_env());
    let listener = TcpListener::bind(bind_addr).await?;
    eprintln!("jero-manager daemon listening on http://{bind_addr}");

    loop {
        tokio::select! {
            accept_res = listener.accept() => {
                let (stream, peer_addr) = accept_res?;
                let srv = server.clone();
                tokio::spawn(async move {
                    let _ = srv
                        .handle_connection_with_peer(stream, Some(peer_addr.ip()))
                        .await;
                });
            }
            _ = tokio::signal::ctrl_c() => {
                eprintln!("jero-manager received shutdown signal, terminating.");
                break;
            }
        }
    }

    Ok(())
}

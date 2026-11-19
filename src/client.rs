//! Unix Domain Socket client for `jero-kv-cache` context memory holder daemon.

use std::os::fd::OwnedFd;
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use jero_core::ipc::protocol::{
    AllocateArenaRequest, AllocateArenaResponse, DeallocateArenaRequest, DeallocateArenaResponse,
    GetArenaFdRequest, GetArenaFdResponse, IpcMessage, KV_DTYPE_F32, PingRequest, PingResponse,
    recv_msg_frame, send_msg_frame,
};
use jero_core::ipc::shm::SharedMemoryArena;

use crate::error::ManagerError;

/// Canonical Unix domain socket path for `jero-kv-cache`.
pub const DEFAULT_KV_SOCKET_PATH: &str = "/run/jero-runner/kv.sock";

/// Legacy Phase 1 Unix domain socket path fallback when `/run/jero-runner/kv.sock` does not exist yet.
pub const LEGACY_KV_SOCKET_PATH: &str = "/tmp/jero-kv-cache.sock";

/// Environment variable overriding the `jero-kv-cache` Unix domain socket path.
pub const ENV_KV_SOCKET: &str = "JERO_KV_SOCKET";

fn resolve_kv_socket_path_with(
    get_env: impl Fn(&str) -> Option<String>,
    exists: impl Fn(&Path) -> bool,
) -> PathBuf {
    if let Some(val) = get_env(ENV_KV_SOCKET) {
        let trimmed = val.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }
    let canonical = Path::new(DEFAULT_KV_SOCKET_PATH);
    if exists(canonical) {
        return canonical.to_path_buf();
    }
    let legacy = Path::new(LEGACY_KV_SOCKET_PATH);
    if exists(legacy) {
        return legacy.to_path_buf();
    }
    canonical.to_path_buf()
}

/// Resolves the `jero-kv-cache` Unix domain socket path.
///
/// Precedence: `JERO_KV_SOCKET` -> `/run/jero-runner/kv.sock`
/// (falling back to existing `/tmp/jero-kv-cache.sock` if `/run/jero-runner/kv.sock`
/// does not exist yet).
#[must_use]
pub fn resolve_kv_socket_path() -> PathBuf {
    resolve_kv_socket_path_with(|k| std::env::var(k).ok(), Path::exists)
}

/// Builds a 9-field [`AllocateArenaRequest`] populated with baseline F32 model geometry defaults.
fn default_allocate_arena_request(
    model_id: &str,
    arena_bytes: u64,
    n_ctx: u32,
    max_blocks: u32,
) -> AllocateArenaRequest {
    AllocateArenaRequest {
        model_id: model_id.to_owned(),
        arena_bytes,
        n_ctx,
        max_blocks,
        num_layers: 2,
        head_count_kv: 2,
        head_dim: 64,
        vocab_size: 32_000,
        kv_dtype: KV_DTYPE_F32,
    }
}

/// Client connection to the `jero-kv-cache` memory holder daemon over Unix Domain Sockets.
#[derive(Debug)]
pub struct KvClient {
    stream: UnixStream,
    socket_path: Option<PathBuf>,
}

impl KvClient {
    /// Connects to a running `jero-kv-cache` daemon at `socket_path`.
    ///
    /// # Errors
    /// Returns [`ManagerError::Kv`] if connecting to the socket fails.
    pub fn connect(socket_path: impl AsRef<Path>) -> Result<Self, ManagerError> {
        let path = socket_path.as_ref();
        let stream = UnixStream::connect(path).map_err(|e| {
            ManagerError::Kv(format!(
                "failed to connect to jero-kv-cache socket '{}': {e}",
                path.display()
            ))
        })?;
        Ok(Self {
            stream,
            socket_path: Some(path.to_path_buf()),
        })
    }

    /// Connects to the default `jero-kv-cache` daemon resolved via [`resolve_kv_socket_path`].
    ///
    /// # Errors
    /// Returns [`ManagerError::Kv`] if connecting to the socket fails.
    pub fn connect_default() -> Result<Self, ManagerError> {
        Self::connect(resolve_kv_socket_path())
    }

    /// Wraps an existing connected [`UnixStream`].
    #[must_use]
    pub fn from_stream(stream: UnixStream) -> Self {
        Self {
            stream,
            socket_path: None,
        }
    }

    /// Returns the configured socket path if connected via path.
    #[must_use]
    pub fn socket_path(&self) -> Option<&Path> {
        self.socket_path.as_deref()
    }

    /// Sends a liveness ping (`PingRequest` / Type ID 107) and returns the echoed nanosecond timestamp.
    ///
    /// # Errors
    /// Returns [`ManagerError`] on communication failure or protocol violation.
    pub fn ping(&mut self) -> Result<u64, ManagerError> {
        let now_ns = u64::try_from(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_err(|e| ManagerError::Kv(e.to_string()))?
                .as_nanos(),
        )
        .unwrap_or(0);

        let req = IpcMessage::PingRequest(PingRequest {
            timestamp_ns: now_ns,
        });

        send_msg_frame(&self.stream, &req, None)
            .map_err(|e| ManagerError::Kv(format!("failed to send PingRequest to k3-kv: {e}")))?;

        let (reply, _) = recv_msg_frame(&self.stream).map_err(|e| {
            ManagerError::Kv(format!("failed to receive PingResponse from k3-kv: {e}"))
        })?;

        match reply {
            IpcMessage::PingResponse(PingResponse { timestamp_ns }) => Ok(timestamp_ns),
            other => Err(ManagerError::Protocol(format!(
                "expected PingResponse from k3-kv, got {other:?}"
            ))),
        }
    }

    /// Sends a heartbeat request (alias for [`Self::ping`]).
    ///
    /// # Errors
    /// Returns [`ManagerError`] on communication failure.
    pub fn heartbeat(&mut self) -> Result<u64, ManagerError> {
        self.ping()
    }

    /// Requests allocation of a shared-memory arena for a model using explicit request geometry
    /// (`AllocateArenaRequest` / Type ID 101).
    ///
    /// Validates `req` upfront before serializing or sending over the Unix domain socket.
    ///
    /// # Errors
    /// Returns [`ManagerError::Kv`] if [`AllocateArenaRequest::validate`] fails or socket I/O
    /// fails, [`ManagerError::Status`] if `k3-kv` returns a non-success status, or
    /// [`ManagerError::Protocol`] on an unexpected response message type.
    pub fn allocate_arena_with_request(
        &mut self,
        req: AllocateArenaRequest,
    ) -> Result<AllocateArenaResponse, ManagerError> {
        req.validate()
            .map_err(|msg| ManagerError::Kv(msg.to_owned()))?;
        let msg = IpcMessage::AllocateArenaRequest(req);

        send_msg_frame(&self.stream, &msg, None).map_err(|e| {
            ManagerError::Kv(format!("failed to send AllocateArenaRequest to k3-kv: {e}"))
        })?;

        let (reply, _) = recv_msg_frame(&self.stream).map_err(|e| {
            ManagerError::Kv(format!(
                "failed to receive AllocateArenaResponse from k3-kv: {e}"
            ))
        })?;

        match reply {
            IpcMessage::AllocateArenaResponse(resp) => {
                if resp.status.is_success() {
                    Ok(resp)
                } else {
                    Err(ManagerError::Status {
                        status: resp.status,
                        message: resp.error_message.unwrap_or_else(|| {
                            format!("AllocateArena failed with status {:?}", resp.status)
                        }),
                    })
                }
            }
            other => Err(ManagerError::Protocol(format!(
                "expected AllocateArenaResponse from k3-kv, got {other:?}"
            ))),
        }
    }

    /// Requests allocation of a shared-memory arena for a model (`AllocateArenaRequest` / Type ID 101)
    /// using baseline default geometry (`num_layers = 2`, `head_count_kv = 2`, `head_dim = 64`,
    /// `vocab_size = 32_000`, `kv_dtype = KV_DTYPE_F32`).
    ///
    /// # Errors
    /// Returns [`ManagerError`] on validation failure, communication failure, or non-success status.
    pub fn allocate_arena(
        &mut self,
        model_id: &str,
        arena_bytes: u64,
        n_ctx: u32,
        max_blocks: u32,
    ) -> Result<AllocateArenaResponse, ManagerError> {
        let req = default_allocate_arena_request(model_id, arena_bytes, n_ctx, max_blocks);
        self.allocate_arena_with_request(req)
    }

    /// Requests the file descriptor for an arena passed via `SCM_RIGHTS` (`GetArenaFdRequest` / Type ID 103).
    ///
    /// # Errors
    /// Returns [`ManagerError`] on communication failure, non-success status, or missing FD.
    pub fn get_arena_fd(
        &mut self,
        arena_id: u64,
    ) -> Result<(GetArenaFdResponse, OwnedFd), ManagerError> {
        let req = IpcMessage::GetArenaFdRequest(GetArenaFdRequest { arena_id });

        send_msg_frame(&self.stream, &req, None).map_err(|e| {
            ManagerError::Kv(format!("failed to send GetArenaFdRequest to k3-kv: {e}"))
        })?;

        let (reply, opt_fd) = recv_msg_frame(&self.stream).map_err(|e| {
            ManagerError::Kv(format!(
                "failed to receive GetArenaFdResponse from k3-kv: {e}"
            ))
        })?;

        match reply {
            IpcMessage::GetArenaFdResponse(resp) => {
                if !resp.status.is_success() {
                    return Err(ManagerError::Status {
                        status: resp.status,
                        message: resp.error_message.unwrap_or_else(|| {
                            format!("GetArenaFd failed with status {:?}", resp.status)
                        }),
                    });
                }
                let fd = opt_fd.ok_or_else(|| {
                    ManagerError::Protocol(
                        "k3-kv returned GetArenaFdResponse with success but no SCM_RIGHTS FD was attached"
                            .to_owned(),
                    )
                })?;
                Ok((resp, fd))
            }
            other => Err(ManagerError::Protocol(format!(
                "expected GetArenaFdResponse from k3-kv, got {other:?}"
            ))),
        }
    }

    /// Fetches the file descriptor for an arena and maps it into process virtual memory as a [`SharedMemoryArena`].
    ///
    /// # Errors
    /// Returns [`ManagerError`] on failure.
    pub fn attach_arena(
        &mut self,
        arena_id: u64,
        arena_bytes: usize,
        name: &str,
    ) -> Result<SharedMemoryArena, ManagerError> {
        let (_resp, fd) = self.get_arena_fd(arena_id)?;
        let arena = SharedMemoryArena::from_existing_fd(fd, arena_bytes, name, false)?;
        Ok(arena)
    }

    /// Allocates an arena from an explicit [`AllocateArenaRequest`] and attaches/maps it into local
    /// virtual memory in a single convenience call.
    ///
    /// Validates `req` upfront before issuing any IPC call and deallocates the created arena if
    /// attachment fails.
    ///
    /// # Errors
    /// Returns [`ManagerError`] on validation failure, allocation failure, `arena_bytes` overflow,
    /// or shared-memory attachment failure.
    pub fn allocate_and_attach_with_request(
        &mut self,
        req: AllocateArenaRequest,
    ) -> Result<(AllocateArenaResponse, SharedMemoryArena), ManagerError> {
        req.validate()
            .map_err(|msg| ManagerError::Kv(msg.to_owned()))?;
        let model_id = req.model_id.clone();
        let alloc_resp = self.allocate_arena_with_request(req)?;
        let bytes_len = usize::try_from(alloc_resp.arena_bytes).map_err(|_| {
            let _ = self.deallocate_arena(alloc_resp.arena_id);
            ManagerError::Kv("allocated arena_bytes exceeds usize".to_owned())
        })?;
        let arena = match self.attach_arena(alloc_resp.arena_id, bytes_len, &model_id) {
            Ok(a) => a,
            Err(e) => {
                let _ = self.deallocate_arena(alloc_resp.arena_id);
                return Err(e);
            }
        };
        Ok((alloc_resp, arena))
    }

    /// Allocates an arena using baseline default geometry and attaches/maps it into local virtual
    /// memory in a single convenience call.
    ///
    /// # Errors
    /// Returns [`ManagerError`] on failure.
    pub fn allocate_and_attach(
        &mut self,
        model_id: &str,
        arena_bytes: u64,
        n_ctx: u32,
        max_blocks: u32,
    ) -> Result<(AllocateArenaResponse, SharedMemoryArena), ManagerError> {
        let req = default_allocate_arena_request(model_id, arena_bytes, n_ctx, max_blocks);
        self.allocate_and_attach_with_request(req)
    }

    /// Requests deallocation of an arena in `k3-kv` (`DeallocateArenaRequest` / Type ID 105).
    ///
    /// # Errors
    /// Returns [`ManagerError`] on communication failure or non-success status.
    pub fn deallocate_arena(
        &mut self,
        arena_id: u64,
    ) -> Result<DeallocateArenaResponse, ManagerError> {
        let req = IpcMessage::DeallocateArenaRequest(DeallocateArenaRequest { arena_id });

        send_msg_frame(&self.stream, &req, None).map_err(|e| {
            ManagerError::Kv(format!(
                "failed to send DeallocateArenaRequest to k3-kv: {e}"
            ))
        })?;

        let (reply, _) = recv_msg_frame(&self.stream).map_err(|e| {
            ManagerError::Kv(format!(
                "failed to receive DeallocateArenaResponse from k3-kv: {e}"
            ))
        })?;

        match reply {
            IpcMessage::DeallocateArenaResponse(resp) => {
                if resp.status.is_success() {
                    Ok(resp)
                } else {
                    Err(ManagerError::Status {
                        status: resp.status,
                        message: resp.error_message.unwrap_or_else(|| {
                            format!("DeallocateArena failed with status {:?}", resp.status)
                        }),
                    })
                }
            }
            other => Err(ManagerError::Protocol(format!(
                "expected DeallocateArenaResponse from k3-kv, got {other:?}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_socket_path_precedence_and_legacy_fallback() {
        // 1. Explicit JERO_KV_SOCKET overrides filesystem paths
        assert_eq!(
            resolve_kv_socket_path_with(
                |k| (k == ENV_KV_SOCKET).then(|| "/tmp/override-kv.sock".to_owned()),
                |_| true
            ),
            PathBuf::from("/tmp/override-kv.sock")
        );

        // 2. Canonical /run/jero-runner/kv.sock preferred when both exist
        assert_eq!(
            resolve_kv_socket_path_with(|_| None, |_| true),
            PathBuf::from(DEFAULT_KV_SOCKET_PATH)
        );

        // 3. Falls back to /tmp/jero-kv-cache.sock when only legacy path exists
        assert_eq!(
            resolve_kv_socket_path_with(|_| None, |p| p == Path::new(LEGACY_KV_SOCKET_PATH)),
            PathBuf::from(LEGACY_KV_SOCKET_PATH)
        );

        // 4. Defaults to /run/jero-runner/kv.sock when neither exists
        assert_eq!(
            resolve_kv_socket_path_with(|_| None, |_| false),
            PathBuf::from(DEFAULT_KV_SOCKET_PATH)
        );
    }
}

---
name: jero-manager
description: "Develop and maintain the jero-manager control plane, capacity admission controller, prefix cache, prompt router, worker supervisor, and Arrow Flight/telemetry ingestion."
---

# `jero-manager` Control Plane, Admission Control & Worker Supervision

## 1. Lockable Memory Admission Control (`InsufficientLockableMemory`)

- Always check available lockable memory against `RLIMIT_MEMLOCK` before spawning or allocating `jero-kv-cache` / `jero-engine` worker arenas. Refuse over-committed requests deterministically with `InsufficientLockableMemory` (exit code `11`) rather than allowing `mlock` exhaustion to destabilize the host kernel.

## 2. Telemetry & W3C Trace Propagation (`mu-observable-ipc`)

- Emit structured request spans, prefix-cache hit metrics, and hardware utilization snapshots (`memlock_used_bytes`, `memlock_limit_bytes`) via `mu-observable-ipc` with `DeterministicSecretRedactor` scrubbing enabled.

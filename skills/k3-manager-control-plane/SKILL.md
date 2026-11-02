---
name: k3-manager-control-plane
description: "Develop and maintain the k3-manager control plane, capacity admission controller, prefix cache, prompt router, worker supervisor, and Arrow Flight/telemetry ingestion."
---

# `k3-manager` Control Plane, Admission Control & Worker Supervision

## 1. Lockable Memory Admission Control (`InsufficientLockableMemory`)

- Always check available lockable memory against `RLIMIT_MEMLOCK` before spawning or allocating `k3-kv` / `k3-engine` worker arenas. Refuse over-committed requests deterministically with `InsufficientLockableMemory` (exit code `11`) rather than allowing `mlock` exhaustion to destabilize the host kernel.

## 2. Telemetry & W3C Trace Propagation (`muobserve-ipc`)

- Emit structured request spans, prefix-cache hit metrics, and hardware utilization snapshots (`memlock_used_bytes`, `memlock_limit_bytes`) via `muobserve-ipc` with `DeterministicSecretRedactor` scrubbing enabled.

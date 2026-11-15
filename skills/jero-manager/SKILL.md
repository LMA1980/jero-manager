---
name: jero-manager
description: "Develop and maintain the jero-manager control plane, capacity admission controller, prefix cache, prompt router, worker supervisor, and Arrow Flight/telemetry ingestion."
---

# `jero-manager` Control Plane, Admission Control & Worker Supervision

## 1. Lockable Memory Admission Control (`InsufficientLockableMemory`)

- Always check available lockable memory against `RLIMIT_MEMLOCK` before spawning or allocating `jero-kv-cache` / `jero-engine` worker arenas. Refuse over-committed requests deterministically with `InsufficientLockableMemory` (exit code `11`) rather than allowing `mlock` exhaustion to destabilize the host kernel.

## 2. Telemetry & W3C Trace Propagation (`mu-observable-ipc`)

- Emit structured request spans, prefix-cache hit metrics, and hardware utilization snapshots (`memlock_used_bytes`, `memlock_limit_bytes`) via `mu-observable-ipc` with `DeterministicSecretRedactor` scrubbing enabled.

## 3. Architecture Recipe Capability Filter (`src/store/filter.rs`)

- Keep `CapabilityFilter` (`BUILTIN_RECIPES`) synchronized with compiled `jero-engine` architecture recipes, including dense/MoE transformers (`llama`, `mistral`, `mixtral`, `phi3`, `qwen2`, `qwen3`, `qwen3moe`, `qwen35`, `qwen35moe`, `deepseek2`, `gemma2`, `gemma3`, `gemma4`, `dory`, `granite`, `foundation-sec`), ternary `bitnet*`, bidirectional/guardrail/action models (`modernbert`, `bert`, `deberta-v3`, `shieldgemma`, `llama-guard`, `functiongemma`, `octopus`, `xlam`), and SSM/hybrid recipes (`mamba`, `mamba2`, `rwkv6`, `rwkv7`, `jamba`, `zamba2`, `granitehybrid`, `lfm2`, `lfm2moe`).

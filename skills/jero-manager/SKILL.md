---
name: jero-manager
description: "Develop and maintain the jero-manager control plane, capacity admission controller, prefix cache, prompt router, worker supervisor, and Arrow Flight/telemetry ingestion."
---

# `jero-manager` Control Plane, Admission Control & Worker Supervision

## 1. Lockable Memory Admission Control (`InsufficientLockableMemory`)

- Always check available lockable memory against `RLIMIT_MEMLOCK` before spawning or allocating `jero-kv-cache` / `jero-engine` worker arenas. Refuse over-committed requests deterministically with `InsufficientLockableMemory` (exit code `11`) rather than allowing `mlock` exhaustion to destabilize the host kernel.

## 2. Telemetry & W3C Trace Propagation (`mu-observable-ipc`)

- Emit structured request spans, prefix-cache hit metrics, and hardware utilization snapshots (`memlock_used_bytes`, `memlock_limit_bytes`) via `mu-observable-ipc` with `DeterministicSecretRedactor` scrubbing enabled.

## 3. Architecture Recipe Capability Filter & `SafeTensors` Admission (`src/store/`)

- Keep `CapabilityFilter` (`BUILTIN_RECIPES` in `src/store/filter.rs`) synchronized with compiled peer architecture recipes across `jero-arch-*`, including dense/MoE transformers (`llama`, `mistral`, `mixtral`, `phi3`, `qwen2`, `qwen3`, `qwen3moe`, `qwen35`, `qwen35moe`, `deepseek2`, `gemma2`, `gemma3`, `gemma4`, `dory`, `granite`, `foundation-sec`), ternary `bitnet*`, bidirectional/guardrail/action/embedding/reranker encoders (`modernbert`, `bert`, `nomic-bert`, `bge`, `jina-bert`, `qwen3-embedding`, `deberta-v3`, `cross-encoder`, `reranker`, `bge-reranker`, `jina-reranker`, `modernbert-reranker`, `shieldgemma`, `llama-guard`, `functiongemma`, `octopus`, `xlam`), multimodal/speech recipes (`clip`, `siglip`, `whisper`), SSM/hybrid recipes (`mamba`, `mamba2`, `rwkv6`, `rwkv7`, `jamba`, `zamba2`, `granitehybrid`, `lfm2`, `lfm2moe`), and machine models (`ttm`, `ft-transformer`, `tabpfn-v2`, `decision-transformer`).
- Admit both `ModelFormat::Gguf` and `ModelFormat::SafeTensors` checkpoints (`validate_model_manifest`, `evaluate_candidate`, and `ContentAddressedStore::validate_safetensors_weights_and_admit` backed by `jero_core::SafeTensorsFile::from_bytes`), rejecting malformed headers, unsupported dtypes, and out-of-bounds tensor offsets upfront.

## 4. Multimodal Protocol Endpoints & Zero-Copy Conditioning (`src/protocol/`)

- **`POST /v1/embeddings` & `POST /v1/rerank`**: Serve OpenAI-compatible dense embeddings (`float` and little-endian IEEE-754 `base64`, optional Matryoshka `dimensions` truncation/renormalization via `jero_kernels_rvv::pool_and_l2_normalize_f32`) and cross-encoder / cosine-sigmoid document ranking via `jero-arch-encoder` and `jero-arch-qwen`.
- **`POST /v1/audio/transcriptions`**: Accept `multipart/form-data` WAV uploads and `application/json` (`audio_base64` or `pcm_f32`), enforce fail-closed RIFF/WAVE validation and bounded 16 kHz mono resampling (30 s window ceiling), and dispatch `WhisperRecipe` (`jero-arch-whisper`) for `json`, `text`, and `verbose_json` responses.
- **Vision `mmproj` Soft-Prompt Conditioning**: Extract base64-encoded PNG/PPM/BMP/raw-RGB payloads from OpenAI `image_url` data URIs (`POST /v1/chat/completions`) and Anthropic Claude `image` base64 blocks (`POST /v1/messages`), reject remote `http://`/`https://` URLs fail-closed for SSRF protection, validate bounded byte/pixel ceilings (`400`/`413`/`422`), project via `VlmMmprojBridge` (`jero-arch-vision`) into `soft_prompt_embeds`, enforce token bounds against `n_ctx`, and stage continuous tensors via `stage_soft_prompt_feature_slot` (`FeatureInputSlotHeader`).

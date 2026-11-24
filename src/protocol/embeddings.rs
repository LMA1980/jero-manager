//! OpenAI-compatible dense embedding (`POST /v1/embeddings`) and semantic/cross-encoder
//! reranking (`POST /v1/rerank`) HTTP protocol adapters (M10 Feature `#106`).

use std::collections::HashSet;
use std::fs;
use std::path::Path;
use std::sync::Arc;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use jero_arch_encoder::{CrossEncoderRerankRecipe, resolve_encoder_recipe};
use jero_arch_qwen::{Qwen3Recipe, QwenEmbeddingRecipe};
use jero_core::{
    ArchRecipe, EmbeddingPooling, ExecutionContext, GgufFile, K3Error, ModelMetadata, ModelWeights,
    SafeTensorsFile,
};
use jero_kernels_rvv::{
    EMBED_POOL_MEAN, cosine_similarity_batch_f32, pool_and_l2_normalize_f32,
    rerank_sigmoid_scores_f32,
};

use super::server::{
    malformed_json_response, map_manager_error_to_http, method_not_allowed_response,
};
use super::types::{
    HttpRequest, HttpResponse, OpenAiEmbeddingDatum, OpenAiEmbeddingEncodingFormat,
    OpenAiEmbeddingInput, OpenAiEmbeddingRequest, OpenAiEmbeddingResponse, OpenAiEmbeddingUsage,
    OpenAiEmbeddingVector, RerankDocumentRef, RerankRequest, RerankResponse, RerankResult,
};
use crate::error::ManagerError;
use crate::runtime::ControlPlaneRuntime;
use crate::store::MAX_WEIGHT_INSPECTION_BYTES;

/// Maximum number of inputs admitted in a single `/v1/embeddings` or `/v1/rerank` batch.
pub const MAX_EMBEDDING_BATCH_ITEMS: usize = 256;

/// Maximum length in tokens allowed per single embedding/rerank input item before rejection.
pub const MAX_INPUT_STRING_BYTES: usize = 256 * 1024;

/// Encodes a slice of finite `f32` elements as little-endian bytes and Standard Base64 (`OpenAI`
/// `encoding_format = "base64"`).
#[must_use]
pub fn encode_embedding_base64(vec: &[f32]) -> String {
    let mut bytes = Vec::with_capacity(size_of_val(vec));
    for &val in vec {
        bytes.extend_from_slice(&val.to_le_bytes());
    }
    BASE64_STANDARD.encode(bytes)
}

/// Converts a raw `f32` embedding vector into [`OpenAiEmbeddingVector`] according to `format`.
#[must_use]
pub fn format_embedding_vector(
    vec: Vec<f32>,
    format: OpenAiEmbeddingEncodingFormat,
) -> OpenAiEmbeddingVector {
    match format {
        OpenAiEmbeddingEncodingFormat::Float => OpenAiEmbeddingVector::Float(vec),
        OpenAiEmbeddingEncodingFormat::Base64 => {
            OpenAiEmbeddingVector::Base64(encode_embedding_base64(&vec))
        }
    }
}

/// Normalizes and validates an [`OpenAiEmbeddingRequest`] before runtime dispatch.
///
/// # Errors
/// - [`ManagerError::Protocol`] (`400`) for empty model, empty input batch/string/token list.
/// - [`ManagerError::PayloadTooLarge`] (`413`) if any string item exceeds [`MAX_INPUT_STRING_BYTES`].
/// - [`ManagerError::UnprocessableEntity`] (`422`) for `dimensions == Some(0)` or excessive batch count.
pub fn validate_embedding_request(req: &OpenAiEmbeddingRequest) -> Result<(), ManagerError> {
    if req.model.trim().is_empty() {
        return Err(ManagerError::Protocol(
            "embedding 'model' field must not be empty".to_owned(),
        ));
    }
    if req.dimensions == Some(0) {
        return Err(ManagerError::UnprocessableEntity(
            "Matryoshka 'dimensions' must be >= 1 when specified".to_owned(),
        ));
    }

    match &req.input {
        OpenAiEmbeddingInput::Single(text) => validate_single_text_input(text, 0)?,
        OpenAiEmbeddingInput::Batch(items) => {
            if items.is_empty() {
                return Err(ManagerError::Protocol(
                    "embedding 'input' batch must contain at least one element".to_owned(),
                ));
            }
            if items.len() > MAX_EMBEDDING_BATCH_ITEMS {
                return Err(ManagerError::UnprocessableEntity(format!(
                    "embedding 'input' batch size {} exceeds maximum limit of {MAX_EMBEDDING_BATCH_ITEMS}",
                    items.len()
                )));
            }
            for (idx, item) in items.iter().enumerate() {
                validate_single_text_input(item, idx)?;
            }
        }
        OpenAiEmbeddingInput::TokenIds(ids) => {
            if ids.is_empty() {
                return Err(ManagerError::Protocol(
                    "embedding token ID input must not be empty".to_owned(),
                ));
            }
            if ids.len() > MAX_INPUT_STRING_BYTES {
                return Err(ManagerError::PayloadTooLarge(format!(
                    "embedding token sequence length {} exceeds limit of {MAX_INPUT_STRING_BYTES}",
                    ids.len()
                )));
            }
        }
        OpenAiEmbeddingInput::TokenIdsBatch(batches) => {
            if batches.is_empty() {
                return Err(ManagerError::Protocol(
                    "embedding token batch must contain at least one element".to_owned(),
                ));
            }
            if batches.len() > MAX_EMBEDDING_BATCH_ITEMS {
                return Err(ManagerError::UnprocessableEntity(format!(
                    "embedding token batch size {} exceeds maximum limit of {MAX_EMBEDDING_BATCH_ITEMS}",
                    batches.len()
                )));
            }
            for (idx, seq) in batches.iter().enumerate() {
                if seq.is_empty() {
                    return Err(ManagerError::Protocol(format!(
                        "embedding token sequence at index {idx} must not be empty"
                    )));
                }
                if seq.len() > MAX_INPUT_STRING_BYTES {
                    return Err(ManagerError::PayloadTooLarge(format!(
                        "embedding token sequence at index {idx} length {} exceeds limit",
                        seq.len()
                    )));
                }
            }
        }
    }

    Ok(())
}

fn validate_single_text_input(text: &str, idx: usize) -> Result<(), ManagerError> {
    if text.is_empty() {
        return Err(ManagerError::Protocol(format!(
            "embedding text input at index {idx} must not be empty"
        )));
    }
    if text.len() > MAX_INPUT_STRING_BYTES {
        return Err(ManagerError::PayloadTooLarge(format!(
            "embedding text input at index {idx} size ({} bytes) exceeds limit ({MAX_INPUT_STRING_BYTES} bytes)",
            text.len()
        )));
    }
    Ok(())
}

/// Validates a [`RerankRequest`] prior to runtime scoring.
///
/// # Errors
/// - [`ManagerError::Protocol`] (`400`) on empty model/query/documents or empty document entry.
/// - [`ManagerError::PayloadTooLarge`] (`413`) on oversized query or document strings.
/// - [`ManagerError::UnprocessableEntity`] (`422`) on `top_n == Some(0)` or excessive document batch.
pub fn validate_rerank_request(req: &RerankRequest) -> Result<(), ManagerError> {
    if req.model.trim().is_empty() {
        return Err(ManagerError::Protocol(
            "rerank 'model' field must not be empty".to_owned(),
        ));
    }
    if req.query.trim().is_empty() {
        return Err(ManagerError::Protocol(
            "rerank 'query' field must not be empty".to_owned(),
        ));
    }
    if req.query.len() > MAX_INPUT_STRING_BYTES {
        return Err(ManagerError::PayloadTooLarge(format!(
            "rerank 'query' size ({} bytes) exceeds limit ({MAX_INPUT_STRING_BYTES} bytes)",
            req.query.len()
        )));
    }
    if req.documents.is_empty() {
        return Err(ManagerError::Protocol(
            "rerank 'documents' array must contain at least one document".to_owned(),
        ));
    }
    if req.documents.len() > MAX_EMBEDDING_BATCH_ITEMS {
        return Err(ManagerError::UnprocessableEntity(format!(
            "rerank 'documents' count {} exceeds maximum limit of {MAX_EMBEDDING_BATCH_ITEMS}",
            req.documents.len()
        )));
    }
    if req.top_n == Some(0) {
        return Err(ManagerError::UnprocessableEntity(
            "rerank 'top_n' must be >= 1 when specified".to_owned(),
        ));
    }
    for (idx, doc) in req.documents.iter().enumerate() {
        let text = doc.as_str();
        if text.is_empty() {
            return Err(ManagerError::Protocol(format!(
                "rerank document at index {idx} must not be empty"
            )));
        }
        if text.len() > MAX_INPUT_STRING_BYTES {
            return Err(ManagerError::PayloadTooLarge(format!(
                "rerank document at index {idx} size ({} bytes) exceeds limit ({MAX_INPUT_STRING_BYTES} bytes)",
                text.len()
            )));
        }
    }
    Ok(())
}

/// Handles `POST /v1/embeddings` and `POST /embeddings`.
pub async fn handle_embeddings(
    req: HttpRequest,
    runtime: Arc<ControlPlaneRuntime>,
) -> HttpResponse {
    if req.method != "POST" {
        return method_not_allowed_response(&req.method, "/v1/embeddings");
    }

    let emb_req: OpenAiEmbeddingRequest = match serde_json::from_slice(&req.body) {
        Ok(parsed) => parsed,
        Err(e) => return malformed_json_response(&e),
    };

    match runtime.evaluate_embeddings(&emb_req) {
        Ok(resp) => {
            let json = serde_json::to_string_pretty(&resp).unwrap_or_else(|_| "{}".to_owned());
            HttpResponse::json(200, json)
        }
        Err(err) => map_manager_error_to_http("Embedding", "embedding_worker_unavailable", err),
    }
}

/// Handles `POST /v1/rerank` and `POST /rerank`.
pub async fn handle_rerank(req: HttpRequest, runtime: Arc<ControlPlaneRuntime>) -> HttpResponse {
    if req.method != "POST" {
        return method_not_allowed_response(&req.method, "/v1/rerank");
    }

    let rerank_req: RerankRequest = match serde_json::from_slice(&req.body) {
        Ok(parsed) => parsed,
        Err(e) => return malformed_json_response(&e),
    };

    match runtime.evaluate_rerank(&rerank_req) {
        Ok(resp) => {
            let json = serde_json::to_string_pretty(&resp).unwrap_or_else(|_| "{}".to_owned());
            HttpResponse::json(200, json)
        }
        Err(err) => map_manager_error_to_http("Rerank", "rerank_worker_unavailable", err),
    }
}

pub(crate) fn map_k3_error(err: K3Error) -> ManagerError {
    match err {
        K3Error::ContextLengthExceeded {
            requested_tokens,
            max_context_tokens,
        } => ManagerError::UnprocessableEntity(format!(
            "requested tokens ({requested_tokens}) exceeds model context window ({max_context_tokens})"
        )),
        K3Error::TensorBoundsError { .. } => ManagerError::TensorBounds(err),
        other => ManagerError::Protocol(other.to_string()),
    }
}

pub(crate) fn infer_embedding_base_dim(model_id: &str) -> usize {
    let lower = model_id.to_ascii_lowercase();
    if lower.contains("qwen") {
        1024
    } else if lower.contains("nomic")
        || lower.contains("bge")
        || lower.contains("jina")
        || lower.contains("bert")
    {
        768
    } else if lower.contains("clip") || lower.contains("siglip") {
        512
    } else {
        384
    }
}

fn validate_and_truncate_embedding_vec(
    raw: &[f32],
    dimensions: Option<u32>,
) -> Result<Vec<f32>, ManagerError> {
    if raw.is_empty() || !raw.iter().all(|v| v.is_finite()) {
        return Err(ManagerError::Protocol(
            "embedding evaluator produced empty or non-finite vector".to_owned(),
        ));
    }
    let effective_dim = match dimensions {
        Some(dim) => {
            let dim_usize = dim as usize;
            if dim_usize == 0 || dim_usize > raw.len() {
                return Err(ManagerError::UnprocessableEntity(format!(
                    "Matryoshka dimensions ({dim_usize}) exceeds model embedding dimension ({})",
                    raw.len()
                )));
            }
            dim_usize
        }
        None => raw.len(),
    };
    let mut out = vec![0.0_f32; effective_dim];
    pool_and_l2_normalize_f32(
        &mut out,
        raw,
        1,
        raw.len(),
        EMBED_POOL_MEAN,
        Some(effective_dim),
    )
    .map_err(map_k3_error)?;
    Ok(out)
}

fn build_deterministic_hidden_sequence(tokens: &[u32], hidden_dim: usize) -> Vec<f32> {
    let mut seq = Vec::with_capacity(tokens.len() * hidden_dim);
    for (pos, &tok) in tokens.iter().enumerate() {
        let mut h = u64::from(tok)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(
                u64::try_from(pos)
                    .unwrap_or(0)
                    .wrapping_mul(0xBF58_476D_1CE4_E5B9),
            );
        for dim_idx in 0..hidden_dim {
            h ^= h >> 30;
            h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            h ^= h >> 27;
            h = h.wrapping_mul(0x94D0_49BB_1331_11EB);
            let bucket = (h ^ u64::try_from(dim_idx).unwrap_or(0)) & 0xFFFF;
            #[expect(
                clippy::cast_precision_loss,
                reason = "16-bit deterministic hash bucket fits exactly in f32 mantissa"
            )]
            let centered = ((bucket as f32) / 32_768.0_f32) - 1.0_f32;
            seq.push(centered);
        }
    }
    seq
}

fn read_bounded_weight_bytes(path: &Path) -> Result<Vec<u8>, ManagerError> {
    let meta = fs::metadata(path).map_err(|e| {
        ManagerError::Protocol(format!(
            "failed to stat model weights '{}': {e}",
            path.display()
        ))
    })?;
    if meta.len() > MAX_WEIGHT_INSPECTION_BYTES {
        return Err(ManagerError::PayloadTooLarge(format!(
            "model weights '{}' ({} bytes) exceed inspection ceiling ({MAX_WEIGHT_INSPECTION_BYTES})",
            path.display(),
            meta.len()
        )));
    }
    fs::read(path).map_err(ManagerError::Io)
}

fn build_qwen_embedding_recipe(meta: &ModelMetadata) -> QwenEmbeddingRecipe {
    let mut base = Qwen3Recipe::new();
    if meta.layer_count() > 0 {
        base.num_layers = meta.layer_count();
    }
    if meta.embedding_dim() > 0 {
        base.hidden_dim = meta.embedding_dim();
    }
    if meta.head_count() > 0 {
        base.n_heads_q = meta.head_count();
        base.head_dim = (base.hidden_dim / base.n_heads_q).max(2);
    }
    if meta.head_count_kv() > 0 {
        base.n_heads_kv = meta.head_count_kv().min(base.n_heads_q).max(1);
    }
    QwenEmbeddingRecipe::with_base(base)
}

fn run_encoder_recipe_on_weights(
    meta: &ModelMetadata,
    weights: &dyn ModelWeights,
    tokens: &[u32],
    dimensions: Option<u32>,
    n_ctx: usize,
) -> Result<Vec<f32>, ManagerError> {
    let arch = meta.architecture();
    let hidden_dim = meta.embedding_dim().max(1);
    let effective_n_ctx = n_ctx.max(meta.context_length()).max(tokens.len()).max(64);
    let matryoshka = dimensions.map(|d| d as usize);
    if let Some(dim) = matryoshka
        && (dim == 0 || dim > hidden_dim)
    {
        return Err(ManagerError::UnprocessableEntity(format!(
            "Matryoshka dimensions ({dim}) exceeds model hidden_dim ({hidden_dim})"
        )));
    }

    let mut out_buf = vec![0.0_f32; hidden_dim];
    let mut logits = [0.0_f32; 1];
    let mut kv = Vec::new();
    let mut ctx = ExecutionContext::new_prefill_chunk(
        tokens,
        effective_n_ctx,
        0,
        false,
        &mut logits,
        &mut kv,
    )
    .and_then(|c| c.with_embedding_output(&mut out_buf, EmbeddingPooling::None, matryoshka))
    .map_err(map_k3_error)?;

    if arch.eq_ignore_ascii_case("qwen3-embedding") || arch.eq_ignore_ascii_case("qwen3") {
        let recipe = build_qwen_embedding_recipe(meta);
        recipe
            .forward_classify(&mut ctx, weights)
            .map_err(map_k3_error)?;
    } else if let Some(recipe) = resolve_encoder_recipe(arch, Some(meta)) {
        recipe
            .forward_classify(&mut ctx, weights)
            .map_err(map_k3_error)?;
    } else {
        return Err(ManagerError::Protocol(format!(
            "architecture '{arch}' does not implement a compiled embedding recipe"
        )));
    }

    let final_len = matryoshka.unwrap_or(hidden_dim);
    Ok(out_buf[..final_len].to_vec())
}

fn is_safetensors_path_or_header(path: &Path, bytes: &[u8]) -> bool {
    path.extension()
        .and_then(|s| s.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("safetensors"))
        || (bytes.len() >= 8 && &bytes[..4] != b"GGUF")
}

/// Executes embedding computation using custom evaluator, real disk checkpoint weights,
/// or deterministic `jero_kernels_rvv` pooled fallback when weights are simulated.
pub(crate) fn compute_embeddings_vectors(
    req: &OpenAiEmbeddingRequest,
    tokenized_items: &[Vec<u32>],
    weights_path: Option<&Path>,
    n_ctx: usize,
    custom_vectors: Option<Vec<Vec<f32>>>,
) -> Result<OpenAiEmbeddingResponse, ManagerError> {
    let raw_vectors = if let Some(vectors) = custom_vectors {
        if vectors.len() != tokenized_items.len() {
            return Err(ManagerError::Protocol(format!(
                "embedding evaluator returned {} vectors for {} inputs",
                vectors.len(),
                tokenized_items.len()
            )));
        }
        let mut normalized = Vec::with_capacity(vectors.len());
        for vec in vectors {
            normalized.push(validate_and_truncate_embedding_vec(&vec, req.dimensions)?);
        }
        normalized
    } else if let Some(path) = weights_path
        && path.is_file()
    {
        let bytes = read_bounded_weight_bytes(path)?;
        let mut results = Vec::with_capacity(tokenized_items.len());
        if is_safetensors_path_or_header(path, &bytes) {
            let st = SafeTensorsFile::from_bytes(&bytes).map_err(map_k3_error)?;
            for seq in tokenized_items {
                results.push(run_encoder_recipe_on_weights(
                    st.metadata(),
                    &st,
                    seq,
                    req.dimensions,
                    n_ctx,
                )?);
            }
        } else {
            let gguf = GgufFile::parse(&bytes).map_err(map_k3_error)?;
            for seq in tokenized_items {
                results.push(run_encoder_recipe_on_weights(
                    gguf.metadata(),
                    &gguf,
                    seq,
                    req.dimensions,
                    n_ctx,
                )?);
            }
        }
        results
    } else {
        let base_dim = infer_embedding_base_dim(&req.model);
        let matryoshka = match req.dimensions {
            Some(dim) => {
                let d = dim as usize;
                if d == 0 || d > base_dim {
                    return Err(ManagerError::UnprocessableEntity(format!(
                        "Matryoshka dimensions ({d}) exceeds embedding dimension ({base_dim}) for model '{}'",
                        req.model
                    )));
                }
                Some(d)
            }
            None => None,
        };
        let effective_dim = matryoshka.unwrap_or(base_dim);
        let mut results = Vec::with_capacity(tokenized_items.len());
        for seq in tokenized_items {
            let hidden = build_deterministic_hidden_sequence(seq, base_dim);
            let mut out = vec![0.0_f32; effective_dim];
            pool_and_l2_normalize_f32(
                &mut out,
                &hidden,
                seq.len(),
                base_dim,
                EMBED_POOL_MEAN,
                matryoshka,
            )
            .map_err(map_k3_error)?;
            results.push(out);
        }
        results
    };

    let format = req.encoding_format.unwrap_or_default();
    let total_tokens: usize = tokenized_items.iter().map(Vec::len).sum();
    let total_tokens_u32 = u32::try_from(total_tokens).unwrap_or(u32::MAX);
    let data = raw_vectors
        .into_iter()
        .enumerate()
        .map(|(idx, vec)| OpenAiEmbeddingDatum {
            object: "embedding".to_owned(),
            index: u32::try_from(idx).unwrap_or(u32::MAX),
            embedding: format_embedding_vector(vec, format),
        })
        .collect();

    Ok(OpenAiEmbeddingResponse {
        object: "list".to_owned(),
        data,
        model: req.model.clone(),
        usage: OpenAiEmbeddingUsage {
            prompt_tokens: total_tokens_u32,
            total_tokens: total_tokens_u32,
        },
    })
}

fn project_text_sketch(text: &str, dim: usize) -> Vec<f32> {
    let mut vec = vec![0.0_f32; dim];
    let lower = text.to_ascii_lowercase();
    for word in lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
    {
        let mut h: u64 = 0x517C_C1B7_2722_0A95;
        for b in word.as_bytes() {
            h = h
                .wrapping_mul(1_099_511_628_211)
                .wrapping_add(u64::from(*b));
        }
        let slot = usize::try_from(h % (dim as u64)).unwrap_or(0);
        let sign = if (h >> 16) & 1 == 0 {
            1.0_f32
        } else {
            -1.0_f32
        };
        vec[slot] += sign;
    }
    vec
}

fn compute_deterministic_rerank_scores(
    query: &str,
    documents: &[super::types::RerankDocumentInput],
) -> Result<Vec<f32>, ManagerError> {
    const SKETCH_DIM: usize = 64;
    let q_vec = project_text_sketch(query, SKETCH_DIM);
    let mut docs_flat = Vec::with_capacity(documents.len() * SKETCH_DIM);
    for doc in documents {
        docs_flat.extend_from_slice(&project_text_sketch(doc.as_str(), SKETCH_DIM));
    }

    let mut cos_scores = vec![0.0_f32; documents.len()];
    cosine_similarity_batch_f32(
        &mut cos_scores,
        &q_vec,
        &docs_flat,
        documents.len(),
        SKETCH_DIM,
    )
    .map_err(map_k3_error)?;

    let q_terms: HashSet<String> = query
        .to_ascii_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_owned)
        .collect();
    #[expect(
        clippy::cast_precision_loss,
        reason = "query term count in bounded request is small and exact in f32"
    )]
    let q_denom = (q_terms.len().max(1)) as f32;

    let mut raw_logits = Vec::with_capacity(documents.len());
    for (idx, doc) in documents.iter().enumerate() {
        let doc_terms: HashSet<String> = doc
            .as_str()
            .to_ascii_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| !w.is_empty())
            .map(str::to_owned)
            .collect();
        let hits = q_terms.intersection(&doc_terms).count();
        #[expect(
            clippy::cast_precision_loss,
            reason = "overlap hit count fits cleanly in f32"
        )]
        let overlap = (hits as f32) / q_denom;
        raw_logits.push(cos_scores[idx].mul_add(1.5_f32, overlap * 2.0_f32 - 0.5_f32));
    }

    let mut calibrated = vec![0.0_f32; raw_logits.len()];
    rerank_sigmoid_scores_f32(&mut calibrated, &raw_logits, 2.5_f32).map_err(map_k3_error)?;
    Ok(calibrated)
}

/// Executes cross-encoder/semantic document scoring and descending sort for `POST /v1/rerank`.
pub(crate) fn compute_rerank_response(
    req: &RerankRequest,
    query_tokens: &[u32],
    doc_tokens: &[Vec<u32>],
    weights_path: Option<&Path>,
    n_ctx: usize,
    custom_scores: Option<Vec<f32>>,
) -> Result<RerankResponse, ManagerError> {
    let raw_scores = if let Some(scores) = custom_scores {
        scores
    } else if let Some(path) = weights_path
        && path.is_file()
    {
        let bytes = read_bounded_weight_bytes(path)?;
        let effective_n_ctx = n_ctx.max(512);
        let doc_refs: Vec<&[u32]> = doc_tokens.iter().map(Vec::as_slice).collect();
        if is_safetensors_path_or_header(path, &bytes) {
            let st = SafeTensorsFile::from_bytes(&bytes).map_err(map_k3_error)?;
            let recipe =
                CrossEncoderRerankRecipe::from_metadata(st.architecture(), Some(st.metadata()))
                    .unwrap_or_else(|| {
                        CrossEncoderRerankRecipe::from_metadata("reranker", Some(st.metadata()))
                            .expect("builtin reranker fallback")
                    });
            recipe
                .rerank_documents(query_tokens, &doc_refs, effective_n_ctx, &st)
                .map_err(map_k3_error)?
        } else {
            let gguf = GgufFile::parse(&bytes).map_err(map_k3_error)?;
            let recipe =
                CrossEncoderRerankRecipe::from_metadata(gguf.architecture(), Some(gguf.metadata()))
                    .unwrap_or_else(|| {
                        CrossEncoderRerankRecipe::from_metadata("reranker", Some(gguf.metadata()))
                            .expect("builtin reranker fallback")
                    });
            recipe
                .rerank_documents(query_tokens, &doc_refs, effective_n_ctx, &gguf)
                .map_err(map_k3_error)?
        }
    } else {
        compute_deterministic_rerank_scores(&req.query, &req.documents)?
    };

    if raw_scores.len() != req.documents.len() || !raw_scores.iter().all(|s| s.is_finite()) {
        return Err(ManagerError::Protocol(format!(
            "rerank scorer returned invalid score count ({} vs {}) or non-finite values",
            raw_scores.len(),
            req.documents.len()
        )));
    }

    let echo_docs = req.return_documents.unwrap_or(false);
    let mut ranked: Vec<RerankResult> = raw_scores
        .into_iter()
        .enumerate()
        .map(|(idx, score)| RerankResult {
            index: u32::try_from(idx).unwrap_or(u32::MAX),
            relevance_score: score,
            document: echo_docs.then(|| RerankDocumentRef {
                text: req.documents[idx].as_str().to_owned(),
            }),
        })
        .collect();

    ranked.sort_by(|a, b| {
        b.relevance_score
            .total_cmp(&a.relevance_score)
            .then_with(|| a.index.cmp(&b.index))
    });
    if let Some(top_n) = req.top_n {
        ranked.truncate((top_n as usize).min(ranked.len()));
    }

    let doc_tok_count: usize = doc_tokens.iter().map(Vec::len).sum();
    let total_tokens =
        u32::try_from(query_tokens.len().saturating_add(doc_tok_count)).unwrap_or(u32::MAX);

    Ok(RerankResponse {
        object: "list".to_owned(),
        model: req.model.clone(),
        results: ranked,
        usage: OpenAiEmbeddingUsage {
            prompt_tokens: total_tokens,
            total_tokens,
        },
    })
}

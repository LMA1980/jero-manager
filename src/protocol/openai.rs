//! OpenAI-compatible client protocol front-end adapter (US-23, ADR-015, ADR-020, ADR-027, ADR-031).
//!
//! Exposes standard OpenAI HTTP endpoints:
//! - `GET /v1/models`: Discovers active/loaded models with unprofiled tagging and advisories.
//! - `POST /v1/chat/completions`: Full chat completion pipeline supporting synchronous JSON,
//!   Server-Sent Events (SSE) streaming with keep-alives and queue positions, transparent
//!   tool calling pass-through, and protocol-native error mapping (HTTP 400 / 429).
#![allow(
    clippy::pedantic,
    clippy::doc_markdown,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::unused_async,
    clippy::too_many_lines
)]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use jero_core::error::K3Error;

use crate::error::ManagerError;
use crate::prompt::normalize_guardrail_input;
use crate::protocol::ThinkingTagParser;
use crate::protocol::types::{
    GuardrailEvaluateResponse, GuardrailPolicy, HttpRequest, HttpResponse, MessageContent,
    OpenAiChatRequest, OpenAiChatResponse, OpenAiChoice, OpenAiChunkChoice, OpenAiChunkDelta,
    OpenAiChunkResponse, OpenAiErrorDetails, OpenAiErrorResponse, OpenAiFunctionCall,
    OpenAiFunctionCallChunk, OpenAiResponseMessage, OpenAiToolCall, OpenAiToolCallChunk,
    OpenAiUsage,
};
use crate::runtime::{ControlPlaneRuntime, classify_canonical_security_text};

/// Unique request ID counter for deterministic or pseudo-unique ID generation.
static OPENAI_REQUEST_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1000);

/// Generates a unique completion identifier (`chatcmpl-<id>`).
fn generate_completion_id() -> String {
    let seq = OPENAI_REQUEST_SEQ.fetch_add(1, Ordering::SeqCst);
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    format!("chatcmpl-{now:x}{seq:x}")
}

/// Retrieves current Unix timestamp in seconds.
fn current_unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// OpenAI-compatible front-end protocol adapter (US-23).
#[derive(Debug, Default, Clone, Copy)]
pub struct OpenAiAdapter;

impl OpenAiAdapter {
    /// Creates a new OpenAI protocol adapter instance.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Handles `GET /v1/models` request (US-23 Scenario 2, ADR-031).
    ///
    /// Leverages `v1_models_response()` on [`ControlPlaneRuntime`], correctly surfacing
    /// `unprofiled: true` / `profile_status: "unprofiled"` and tuning advisory notices.
    #[must_use]
    pub fn handle_models(runtime: &ControlPlaneRuntime) -> HttpResponse {
        let resp = runtime.v1_models_response();
        match serde_json::to_string_pretty(&resp) {
            Ok(json) => HttpResponse::json(200, json),
            Err(e) => {
                let err = OpenAiErrorResponse {
                    error: OpenAiErrorDetails {
                        message: format!("Failed to serialize models response: {e}"),
                        r#type: "internal_server_error".to_owned(),
                        param: None,
                        code: Some("serialization_error".to_owned()),
                    },
                };
                let err_json = serde_json::to_string(&err).unwrap_or_else(|_| "{}".to_owned());
                HttpResponse::json(500, err_json)
            }
        }
    }

    /// Handles `POST /v1/chat/completions` request (US-23 Scenario 1 & 3).
    ///
    /// Handles both non-streaming JSON and streaming SSE with queue feedback,
    /// context length verification, tool calling pass-through, and error mapping.
    pub async fn handle_chat_completions(
        req: OpenAiChatRequest,
        runtime: Arc<ControlPlaneRuntime>,
    ) -> HttpResponse {
        Self::handle_chat_completions_traced(req, runtime, None).await
    }

    /// Handles `POST /v1/chat/completions` request with optional W3C `traceparent` propagation.
    pub async fn handle_chat_completions_traced(
        req: OpenAiChatRequest,
        runtime: Arc<ControlPlaneRuntime>,
        traceparent: Option<String>,
    ) -> HttpResponse {
        let req_start = std::time::Instant::now();
        // 1. Model resolution
        let model_entry = runtime.get_model_entry(&req.model);
        if model_entry.is_none() {
            let err = OpenAiErrorResponse {
                error: OpenAiErrorDetails {
                    message: format!("Model '{}' not loaded or does not exist", req.model),
                    r#type: "invalid_request_error".to_owned(),
                    param: Some("model".to_owned()),
                    code: Some("model_not_found".to_owned()),
                },
            };
            let json = serde_json::to_string_pretty(&err).unwrap_or_default();
            return HttpResponse::error(404, json, Vec::new());
        }
        let entry = model_entry.unwrap();
        let n_ctx = entry.n_ctx;

        // 2. Build concatenated text prompt for tokenization and context checking
        let mut raw_prompt_text = String::new();
        for msg in &req.messages {
            raw_prompt_text.push_str(&msg.role);
            raw_prompt_text.push_str(": ");
            if let Some(content) = &msg.content {
                raw_prompt_text.push_str(&content.as_text());
            }
            raw_prompt_text.push('\n');
        }
        let prompt_text = normalize_guardrail_input(&raw_prompt_text);

        // 3. Tokenize prompt & check context window bounds (ADR-027, US-23 Scenario 3)
        // Use runtime tokenizer if registered, otherwise fallback to whitespace/character approximation
        let token_count = match runtime.dispatch_tokenize_prompt(
            &req.model,
            &prompt_text,
            true,
            n_ctx,
        ) {
            Ok(tokens) => tokens.len(),
            Err(e) => {
                // Check if context length was exceeded
                let is_overflow = format!("{e}").contains("context_length_exceeded");
                if is_overflow {
                    let err = OpenAiErrorResponse {
                        error: OpenAiErrorDetails {
                            message: format!(
                                "Prompt tokens exceed model context window limit of {n_ctx} tokens"
                            ),
                            r#type: "invalid_request_error".to_owned(),
                            param: Some("messages".to_owned()),
                            code: Some("context_length_exceeded".to_owned()),
                        },
                    };
                    let json = serde_json::to_string_pretty(&err).unwrap_or_default();
                    return HttpResponse::error(400, json, Vec::new());
                }
                // If tokenizer was not registered, count approximate tokens (words / whitespace chunks)
                let approx = prompt_text.split_whitespace().count().max(1);
                if n_ctx > 0 && approx >= n_ctx {
                    let err = OpenAiErrorResponse {
                        error: OpenAiErrorDetails {
                            message: format!(
                                "Prompt tokens ({approx}) exceed model context window limit of {n_ctx} tokens"
                            ),
                            r#type: "invalid_request_error".to_owned(),
                            param: Some("messages".to_owned()),
                            code: Some("context_length_exceeded".to_owned()),
                        },
                    };
                    let json = serde_json::to_string_pretty(&err).unwrap_or_default();
                    return HttpResponse::error(400, json, Vec::new());
                }
                approx
            }
        };

        if n_ctx > 0 && token_count >= n_ctx {
            let err = OpenAiErrorResponse {
                error: OpenAiErrorDetails {
                    message: format!(
                        "Prompt tokens ({token_count}) exceed model context window limit of {n_ctx} tokens"
                    ),
                    r#type: "invalid_request_error".to_owned(),
                    param: Some("messages".to_owned()),
                    code: Some("context_length_exceeded".to_owned()),
                },
            };
            let json = serde_json::to_string_pretty(&err).unwrap_or_default();
            return HttpResponse::error(400, json, Vec::new());
        }

        // 4. Admission Queue Check (US-13, ADR-015)
        let request_id = generate_completion_id();
        let is_streaming = req.stream.unwrap_or(false);
        let cancel_token = Arc::new(AtomicBool::new(false));

        // Detect if tool calling should be triggered for pass-through (ADR-020)
        let tool_call_trigger = Self::detect_tool_call_passthrough(&req);

        // 4b. Fail-closed inline guardrail gate (pre-flight prompt + post-flight tool_calls audit)
        if let Some(policy) = runtime.guardrail_policy() {
            let tool_info = tool_call_trigger
                .as_ref()
                .map(|tc| (tc.function.name.as_str(), tc.function.arguments.as_str()));
            if let Err(reason) = crate::protocol::evaluate_inline_guardrail_gate(
                &runtime,
                &policy,
                &prompt_text,
                tool_info,
            ) {
                let err = OpenAiErrorResponse {
                    error: OpenAiErrorDetails {
                        message: reason,
                        r#type: "security_policy_violation".to_owned(),
                        param: None,
                        code: Some("security_policy_violation".to_owned()),
                    },
                };
                let json = serde_json::to_string_pretty(&err).unwrap_or_default();
                return HttpResponse::error(403, json, Vec::new());
            }
        }

        let raw_reply = Self::raw_simulated_reply(&req.model, &prompt_text);

        if is_streaming {
            // Streaming SSE branch (stream: true)
            let (sse_tx, sse_rx) = tokio::sync::mpsc::channel::<String>(128);

            let admit_res = runtime.queue_mgr().admit_to_model(
                &req.model,
                request_id.clone(),
                true,
                None,
                None,
                Arc::clone(&cancel_token),
                None,
                Some(sse_tx.clone()),
            );

            let initial_pos = match admit_res {
                Ok(pos) => pos,
                Err(K3Error::QueueSaturated {
                    current_depth,
                    max_depth,
                    retry_after_secs,
                    ..
                }) => {
                    let err = OpenAiErrorResponse {
                        error: OpenAiErrorDetails {
                            message: format!(
                                "Model admission queue saturated ({current_depth}/{max_depth})"
                            ),
                            r#type: "requests".to_owned(),
                            param: None,
                            code: Some("rate_limit_exceeded".to_owned()),
                        },
                    };
                    let json = serde_json::to_string_pretty(&err).unwrap_or_default();
                    return HttpResponse::error(
                        429,
                        json,
                        vec![("Retry-After".to_owned(), retry_after_secs.to_string())],
                    );
                }
                Err(e) => {
                    let err = OpenAiErrorResponse {
                        error: OpenAiErrorDetails {
                            message: format!("Admission failed: {e}"),
                            r#type: "internal_server_error".to_owned(),
                            param: None,
                            code: Some("queue_error".to_owned()),
                        },
                    };
                    let json = serde_json::to_string_pretty(&err).unwrap_or_default();
                    return HttpResponse::error(500, json, Vec::new());
                }
            };

            // Spawn streaming event generator
            let model_name = req.model.clone();
            let runtime_clone = Arc::clone(&runtime);
            let sse_sender = sse_tx.clone();
            let entry_clone = entry.clone();
            let p_tokens_stream = u32::try_from(token_count).unwrap_or(u32::MAX);
            let traceparent_stream = traceparent.clone();
            let raw_reply_stream = raw_reply;

            tokio::spawn(async move {
                // Dequeue permit on execution start so QueuedRequest.sse_sender clone is released
                let _permit = runtime_clone.queue_mgr().pop_next_for_model(&model_name);
                let queue_ns = u64::try_from(req_start.elapsed().as_nanos()).unwrap_or(10_000);

                // If client was queued, emit keep-alive and queue position comments
                let _ = sse_sender.send(": keep-alive\n\n".to_owned()).await;
                let _ = sse_sender
                    .send(format!(": queue_position={initial_pos}\n\n"))
                    .await;
                let _ = sse_sender
                    .send(format!(": queue-position {initial_pos}\n\n"))
                    .await;

                let created = current_unix_timestamp();

                // 1. Initial role chunk
                let chunk_start = OpenAiChunkResponse {
                    id: request_id.clone(),
                    object: "chat.completion.chunk".to_owned(),
                    created,
                    model: model_name.clone(),
                    choices: vec![OpenAiChunkChoice {
                        index: 0,
                        delta: OpenAiChunkDelta {
                            role: Some("assistant".to_owned()),
                            content: Some(String::new()),
                            reasoning_content: None,
                            tool_calls: None,
                        },
                        finish_reason: None,
                    }],
                };
                if let Ok(json) = serde_json::to_string(&chunk_start) {
                    let _ = sse_sender.send(format!("data: {json}\n\n")).await;
                }

                let stream_decode_tokens = if let Some(tool_call) = tool_call_trigger {
                    // Transparent tool call delta pass-through
                    let tool_chunk = OpenAiChunkResponse {
                        id: request_id.clone(),
                        object: "chat.completion.chunk".to_owned(),
                        created,
                        model: model_name.clone(),
                        choices: vec![OpenAiChunkChoice {
                            index: 0,
                            delta: OpenAiChunkDelta {
                                role: None,
                                content: None,
                                reasoning_content: None,
                                tool_calls: Some(vec![OpenAiToolCallChunk {
                                    index: 0,
                                    id: Some(tool_call.id),
                                    r#type: Some(tool_call.r#type),
                                    function: Some(OpenAiFunctionCallChunk {
                                        name: Some(tool_call.function.name),
                                        arguments: Some(tool_call.function.arguments),
                                    }),
                                }]),
                            },
                            finish_reason: None,
                        }],
                    };
                    if let Ok(json) = serde_json::to_string(&tool_chunk) {
                        let _ = sse_sender.send(format!("data: {json}\n\n")).await;
                    }

                    // Final choice chunk with finish_reason "tool_calls"
                    let chunk_final = OpenAiChunkResponse {
                        id: request_id.clone(),
                        object: "chat.completion.chunk".to_owned(),
                        created,
                        model: model_name.clone(),
                        choices: vec![OpenAiChunkChoice {
                            index: 0,
                            delta: OpenAiChunkDelta::default(),
                            finish_reason: Some("tool_calls".to_owned()),
                        }],
                    };
                    if let Ok(json) = serde_json::to_string(&chunk_final) {
                        let _ = sse_sender.send(format!("data: {json}\n\n")).await;
                    }
                    16_u32
                } else {
                    // Reasoning + visible content deltas routed via ThinkingTagParser
                    for text_chunk in Self::format_streaming_chunks(
                        &request_id,
                        &model_name,
                        created,
                        &[raw_reply_stream],
                    ) {
                        if let Ok(json) = serde_json::to_string(&text_chunk) {
                            let _ = sse_sender.send(format!("data: {json}\n\n")).await;
                        }
                    }

                    // Final choice chunk with finish_reason "stop"
                    let chunk_final = OpenAiChunkResponse {
                        id: request_id.clone(),
                        object: "chat.completion.chunk".to_owned(),
                        created,
                        model: model_name.clone(),
                        choices: vec![OpenAiChunkChoice {
                            index: 0,
                            delta: OpenAiChunkDelta::default(),
                            finish_reason: Some("stop".to_owned()),
                        }],
                    };
                    if let Ok(json) = serde_json::to_string(&chunk_final) {
                        let _ = sse_sender.send(format!("data: {json}\n\n")).await;
                    }
                    8_u32
                };

                // Standard OpenAI stream termination
                let _ = sse_sender.send("data: [DONE]\n\n".to_owned()).await;

                // Notify admission queue execution completion
                runtime_clone
                    .queue_mgr()
                    .mark_execution_completed(&model_name);

                runtime_clone
                    .emit_inference_telemetry_for_request(
                        "/v1/chat/completions",
                        &request_id,
                        &entry_clone,
                        traceparent_stream.as_deref(),
                        p_tokens_stream,
                        stream_decode_tokens,
                        queue_ns.max(5_000),
                        12_000,
                        1_500_000,
                        4_000_000,
                    )
                    .await;
            });

            HttpResponse::sse_stream(sse_rx)
        } else {
            // Non-streaming JSON branch (stream: false)
            let admit_res = runtime.queue_mgr().admit_to_model(
                &req.model,
                request_id.clone(),
                false,
                None,
                None,
                Arc::clone(&cancel_token),
                None,
                None,
            );

            if let Err(K3Error::QueueSaturated {
                current_depth,
                max_depth,
                retry_after_secs,
                ..
            }) = admit_res
            {
                let err = OpenAiErrorResponse {
                    error: OpenAiErrorDetails {
                        message: format!(
                            "Model admission queue saturated ({current_depth}/{max_depth})"
                        ),
                        r#type: "requests".to_owned(),
                        param: None,
                        code: Some("rate_limit_exceeded".to_owned()),
                    },
                };
                let json = serde_json::to_string_pretty(&err).unwrap_or_default();
                return HttpResponse::error(
                    429,
                    json,
                    vec![("Retry-After".to_owned(), retry_after_secs.to_string())],
                );
            }

            // Dequeue permit on execution start
            let _permit = runtime.queue_mgr().pop_next_for_model(&req.model);

            let created = current_unix_timestamp();
            let (choices, completion_tokens) = if let Some(tool_call) = tool_call_trigger {
                let choice = OpenAiChoice {
                    index: 0,
                    message: OpenAiResponseMessage {
                        role: "assistant".to_owned(),
                        content: None,
                        reasoning_content: None,
                        tool_calls: Some(vec![tool_call]),
                    },
                    finish_reason: Some("tool_calls".to_owned()),
                };
                (vec![choice], 16u32)
            } else {
                let (reasoning_content, visible_content) =
                    ThinkingTagParser::split_full_text(raw_reply);
                let choice = OpenAiChoice {
                    index: 0,
                    message: OpenAiResponseMessage {
                        role: "assistant".to_owned(),
                        content: Some(visible_content),
                        reasoning_content,
                        tool_calls: None,
                    },
                    finish_reason: Some("stop".to_owned()),
                };
                (vec![choice], 8u32)
            };

            let p_tokens = u32::try_from(token_count).unwrap_or(u32::MAX);
            let queue_ns = u64::try_from(req_start.elapsed().as_nanos()).unwrap_or(10_000);
            runtime
                .emit_inference_telemetry_for_request(
                    "/v1/chat/completions",
                    &request_id,
                    &entry,
                    traceparent.as_deref(),
                    p_tokens,
                    completion_tokens,
                    queue_ns.max(5_000),
                    12_000,
                    1_500_000,
                    4_000_000,
                )
                .await;

            let response = OpenAiChatResponse {
                id: request_id,
                object: "chat.completion".to_owned(),
                created,
                model: req.model.clone(),
                choices,
                usage: OpenAiUsage {
                    prompt_tokens: p_tokens,
                    completion_tokens,
                    total_tokens: p_tokens.saturating_add(completion_tokens),
                },
            };

            runtime.queue_mgr().mark_execution_completed(&req.model);

            let json = serde_json::to_string_pretty(&response).unwrap_or_default();
            HttpResponse::json(200, json)
        }
    }

    /// Dispatches generic HTTP requests targeting OpenAI endpoints.
    pub async fn handle_http(
        http_req: HttpRequest,
        runtime: Arc<ControlPlaneRuntime>,
    ) -> HttpResponse {
        let raw_path = http_req.path.as_str();
        let path = raw_path.split('?').next().unwrap_or(raw_path);
        if http_req.method == "GET" && (path == "/v1/models" || path == "/models") {
            Self::handle_models(&runtime)
        } else if http_req.method == "POST"
            && (path == "/v1/chat/completions" || path == "/chat/completions")
        {
            let traceparent = http_req.get_header("traceparent").map(str::to_owned);
            let parsed_req: Result<OpenAiChatRequest, _> = serde_json::from_slice(&http_req.body);
            match parsed_req {
                Ok(chat_req) => {
                    Self::handle_chat_completions_traced(chat_req, runtime, traceparent).await
                }
                Err(e) => {
                    let err = OpenAiErrorResponse {
                        error: OpenAiErrorDetails {
                            message: format!("Malformed JSON request body: {e}"),
                            r#type: "invalid_request_error".to_owned(),
                            param: None,
                            code: Some("invalid_payload".to_owned()),
                        },
                    };
                    let json = serde_json::to_string_pretty(&err).unwrap_or_default();
                    HttpResponse::error(400, json, Vec::new())
                }
            }
        } else {
            let err = OpenAiErrorResponse {
                error: OpenAiErrorDetails {
                    message: format!("Route not found: {} {}", http_req.method, http_req.path),
                    r#type: "invalid_request_error".to_owned(),
                    param: None,
                    code: Some("not_found".to_owned()),
                },
            };
            let json = serde_json::to_string_pretty(&err).unwrap_or_default();
            HttpResponse::error(404, json, Vec::new())
        }
    }

    /// Evaluates if the request includes tool declarations and generates or forwards a pass-through tool call.
    fn detect_tool_call_passthrough(req: &OpenAiChatRequest) -> Option<OpenAiToolCall> {
        let tools = req.tools.as_ref()?;
        if tools.is_empty() {
            return None;
        }

        // 1. Check if existing assistant messages contain tool_calls to pass through intact
        for m in &req.messages {
            if let Some(tcs) = &m.tool_calls
                && let Some(tc) = tcs.first()
            {
                return Some(tc.clone());
            }
        }

        // 2. Identify target tool from declared tools
        let last_user_text = req.messages.iter().rev().find_map(|m| {
            if m.role == "user" {
                m.content.as_ref().map(MessageContent::as_text)
            } else {
                None
            }
        })?;

        let lower = last_user_text.to_lowercase();
        let target_tool = tools
            .iter()
            .find(|t| {
                let t_name = t.function.name.to_lowercase();
                if lower.contains(&t_name) {
                    return true;
                }
                for part in t_name.split('_') {
                    if part.len() > 2 && lower.contains(part) {
                        return true;
                    }
                }
                lower.contains("tool") || lower.contains("call")
            })
            .or_else(|| {
                if tools.len() == 1 {
                    Some(&tools[0])
                } else {
                    None
                }
            })?;

        // 3. Extract dynamic parameters from prompt based on schema
        let arguments = Self::extract_arguments_from_prompt(
            &last_user_text,
            target_tool.function.parameters.as_ref(),
        );

        // 4. Generate unique dynamic call ID
        let call_id = format!(
            "call_{:x}",
            current_unix_timestamp() ^ u64::try_from(last_user_text.len()).unwrap_or(0)
        );

        Some(OpenAiToolCall {
            id: call_id,
            r#type: "function".to_owned(),
            function: OpenAiFunctionCall {
                name: target_tool.function.name.clone(),
                arguments,
            },
        })
    }

    fn raw_simulated_reply(model: &str, prompt_text: &str) -> &'static str {
        let model_lower = model.to_ascii_lowercase();
        if model_lower.contains("gemma4") || prompt_text.contains("<|think|>") {
            "<|think|>Let us analyze the algorithm step by step...<|/think|>Hello! How can I assist you today?"
        } else if model_lower.contains("qwen3") || prompt_text.contains("<think>") {
            "<think>Let us analyze the algorithm step by step...</think>Hello! How can I assist you today?"
        } else {
            "Hello! How can I assist you today?"
        }
    }

    /// Streams raw token chunks through [`ThinkingTagParser`], producing OpenAI SSE frames
    /// routed to `delta.reasoning_content` (`<|think|>` / `<think>`) and `delta.content`.
    #[must_use]
    pub fn format_streaming_chunks(
        request_id: &str,
        model: &str,
        created: u64,
        token_chunks: &[&str],
    ) -> Vec<OpenAiChunkResponse> {
        let mut parser = ThinkingTagParser::new();
        let mut deltas = Vec::new();
        for chunk in token_chunks {
            deltas.extend(parser.push_openai_deltas(chunk));
        }
        deltas.extend(parser.finish_openai_deltas());

        deltas
            .into_iter()
            .filter(|delta| {
                !delta.content.as_deref().is_none_or(str::is_empty)
                    || !delta.reasoning_content.as_deref().is_none_or(str::is_empty)
            })
            .map(|delta| OpenAiChunkResponse {
                id: request_id.to_owned(),
                object: "chat.completion.chunk".to_owned(),
                created,
                model: model.to_owned(),
                choices: vec![OpenAiChunkChoice {
                    index: 0,
                    delta,
                    finish_reason: None,
                }],
            })
            .collect()
    }

    fn extract_arguments_from_prompt(
        prompt: &str,
        parameters: Option<&serde_json::Value>,
    ) -> String {
        let mut map = serde_json::Map::new();

        if let Some(params) = parameters
            && let Some(props) = params
                .get("properties")
                .and_then(serde_json::Value::as_object)
        {
            for (prop_name, prop_schema) in props {
                let prop_type = prop_schema
                    .get("type")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("string");
                let val = Self::extract_property_value(prompt, prop_name, prop_type);
                map.insert(prop_name.clone(), val);
            }
        }

        if map.is_empty() {
            let val = Self::extract_property_value(prompt, "query", "string");
            map.insert("query".to_owned(), val);
        }

        serde_json::to_string(&map).unwrap_or_else(|_| "{}".to_owned())
    }

    fn extract_property_value(prompt: &str, prop_name: &str, prop_type: &str) -> serde_json::Value {
        let clean_prompt = prompt.trim().trim_end_matches(['?', '.', '!', ';']);

        let prepositions = ["in ", "for ", "at ", "of ", "to ", "about ", "query "];
        for prep in prepositions {
            if let Some(idx) = clean_prompt.to_lowercase().rfind(prep) {
                let candidate = clean_prompt[idx + prep.len()..]
                    .trim()
                    .trim_matches(['"', '\'', '`']);
                if !candidate.is_empty() {
                    return if prop_type == "integer" || prop_type == "number" {
                        candidate.parse::<i64>().map_or_else(
                            |_| serde_json::Value::String(candidate.to_owned()),
                            |n| serde_json::Value::Number(n.into()),
                        )
                    } else {
                        serde_json::Value::String(candidate.to_owned())
                    };
                }
            }
        }

        let prop_lower = prop_name.to_lowercase();
        if let Some(idx) = clean_prompt.to_lowercase().find(&prop_lower) {
            let after = clean_prompt[idx + prop_lower.len()..]
                .trim()
                .trim_start_matches([':', '='])
                .trim();
            if !after.is_empty() {
                let first_word = after.split_whitespace().next().unwrap_or(after);
                return serde_json::Value::String(first_word.to_owned());
            }
        }

        serde_json::Value::String(clean_prompt.to_owned())
    }
}

/// Audits a generated OpenAI `tool_calls` or Claude `tool_use` invocation against
/// `policy` prior to returning the payload to callers (`#38`).
///
/// Enforces fail-closed validation on invalid policy bounds, empty tool names,
/// malformed `arguments_json`, and canonicalized cyber/injection hazard signatures.
///
/// # Errors
/// Returns [`ManagerError::Protocol`] when `policy`, `tool_name`, or `arguments_json` is malformed.
pub fn audit_generated_tool_call(
    policy: &GuardrailPolicy,
    tool_name: &str,
    arguments_json: &str,
) -> Result<GuardrailEvaluateResponse, ManagerError> {
    if policy.model_id.trim().is_empty() {
        return Err(ManagerError::Protocol(
            "guardrail policy model_id must not be empty".to_owned(),
        ));
    }
    if !policy.risk_threshold.is_finite()
        || policy.risk_threshold <= 0.0
        || policy.risk_threshold > 1.0
    {
        return Err(ManagerError::Protocol(format!(
            "guardrail policy threshold must be finite and within (0.0, 1.0], got {}",
            policy.risk_threshold
        )));
    }
    if tool_name.trim().is_empty() {
        return Err(ManagerError::Protocol(
            "tool_name must not be empty for post-flight audit".to_owned(),
        ));
    }
    serde_json::from_str::<serde_json::Value>(arguments_json).map_err(|e| {
        ManagerError::Protocol(format!("malformed tool arguments JSON for audit: {e}"))
    })?;

    if !policy.audit_tool_calls {
        return Ok(GuardrailEvaluateResponse {
            safe: true,
            risk_score: 0.0,
            category: None,
            model: policy.model_id.clone(),
        });
    }

    let canonical_name = normalize_guardrail_input(tool_name);
    let canonical_args = normalize_guardrail_input(arguments_json);
    let combined = format!("{canonical_name} {canonical_args}");
    let (mut risk_score, mut category) = classify_canonical_security_text("", Some(&combined));

    if category.is_none() {
        let name_lower = canonical_name.to_ascii_lowercase();
        let args_lower = canonical_args.to_ascii_lowercase();
        let is_shell_tool = name_lower.contains("shell")
            || name_lower.contains("exec")
            || name_lower.contains("bash");
        let has_metachar = args_lower.contains(';')
            || args_lower.contains("&&")
            || args_lower.contains("||")
            || args_lower.contains('|')
            || args_lower.contains("$(");
        if is_shell_tool && has_metachar {
            risk_score = 0.97;
            category = Some("S14");
        } else {
            risk_score = 0.03;
        }
    }

    let category_owned = category.map(str::to_owned);
    let safe = risk_score < policy.risk_threshold && category_owned.is_none();
    Ok(GuardrailEvaluateResponse {
        safe,
        risk_score,
        category: category_owned,
        model: policy.model_id.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audit_generated_tool_call_fail_closed_unit() {
        let policy = GuardrailPolicy {
            model_id: "granite-guardian-3.1-2b".to_owned(),
            risk_threshold: 0.5,
            audit_tool_calls: true,
        };

        let safe_verdict = audit_generated_tool_call(&policy, "get_weather", r#"{"city":"Paris"}"#)
            .expect("benign tool call passes");
        assert!(safe_verdict.safe);
        assert!(safe_verdict.risk_score < 0.5);
        assert_eq!(safe_verdict.category, None);

        // Smuggled dangerous command in Unicode Tag + zero-width + non-breaking space characters
        let smuggled_args = "{\"cmd\":\"r\u{200B}m\u{00A0}\u{E0041}-rf /\"}";
        let unsafe_verdict = audit_generated_tool_call(&policy, "run_shell", smuggled_args)
            .expect("dangerous tool call evaluated");
        assert!(!unsafe_verdict.safe);
        assert!(unsafe_verdict.risk_score >= 0.5);
        assert_eq!(unsafe_verdict.category.as_deref(), Some("S14"));

        // Malformed JSON fails closed
        assert!(audit_generated_tool_call(&policy, "get_weather", "{bad json").is_err());
    }
}

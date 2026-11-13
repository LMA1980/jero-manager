//! External client protocol front-ends and adapters (US-23, US-24, Milestone M3).
//!
//! Provides network-isolated edge extensions exposing standard `OpenAI` (`GET /v1/models`,
//! `POST /v1/chat/completions`) and Anthropic Claude (`POST /v1/messages`) APIs.

#![allow(clippy::doc_markdown)]

pub mod claude;
pub mod openai;
pub mod server;
pub mod types;

use jero_tokenizer::{ThinkingDelta, ThinkingStreamExtractor};

pub use claude::ClaudeAdapter;
pub use openai::OpenAiAdapter;
pub use server::{MAX_REQUEST_HEADERS_BYTES, ProtocolServer, ProtocolServerConfig, ServerError};
pub use types::{
    ClaudeContent, ClaudeContentBlock, ClaudeErrorDetails, ClaudeErrorResponse, ClaudeMessage,
    ClaudeMessageRequest, ClaudeMessageResponse, ClaudeSystem, ClaudeSystemBlock,
    ClaudeThinkingConfig, ClaudeTool, ClaudeUsage, DEFAULT_MAX_REQUEST_BODY_BYTES,
    DEFAULT_SOCKET_TIMEOUT_SECS, HttpRequest, HttpResponse, HttpResponseBody, MessageContent,
    OpenAiChatRequest, OpenAiChatResponse, OpenAiChoice, OpenAiChunkChoice, OpenAiChunkDelta,
    OpenAiChunkResponse, OpenAiErrorDetails, OpenAiErrorResponse, OpenAiFunction,
    OpenAiFunctionCall, OpenAiFunctionCallChunk, OpenAiMessage, OpenAiResponseMessage, OpenAiStop,
    OpenAiTool, OpenAiToolCall, OpenAiToolCallChunk, OpenAiUsage,
};

/// Stateful reasoning tag parser routing `<|think|>...<|/think|>` (Gemma 4) and
/// `<think>...</think>` (`Qwen3` / Dory1.0) spans to protocol-specific reasoning deltas.
#[derive(Debug, Clone, Default)]
pub struct ThinkingTagParser {
    inner: ThinkingStreamExtractor,
}

impl ThinkingTagParser {
    /// Creates a new [`ThinkingTagParser`].
    #[must_use]
    pub const fn new() -> Self {
        Self {
            inner: ThinkingStreamExtractor::new(),
        }
    }

    /// Pushes a streamed token or text chunk and yields structured [`ThinkingDelta`] segments.
    pub fn push(&mut self, chunk: &str) -> Vec<ThinkingDelta> {
        self.inner.push(chunk)
    }

    /// Flushes any trailing buffered characters when the stream completes.
    pub fn finish(&mut self) -> Vec<ThinkingDelta> {
        self.inner.finish()
    }

    /// Splits a streamed chunk into OpenAI `delta.reasoning_content` / `delta.content` frames.
    pub fn push_openai_deltas(&mut self, chunk: &str) -> Vec<OpenAiChunkDelta> {
        deltas_to_openai(&self.inner.push(chunk))
    }

    /// Flushes remaining OpenAI deltas at end-of-stream.
    pub fn finish_openai_deltas(&mut self) -> Vec<OpenAiChunkDelta> {
        deltas_to_openai(&self.inner.finish())
    }

    /// Parses a complete raw model response into `(reasoning_content, visible_content)`.
    #[must_use]
    pub fn split_full_text(raw: &str) -> (Option<String>, String) {
        let mut parser = Self::new();
        let mut deltas = parser.push(raw);
        deltas.extend(parser.finish());

        let mut thinking = String::new();
        let mut content = String::new();
        for delta in deltas {
            match delta {
                ThinkingDelta::Thinking(t) => thinking.push_str(&t),
                ThinkingDelta::Content(c) => content.push_str(&c),
            }
        }
        let thinking_opt = if thinking.is_empty() {
            None
        } else {
            Some(thinking)
        };
        (thinking_opt, content)
    }
}

fn deltas_to_openai(deltas: &[ThinkingDelta]) -> Vec<OpenAiChunkDelta> {
    deltas
        .iter()
        .map(|delta| match delta {
            ThinkingDelta::Thinking(t) => OpenAiChunkDelta {
                role: None,
                content: None,
                reasoning_content: Some(t.clone()),
                tool_calls: None,
            },
            ThinkingDelta::Content(c) => OpenAiChunkDelta {
                role: None,
                content: Some(c.clone()),
                reasoning_content: None,
                tool_calls: None,
            },
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{ClaudeAdapter, OpenAiAdapter, ThinkingTagParser};

    #[test]
    fn test_gemma4_and_qwen3_thinking_deltas() {
        let gemma_chunks = [
            "<|thi",
            "nk|>",
            "step 1: ",
            "plan",
            "<|/th",
            "ink|>",
            "Final answer",
        ];
        let openai_gemma =
            OpenAiAdapter::format_streaming_chunks("req-gemma4", "gemma4-26b", 100, &gemma_chunks);
        let mut gemma_reasoning = String::new();
        let mut gemma_visible = String::new();
        for chunk in &openai_gemma {
            let delta = &chunk.choices[0].delta;
            if let Some(r) = &delta.reasoning_content {
                gemma_reasoning.push_str(r);
                let json = serde_json::to_string(chunk).expect("serialize openai chunk");
                assert!(json.contains("\"reasoning_content\""));
                assert!(!json.contains("\"content\""));
            }
            if let Some(c) = &delta.content {
                gemma_visible.push_str(c);
            }
        }
        assert_eq!(gemma_reasoning, "step 1: plan");
        assert_eq!(gemma_visible, "Final answer");
        assert!(!gemma_reasoning.contains("<|think|>") && !gemma_visible.contains("<|/think|>"));

        let qwen_chunks = ["<thi", "nk>", "reason", "</th", "ink>", "Result"];
        let openai_qwen =
            OpenAiAdapter::format_streaming_chunks("req-qwen3", "qwen3-8b", 101, &qwen_chunks);
        let qwen_reasoning: String = openai_qwen
            .iter()
            .filter_map(|c| c.choices[0].delta.reasoning_content.as_deref())
            .collect();
        let qwen_visible: String = openai_qwen
            .iter()
            .filter_map(|c| c.choices[0].delta.content.as_deref())
            .collect();
        assert_eq!(qwen_reasoning, "reason");
        assert_eq!(qwen_visible, "Result");

        let claude_gemma = ClaudeAdapter::format_thinking_stream_events(&gemma_chunks);
        let claude_think: String = claude_gemma
            .iter()
            .filter(|ev| ev["type"] == "thinking_delta")
            .filter_map(|ev| ev["thinking"].as_str())
            .collect();
        let claude_text: String = claude_gemma
            .iter()
            .filter(|ev| ev["type"] == "text_delta")
            .filter_map(|ev| ev["text"].as_str())
            .collect();
        assert_eq!(claude_think, "step 1: plan");
        assert_eq!(claude_text, "Final answer");

        let claude_qwen = ClaudeAdapter::format_thinking_stream_events(&qwen_chunks);
        assert!(claude_qwen.iter().any(|ev| ev["type"] == "thinking_delta"));
        assert!(claude_qwen.iter().any(|ev| ev["type"] == "text_delta"));

        let (t1, c1) = ThinkingTagParser::split_full_text("<|think|>abc<|/think|>xyz");
        assert_eq!(t1.as_deref(), Some("abc"));
        assert_eq!(c1, "xyz");

        let (t2, c2) = ThinkingTagParser::split_full_text("<think>abc</think>xyz");
        assert_eq!(t2.as_deref(), Some("abc"));
        assert_eq!(c2, "xyz");

        let (t3, c3) = ThinkingTagParser::split_full_text("Hello! How can I assist you today?");
        assert_eq!(t3, None);
        assert_eq!(c3, "Hello! How can I assist you today?");
    }
}

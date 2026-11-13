//! Architecture Recipe Capability Filter & Tokenizer Alignment (US-20, ADR-017, ADR-023, ADR-024, ADR-025).
//!
//! Validates model architecture and minor algorithmic revisions (`architecture_id`, `recipe_version`)
//! upfront during registry search, model inspection, and load time. Guarantees that control-plane
//! prompt preparation and chat templates are strictly bound to the active execution recipe.

use std::collections::HashMap;
use std::sync::{LazyLock, RwLock};

use jero_core::K3Error;
use jero_tokenizer::{ChatTemplate, SpecialTokens, Tokenizer};

use super::cas::ModelManifest;

/// Information describing a compiled architecture recipe and its capabilities.
#[derive(Debug, Clone)]
pub struct RecipeDescriptor {
    /// Exact architecture identifier (e.g. "synthetic", "qwen3", "llama").
    pub architecture_id: &'static str,
    /// Verified recipe revision string (e.g. "1.0.0").
    pub recipe_version: &'static str,
    /// List of supported quantization formats (e.g. `["Q4_K_M", "F32", "F16", "BF16"]`).
    pub supported_quants: &'static [&'static str],
    /// Target execution layouts (e.g. `["native", "ime_tile_8x16"]`).
    pub target_layouts: &'static [&'static str],
    /// Standard chat template bound to this recipe version.
    pub default_chat_template: Option<ChatTemplate>,
    /// Special stop tokens bound to this recipe version.
    pub special_tokens: Option<SpecialTokens>,
}

static BUILTIN_RECIPES: LazyLock<RwLock<HashMap<String, RecipeDescriptor>>> = LazyLock::new(|| {
    let mut map = HashMap::new();

    // 1. Reference synthetic recipe (ADR-023)
    let synthetic_template = ChatTemplate::new(Some(
        "<|im_start|>user\n{{prompt}}<|im_end|>\n<|im_start|>assistant\n".to_owned(),
    ));
    let synthetic_stop = SpecialTokens {
        stop_tokens: vec![0, 2],
        ..Default::default()
    };

    map.insert(
        "synthetic@1.0.0".to_owned(),
        RecipeDescriptor {
            architecture_id: "synthetic",
            recipe_version: "1.0.0",
            supported_quants: &["Q4_K_M", "F32", "F16", "BF16"],
            target_layouts: &["native", "ime_tile_8x16"],
            default_chat_template: Some(synthetic_template),
            special_tokens: Some(synthetic_stop),
        },
    );

    // 2. Qwen3 base recipe (ADR-023)
    let qwen3_template = ChatTemplate::new(Some(
        "<|im_start|>user\n{{prompt}}<|im_end|>\n<|im_start|>assistant\n".to_owned(),
    ));
    let qwen3_stop = SpecialTokens {
        stop_tokens: vec![151_643, 151_645], // <|endoftext|>, <|im_end|>
        eot_token_id: Some(151_645),
        ..Default::default()
    };

    map.insert(
        "qwen3@1.0.0".to_owned(),
        RecipeDescriptor {
            architecture_id: "qwen3",
            recipe_version: "1.0.0",
            supported_quants: &["Q4_K_M", "F32", "F16", "BF16"],
            target_layouts: &["native", "ime_tile_8x16"],
            default_chat_template: Some(qwen3_template),
            special_tokens: Some(qwen3_stop),
        },
    );

    // 3. Llama base recipe (ADR-023)
    let llama_template = ChatTemplate::new(Some(
        "<|start_header_id|>user<|end_header_id|>\n\n{{prompt}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n".to_owned(),
    ));
    let llama_stop = SpecialTokens {
        stop_tokens: vec![128_001, 128_009], // <|end_of_text|>, <|eot_id|>
        eot_token_id: Some(128_009),
        ..Default::default()
    };

    map.insert(
        "llama@1.0.0".to_owned(),
        RecipeDescriptor {
            architecture_id: "llama",
            recipe_version: "1.0.0",
            supported_quants: &["Q4_K_M", "Q8_0", "Q6_K", "F32", "F16", "BF16"],
            target_layouts: &["native", "ime_tile_8x16"],
            default_chat_template: Some(llama_template),
            special_tokens: Some(llama_stop),
        },
    );

    // 4. Mistral base recipe (ADR-023)
    let mistral_template = ChatTemplate::new(Some("[INST] {{prompt}} [/INST]".to_owned()));
    let mistral_stop = SpecialTokens {
        stop_tokens: vec![2], // </s>
        eot_token_id: Some(2),
        ..Default::default()
    };

    map.insert(
        "mistral@1.0.0".to_owned(),
        RecipeDescriptor {
            architecture_id: "mistral",
            recipe_version: "1.0.0",
            supported_quants: &["Q4_K_M", "Q8_0", "Q6_K", "F32", "F16", "BF16"],
            target_layouts: &["native", "ime_tile_8x16"],
            default_chat_template: Some(mistral_template.clone()),
            special_tokens: Some(mistral_stop.clone()),
        },
    );

    // 5. Mixtral Sparse MoE recipe (ADR-023)
    map.insert(
        "mixtral@1.0.0".to_owned(),
        RecipeDescriptor {
            architecture_id: "mixtral",
            recipe_version: "1.0.0",
            supported_quants: &["Q4_0", "Q4_K_M", "Q8_0", "Q6_K", "F32", "F16", "BF16"],
            target_layouts: &["native", "ime_tile_8x16"],
            default_chat_template: Some(mistral_template),
            special_tokens: Some(mistral_stop),
        },
    );

    // 6. Phi-3 base recipe (ADR-023)
    let phi3_template = ChatTemplate::new(Some(
        "<|user|>\n{{prompt}}<|end|>\n<|assistant|>\n".to_owned(),
    ));
    let phi3_stop = SpecialTokens {
        stop_tokens: vec![32000, 32007], // <|endoftext|>, <|end|>
        eot_token_id: Some(32007),
        ..Default::default()
    };

    map.insert(
        "phi3@1.0.0".to_owned(),
        RecipeDescriptor {
            architecture_id: "phi3",
            recipe_version: "1.0.0",
            supported_quants: &["Q4_K_M", "Q8_0", "Q6_K", "F32", "F16", "BF16"],
            target_layouts: &["native", "ime_tile_8x16"],
            default_chat_template: Some(phi3_template),
            special_tokens: Some(phi3_stop),
        },
    );

    // 7. BitNet base recipe (ADR-023, ADR-024, ADR-037)
    let bitnet_template = ChatTemplate::new(Some(
        "<|im_start|>user\n{{prompt}}<|im_end|>\n<|im_start|>assistant\n".to_owned(),
    ));
    let bitnet_stop = SpecialTokens {
        stop_tokens: vec![2, 32000],
        eot_token_id: Some(2),
        ..Default::default()
    };

    map.insert(
        "bitnet@1.0.0".to_owned(),
        RecipeDescriptor {
            architecture_id: "bitnet",
            recipe_version: "1.0.0",
            supported_quants: &["TQ1_0", "TQ2_0", "Q8_0", "F16", "BF16"],
            target_layouts: &["native", "ime_tile_8x16"],
            default_chat_template: Some(bitnet_template.clone()),
            special_tokens: Some(bitnet_stop.clone()),
        },
    );

    map.insert(
        "bitnet-b1.58@1.0.0".to_owned(),
        RecipeDescriptor {
            architecture_id: "bitnet-b1.58",
            recipe_version: "1.0.0",
            supported_quants: &["TQ1_0", "TQ2_0", "Q8_0", "F16", "BF16"],
            target_layouts: &["native", "ime_tile_8x16"],
            default_chat_template: Some(bitnet_template.clone()),
            special_tokens: Some(bitnet_stop.clone()),
        },
    );

    map.insert(
        "bitnet-25@1.0.0".to_owned(),
        RecipeDescriptor {
            architecture_id: "bitnet-25",
            recipe_version: "1.0.0",
            supported_quants: &["TQ1_0", "TQ2_0", "Q8_0", "F16", "BF16"],
            target_layouts: &["native", "ime_tile_8x16"],
            default_chat_template: Some(bitnet_template),
            special_tokens: Some(bitnet_stop),
        },
    );

    // 8. NVIDIA Nemotron-Labs Dory1.0 looped latent-recurrent nGPT recipe ("dory")
    let dory_template = ChatTemplate::new(Some(
        "<s><|im_start|>user\n{{prompt}}<|im_end|>\n<|im_start|>assistant\n<think>\n".to_owned(),
    ));
    let dory_stop = SpecialTokens {
        bos_token_id: Some(1),
        eos_token_id: Some(2),
        eot_token_id: Some(2),
        stop_tokens: vec![2],
        stop_sequences: vec!["<|im_end|>".to_owned()],
        ..Default::default()
    };

    map.insert(
        "dory@1.0.0".to_owned(),
        RecipeDescriptor {
            architecture_id: "dory",
            recipe_version: "1.0.0",
            supported_quants: &[
                "Q4_0", "Q4_K_M", "Q5_K", "Q6_K", "Q8_0", "F32", "F16", "BF16",
            ],
            target_layouts: &["native", "ime_tile_8x16"],
            default_chat_template: Some(dory_template),
            special_tokens: Some(dory_stop),
        },
    );

    // 9. Google Gemma 2, Gemma 3 & Gemma 4 family recipes ("gemma2", "gemma3", "gemma4")
    let gemma_template = ChatTemplate::new(Some(
        "<start_of_turn>user\n{{prompt}}<end_of_turn>\n<start_of_turn>model\n".to_owned(),
    ));
    let gemma_stop = SpecialTokens {
        bos_token_id: Some(2),
        eos_token_id: Some(1),
        eot_token_id: Some(107),
        add_bos_token: true,
        stop_tokens: vec![1, 107],
        stop_sequences: vec!["<end_of_turn>".to_owned(), "<eos>".to_owned()],
        ..Default::default()
    };

    for arch in ["gemma2", "gemma3", "gemma4"] {
        map.insert(
            format!("{arch}@1.0.0"),
            RecipeDescriptor {
                architecture_id: arch,
                recipe_version: "1.0.0",
                supported_quants: &[
                    "Q4_0", "Q4_K_M", "Q5_K", "Q6_K", "Q8_0", "F32", "F16", "BF16",
                ],
                target_layouts: &["native", "ime_tile_8x16"],
                default_chat_template: Some(gemma_template.clone()),
                special_tokens: Some(gemma_stop.clone()),
            },
        );
    }

    // 10. Single-pass zero-KV bidirectional encoder classifiers ("modernbert", "bert", "deberta-v3")
    for arch in ["modernbert", "bert", "deberta-v3"] {
        map.insert(
            format!("{arch}@1.0.0"),
            RecipeDescriptor {
                architecture_id: arch,
                recipe_version: "1.0.0",
                supported_quants: &[
                    "Q4_0", "Q4_K_M", "Q5_K", "Q6_K", "Q8_0", "F32", "F16", "BF16",
                ],
                target_layouts: &["native", "ime_tile_8x16"],
                default_chat_template: None,
                special_tokens: None,
            },
        );
    }

    // 11. Functional-token & action SLMs ("functiongemma", "octopus", "xlam") enforcing Q8_0 / native i8 floor
    let functiongemma_template = ChatTemplate::new(Some(
        "<start_of_turn>user\n{{prompt}}<end_of_turn>\n<start_of_turn>model\n<start_function_call>"
            .to_owned(),
    ));
    let functiongemma_tokens = SpecialTokens {
        bos_token_id: Some(2),
        eos_token_id: Some(1),
        eot_token_id: Some(107),
        add_bos_token: true,
        stop_tokens: vec![1, 107],
        stop_sequences: vec![
            "<end_of_turn>".to_owned(),
            "<eos>".to_owned(),
            "<end_function_call>".to_owned(),
        ],
        ..Default::default()
    };
    map.insert(
        "functiongemma@1.0.0".to_owned(),
        RecipeDescriptor {
            architecture_id: "functiongemma",
            recipe_version: "1.0.0",
            supported_quants: &["Q8_0", "F16", "BF16", "F32"],
            target_layouts: &["native", "ime_tile_8x16"],
            default_chat_template: Some(functiongemma_template),
            special_tokens: Some(functiongemma_tokens),
        },
    );

    let octopus_template = ChatTemplate::new(Some(
        "Below is the query from the users, please choose the correct function and generate the parameters to call the function.\nQuery: {{prompt}}<nexa_0>\nResponse: ".to_owned(),
    ));
    let octopus_tokens = SpecialTokens {
        bos_token_id: Some(1),
        eos_token_id: Some(2),
        add_bos_token: true,
        stop_tokens: vec![2],
        stop_sequences: vec!["<nexa_end>".to_owned(), "<|im_end|>".to_owned()],
        ..Default::default()
    };
    map.insert(
        "octopus@1.0.0".to_owned(),
        RecipeDescriptor {
            architecture_id: "octopus",
            recipe_version: "1.0.0",
            supported_quants: &["Q8_0", "F16", "BF16", "F32"],
            target_layouts: &["native", "ime_tile_8x16"],
            default_chat_template: Some(octopus_template),
            special_tokens: Some(octopus_tokens),
        },
    );

    let xlam_template = ChatTemplate::new(Some(
        "<|im_start|>user\n{{prompt}}<|tool_call|><|im_end|>\n<|im_start|>assistant\n".to_owned(),
    ));
    let xlam_tokens = SpecialTokens {
        bos_token_id: Some(1),
        eos_token_id: Some(2),
        add_bos_token: true,
        stop_tokens: vec![2],
        stop_sequences: vec!["<|im_end|>".to_owned(), "</tool_call>".to_owned()],
        ..Default::default()
    };
    map.insert(
        "xlam@1.0.0".to_owned(),
        RecipeDescriptor {
            architecture_id: "xlam",
            recipe_version: "1.0.0",
            supported_quants: &["Q8_0", "F16", "BF16", "F32"],
            target_layouts: &["native", "ime_tile_8x16"],
            default_chat_template: Some(xlam_template),
            special_tokens: Some(xlam_tokens),
        },
    );

    RwLock::new(map)
});

/// Suitability and compatibility assessment for candidate models during search and inspection (US-20 Scenario 1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateSuitability {
    /// Whether the model architecture and recipe version can execute on this runtime build.
    pub executable: bool,
    /// Architecture identifier.
    pub architecture_id: String,
    /// Declared or resolved recipe version.
    pub recipe_version: String,
    /// Whether the model's quantization format is supported.
    pub quantization_supported: bool,
    /// Diagnostic explanation if the model is not executable.
    pub reason: Option<String>,
}

/// Aligned prompt preparation and tokenization environment bound to a verified recipe version (US-20 Scenario 3).
#[derive(Debug, Clone)]
pub struct AlignedPromptEnvironment {
    /// Architecture identifier.
    pub architecture_id: &'static str,
    /// Active recipe revision.
    pub recipe_version: &'static str,
    /// Bound chat template.
    pub chat_template: ChatTemplate,
    /// Bound stop sequence detector.
    pub special_tokens: SpecialTokens,
}

impl AlignedPromptEnvironment {
    /// Formats a user prompt using the bound chat template, tokenizes with `tokenizer`,
    /// and validates context window bounds (`n_ctx`).
    ///
    /// # Errors
    /// Returns [`K3Error::ContextLengthExceeded`] or [`K3Error::CapabilityMismatch`].
    pub fn format_and_tokenize(
        &self,
        tokenizer: &dyn Tokenizer,
        prompt: &str,
        n_ctx: usize,
    ) -> Result<Vec<u32>, K3Error> {
        let formatted = self.chat_template.format_prompt(prompt);
        let tokens = tokenizer.encode(&formatted)?;
        if n_ctx == 0 || tokens.len() >= n_ctx {
            return Err(K3Error::ContextLengthExceeded {
                requested_tokens: tokens.len(),
                max_context_tokens: n_ctx,
            });
        }
        Ok(tokens)
    }

    /// Evaluates if `token_id` is an active stop token for this recipe version.
    #[must_use]
    pub fn is_stop_token(&self, token_id: u32) -> bool {
        self.special_tokens.is_stop(token_id)
    }
}

/// Architecture Recipe Capability Filter engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct CapabilityFilter;

impl CapabilityFilter {
    /// Returns a list of all recipe scope keys currently compiled into the runtime.
    ///
    /// # Panics
    /// Panics if the internal recipe registry lock is poisoned.
    #[must_use]
    pub fn supported_recipes() -> Vec<String> {
        let registry = BUILTIN_RECIPES.read().unwrap();
        let mut list: Vec<String> = registry.keys().cloned().collect();
        list.sort();
        list
    }

    /// Returns `true` if `(architecture_id, recipe_version)` is supported.
    ///
    /// # Panics
    /// Panics if the internal recipe registry lock is poisoned.
    #[must_use]
    pub fn is_supported(architecture_id: &str, recipe_version: &str) -> bool {
        let key = format!("{architecture_id}@{recipe_version}");
        let registry = BUILTIN_RECIPES.read().unwrap();
        registry.contains_key(&key)
    }

    /// Evaluates the compatibility of candidate model metadata during search or listing (US-20 Scenario 1).
    ///
    /// # Panics
    /// Panics if the internal recipe registry lock is poisoned.
    #[must_use]
    pub fn evaluate_candidate(manifest: &ModelManifest) -> CandidateSuitability {
        let key = format!("{}@{}", manifest.architecture_id, manifest.recipe_version);
        let registry = BUILTIN_RECIPES.read().unwrap();

        match registry.get(&key) {
            Some(desc) => {
                let quant_supported = desc
                    .supported_quants
                    .iter()
                    .any(|q| q.eq_ignore_ascii_case(&manifest.quantization));

                if !quant_supported {
                    let reason = if matches!(
                        desc.architecture_id,
                        "functiongemma" | "octopus" | "xlam"
                    ) {
                        format!(
                            "quantization format '{}' is not supported by recipe '{}': sub-1B action SLM requires minimum Q8_0 / native i8 IME precision floor (supported: {:?})",
                            manifest.quantization, key, desc.supported_quants
                        )
                    } else {
                        format!(
                            "quantization format '{}' is not supported by recipe '{}' (supported: {:?})",
                            manifest.quantization, key, desc.supported_quants
                        )
                    };
                    CandidateSuitability {
                        executable: false,
                        architecture_id: manifest.architecture_id.clone(),
                        recipe_version: manifest.recipe_version.clone(),
                        quantization_supported: false,
                        reason: Some(reason),
                    }
                } else if manifest.context_length == 0 {
                    CandidateSuitability {
                        executable: false,
                        architecture_id: manifest.architecture_id.clone(),
                        recipe_version: manifest.recipe_version.clone(),
                        quantization_supported: true,
                        reason: Some("model context length must be greater than 0".to_owned()),
                    }
                } else if manifest.architecture_id.eq_ignore_ascii_case("mixtral") {
                    if let Err(err) = jero_arch_mistral::evaluate_moe_memlock_admission(
                        manifest,
                        jero_arch_mistral::DEFAULT_MEMLOCK_LIMIT_BYTES,
                    ) {
                        CandidateSuitability {
                            executable: false,
                            architecture_id: manifest.architecture_id.clone(),
                            recipe_version: manifest.recipe_version.clone(),
                            quantization_supported: true,
                            reason: Some(format!(
                                "{err}: {}",
                                jero_arch_mistral::COMPACT_MOE_ADVISORY
                            )),
                        }
                    } else {
                        CandidateSuitability {
                            executable: true,
                            architecture_id: manifest.architecture_id.clone(),
                            recipe_version: manifest.recipe_version.clone(),
                            quantization_supported: true,
                            reason: None,
                        }
                    }
                } else {
                    CandidateSuitability {
                        executable: true,
                        architecture_id: manifest.architecture_id.clone(),
                        recipe_version: manifest.recipe_version.clone(),
                        quantization_supported: true,
                        reason: None,
                    }
                }
            }
            None => CandidateSuitability {
                executable: false,
                architecture_id: manifest.architecture_id.clone(),
                recipe_version: manifest.recipe_version.clone(),
                quantization_supported: false,
                reason: Some(format!(
                    "architecture '{}' with recipe version '{}' is not compiled into this runtime build (supported recipes: {:?})",
                    manifest.architecture_id,
                    manifest.recipe_version,
                    Self::supported_recipes()
                )),
            },
        }
    }

    /// Upfront load-time validation rejecting unsupported architectures or minor algorithmic
    /// revisions before allocating memory or launching worker processes (US-20 Scenario 2).
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] if the recipe is not compiled or parameters are incompatible.
    ///
    /// # Panics
    /// Panics if the internal recipe registry lock is poisoned.
    pub fn validate_model_manifest(manifest: &ModelManifest) -> Result<RecipeDescriptor, K3Error> {
        let key = format!("{}@{}", manifest.architecture_id, manifest.recipe_version);
        let registry = BUILTIN_RECIPES.read().unwrap();

        let desc = registry.get(&key).ok_or_else(|| K3Error::CapabilityMismatch {
            architecture: manifest.architecture_id.clone(),
            reason: format!(
                "architecture '{}' (recipe version '{}') is not compiled into this runtime build (supported recipes: {:?})",
                manifest.architecture_id,
                manifest.recipe_version,
                Self::supported_recipes()
            ),
        })?;

        // Format check
        let fmt_lower = manifest.format.to_ascii_lowercase();
        if fmt_lower != "gguf" && fmt_lower != "safetensors" && fmt_lower != "synthetic" {
            return Err(K3Error::CapabilityMismatch {
                architecture: manifest.architecture_id.clone(),
                reason: format!("unsupported model format '{}'", manifest.format),
            });
        }

        // Quantization check
        let quant_supported = desc
            .supported_quants
            .iter()
            .any(|q| q.eq_ignore_ascii_case(&manifest.quantization));

        if !quant_supported {
            let reason = if matches!(desc.architecture_id, "functiongemma" | "octopus" | "xlam") {
                format!(
                    "quantization format '{}' is not supported by recipe '{}': sub-1B action SLM requires minimum Q8_0 / native i8 IME precision floor (supported: {:?})",
                    manifest.quantization, key, desc.supported_quants
                )
            } else {
                format!(
                    "quantization format '{}' is not supported by recipe '{}' (supported: {:?})",
                    manifest.quantization, key, desc.supported_quants
                )
            };
            return Err(K3Error::CapabilityMismatch {
                architecture: manifest.architecture_id.clone(),
                reason,
            });
        }

        // Context length bound
        if manifest.context_length == 0 {
            return Err(K3Error::CapabilityMismatch {
                architecture: manifest.architecture_id.clone(),
                reason: format!(
                    "model context length for architecture '{}' must be greater than 0",
                    manifest.architecture_id
                ),
            });
        }

        // MoE memlock admission check for Mixtral
        if manifest.architecture_id.eq_ignore_ascii_case("mixtral") {
            jero_arch_mistral::evaluate_moe_memlock_admission(
                manifest,
                jero_arch_mistral::DEFAULT_MEMLOCK_LIMIT_BYTES,
            )?;
        }

        Ok(desc.clone())
    }

    /// Aligns the control-plane prompt preparation and chat template with the verified architecture recipe (US-20 Scenario 3).
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] if the recipe revision has no matching template.
    ///
    /// # Panics
    /// Panics if the internal recipe registry lock is poisoned.
    pub fn align_prompt_environment(
        architecture_id: &str,
        recipe_version: &str,
    ) -> Result<AlignedPromptEnvironment, K3Error> {
        let key = format!("{architecture_id}@{recipe_version}");
        let registry = BUILTIN_RECIPES.read().unwrap();

        let desc = registry
            .get(&key)
            .ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: architecture_id.to_owned(),
                reason: format!(
                    "cannot align tokenizer for unsupported recipe '{}' (supported: {:?})",
                    key,
                    Self::supported_recipes()
                ),
            })?;

        let chat_template = desc
            .default_chat_template
            .clone()
            .unwrap_or_else(|| ChatTemplate::new(Some("User: {{prompt}}\nAssistant: ".to_owned())));

        let special_tokens = desc.special_tokens.clone().unwrap_or_default();

        Ok(AlignedPromptEnvironment {
            architecture_id: desc.architecture_id,
            recipe_version: desc.recipe_version,
            chat_template,
            special_tokens,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mistral_and_mixtral_descriptors_registered() {
        let recipes = CapabilityFilter::supported_recipes();
        assert!(recipes.contains(&"mistral@1.0.0".to_owned()));
        assert!(recipes.contains(&"mixtral@1.0.0".to_owned()));

        assert!(CapabilityFilter::is_supported("mistral", "1.0.0"));
        assert!(CapabilityFilter::is_supported("mixtral", "1.0.0"));
    }

    #[test]
    fn test_mistral_chat_template_alignment() {
        let env = CapabilityFilter::align_prompt_environment("mistral", "1.0.0")
            .expect("mistral alignment succeeds");
        assert_eq!(env.architecture_id, "mistral");

        let formatted = env.chat_template.format_prompt("Tell me about K3");
        assert_eq!(formatted, "[INST] Tell me about K3 [/INST]");
        assert!(env.is_stop_token(2));
    }

    #[test]
    fn test_mixtral_admission_in_evaluate_candidate() {
        // Oversized Mixtral Q4_K_M with 32K context rejected with advisory
        let oversized = ModelManifest::new(
            "gguf",
            "mixtral",
            "1.0.0",
            "Q4_K_M",
            32_768,
            "sha256:oversized",
            28_990_000_000,
        );
        let suitability = CapabilityFilter::evaluate_candidate(&oversized);
        assert!(!suitability.executable);
        assert!(
            suitability
                .reason
                .as_ref()
                .unwrap()
                .contains("Mixtral-8x7B in Q4_K_M exceeds 28 GiB")
        );

        // Admitted Mixtral Q4_0 with 4K context
        let admitted = ModelManifest::new(
            "gguf",
            "mixtral",
            "1.0.0",
            "Q4_0",
            4096,
            "sha256:admitted",
            26_300_000_000,
        );
        let suitability_admitted = CapabilityFilter::evaluate_candidate(&admitted);
        assert!(suitability_admitted.executable);
        assert!(suitability_admitted.reason.is_none());
    }

    #[test]
    fn test_phi3_chat_template_alignment() {
        let env = CapabilityFilter::align_prompt_environment("phi3", "1.0.0")
            .expect("phi3 alignment succeeds");
        assert_eq!(env.architecture_id, "phi3");

        let formatted = env.chat_template.format_prompt("Tell me about Phi-3");
        assert_eq!(
            formatted,
            "<|user|>\nTell me about Phi-3<|end|>\n<|assistant|>\n"
        );
        assert!(env.is_stop_token(32000));
        assert!(env.is_stop_token(32007));
    }

    #[test]
    fn test_bitnet_filter_registration() {
        // 1. Registry presence
        let recipes = CapabilityFilter::supported_recipes();
        assert!(recipes.contains(&"bitnet@1.0.0".to_owned()));
        assert!(recipes.contains(&"bitnet-b1.58@1.0.0".to_owned()));
        assert!(recipes.contains(&"bitnet-25@1.0.0".to_owned()));

        assert!(CapabilityFilter::is_supported("bitnet", "1.0.0"));
        assert!(CapabilityFilter::is_supported("bitnet-b1.58", "1.0.0"));
        assert!(CapabilityFilter::is_supported("bitnet-25", "1.0.0"));
        assert!(!CapabilityFilter::is_supported("bitnet", "2.0.0"));

        // 2. Candidate evaluation - Admitted
        let manifest_tq1 = ModelManifest::new(
            "gguf",
            "bitnet-b1.58",
            "1.0.0",
            "TQ1_0",
            2048,
            "sha256:bitnet",
            2_000_000_000,
        );
        let suit_tq1 = CapabilityFilter::evaluate_candidate(&manifest_tq1);
        assert!(suit_tq1.executable);
        assert!(suit_tq1.quantization_supported);
        assert!(suit_tq1.reason.is_none());

        let manifest_tq2 = ModelManifest::new(
            "gguf",
            "bitnet-25",
            "1.0.0",
            "TQ2_0",
            4096,
            "sha256:bitnet25",
            2_500_000_000,
        );
        let suit_tq2 = CapabilityFilter::evaluate_candidate(&manifest_tq2);
        assert!(suit_tq2.executable);

        // 3. Candidate evaluation - Rejected unsupported quantization
        let manifest_q4 = ModelManifest::new(
            "gguf",
            "bitnet-b1.58",
            "1.0.0",
            "Q4_K_M",
            2048,
            "sha256:q4",
            2_000_000_000,
        );
        let suit_q4 = CapabilityFilter::evaluate_candidate(&manifest_q4);
        assert!(!suit_q4.executable);
        assert!(!suit_q4.quantization_supported);
        assert!(
            suit_q4
                .reason
                .unwrap()
                .contains("quantization format 'Q4_K_M' is not supported")
        );

        // 4. Load-time validation
        let desc =
            CapabilityFilter::validate_model_manifest(&manifest_tq1).expect("valid manifest");
        assert_eq!(desc.architecture_id, "bitnet-b1.58");
        assert_eq!(
            desc.supported_quants,
            &["TQ1_0", "TQ2_0", "Q8_0", "F16", "BF16"]
        );
        assert_eq!(desc.target_layouts, &["native", "ime_tile_8x16"]);

        // 5. Prompt environment alignment
        let env = CapabilityFilter::align_prompt_environment("bitnet-b1.58", "1.0.0")
            .expect("alignment succeeds");
        assert_eq!(env.architecture_id, "bitnet-b1.58");
        assert!(env.is_stop_token(2));
        assert!(env.is_stop_token(32000));
    }

    #[test]
    fn test_bitnet_capability_ids() {
        let expected_archs = ["bitnet", "bitnet-b1.58", "bitnet-25"];
        let expected_quants = ["TQ1_0", "TQ2_0", "Q8_0", "F16", "BF16"];

        for arch in expected_archs {
            assert!(
                CapabilityFilter::is_supported(arch, "1.0.0"),
                "CapabilityFilter must support {arch}@1.0.0"
            );

            let manifest = ModelManifest::new(
                "gguf",
                arch,
                "1.0.0",
                "TQ1_0",
                2048,
                "sha256:dummy",
                1_000_000,
            );
            let desc = CapabilityFilter::validate_model_manifest(&manifest)
                .unwrap_or_else(|_| panic!("manifest validation must succeed for {arch}"));
            assert_eq!(desc.architecture_id, arch);
            assert_eq!(desc.recipe_version, "1.0.0");
            assert_eq!(desc.supported_quants, expected_quants);
            assert_eq!(desc.target_layouts, &["native", "ime_tile_8x16"]);
        }
    }

    #[test]
    fn test_gemma_family_recipe_descriptors() {
        for arch in ["gemma2", "gemma3", "gemma4"] {
            assert!(
                CapabilityFilter::is_supported(arch, "1.0.0"),
                "CapabilityFilter must support {arch}@1.0.0"
            );

            let env = CapabilityFilter::align_prompt_environment(arch, "1.0.0")
                .unwrap_or_else(|_| panic!("prompt alignment must succeed for {arch}"));
            assert_eq!(env.architecture_id, arch);
            assert_eq!(
                env.chat_template.format_prompt("Explain RVV"),
                "<start_of_turn>user\nExplain RVV<end_of_turn>\n<start_of_turn>model\n"
            );
            assert!(env.is_stop_token(1));
            assert!(env.is_stop_token(107));
        }

        // Legacy gemma v1 remains intentionally uncompiled.
        assert!(!CapabilityFilter::is_supported("gemma", "1.0.0"));
    }

    #[test]
    fn test_functiongemma_rejects_q4_0() {
        assert!(CapabilityFilter::is_supported("functiongemma", "1.0.0"));
        let rejected = ModelManifest::new(
            "gguf",
            "functiongemma",
            "1.0.0",
            "Q4_0",
            2048,
            "sha256:fg-q4",
            270_000_000,
        );
        let suit = CapabilityFilter::evaluate_candidate(&rejected);
        assert!(!suit.executable);
        assert!(!suit.quantization_supported);
        let reason = suit.reason.expect("precision floor reason expected");
        assert!(reason.contains("Q8_0"));
        assert!(reason.contains("precision floor"));

        let err = CapabilityFilter::validate_model_manifest(&rejected)
            .expect_err("Q4_0 must fail load validation");
        assert!(err.to_string().contains("Q8_0"));
    }

    #[test]
    fn test_action_slm_q8_0_precision_enforcement() {
        for arch in ["modernbert", "bert", "deberta-v3"] {
            assert!(
                CapabilityFilter::is_supported(arch, "1.0.0"),
                "{arch}@1.0.0 must be registered"
            );
        }

        for arch in ["functiongemma", "octopus", "xlam"] {
            assert!(CapabilityFilter::is_supported(arch, "1.0.0"));
            for bad_quant in ["Q4_0", "Q4_K_M"] {
                let bad = ModelManifest::new(
                    "gguf",
                    arch,
                    "1.0.0",
                    bad_quant,
                    2048,
                    "sha256:bad",
                    500_000_000,
                );
                let suit = CapabilityFilter::evaluate_candidate(&bad);
                assert!(!suit.executable, "{arch} must reject {bad_quant}");
                assert!(!suit.quantization_supported);
            }
            for ok_quant in ["Q8_0", "F16", "BF16", "F32"] {
                let ok = ModelManifest::new(
                    "gguf",
                    arch,
                    "1.0.0",
                    ok_quant,
                    2048,
                    "sha256:ok",
                    500_000_000,
                );
                let suit = CapabilityFilter::evaluate_candidate(&ok);
                assert!(suit.executable, "{arch} must accept {ok_quant}");
                assert!(suit.quantization_supported);
            }
        }
    }
}

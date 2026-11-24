//! Content-Addressed Store (CAS) & Human-Browsable Model Registry (US-18, ADR-035, ADR-036).
//!
//! Organizes model weights by cryptographic SHA-256 digest (`blobs/sha256/<digest>`),
//! structured canonical manifests (`manifests/sha256/<manifest_digest>.json`),
//! and a human-browsable relative symlink alias tree (`models/<source>/<namespace>/<name>/<tag>`).

use std::collections::{BTreeMap, HashSet};
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use jero_core::{GgufFile, K3Error, QuantType, SafeTensorsFile, sanitize_and_normalize_features};

use super::sha256::Sha256;
use crate::dispatch::validate_target_subsystem;
use crate::persistence::{
    BlobReferenceRecord, CasCatalog, ModelAliasRecord, ModelBlobRecord, ModelManifestRecord,
};

static STAGING_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Maximum weight file byte size buffered into memory for full zero-copy header/tensor inspection.
pub const MAX_WEIGHT_INSPECTION_BYTES: u64 = 256 * 1024 * 1024;

/// Parsed weight inspection metadata from a `SafeTensors` or `GGUF` checkpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightInspectionReport {
    /// Canonical checkpoint format (`"safetensors"` or `"gguf"`).
    pub format: String,
    /// Parsed model architecture string from file metadata, if present.
    pub architecture: Option<String>,
    /// Number of tensors in the weight container.
    pub tensor_count: usize,
    /// Total tensor payload size in bytes.
    pub total_weight_bytes: u64,
    /// Extracted hidden/embedding dimension (`0` if omitted in metadata).
    pub embedding_dim: usize,
    /// Extracted context length (`0` if omitted in metadata).
    pub context_length: usize,
    /// Distinct tensor quantization / dtype identifiers found in the file.
    pub dtypes: Vec<String>,
}

fn quant_label(q: QuantType) -> String {
    format!("{q:?}")
}

/// Inspects an in-memory `SafeTensors` payload using [`SafeTensorsFile::from_bytes`].
///
/// # Errors
/// Returns [`K3Error::TensorBoundsError`] or [`K3Error::CapabilityMismatch`] if header framing,
/// tensor alignment, dtypes (`F32`/`F16`/`BF16`), or data offsets are invalid.
pub fn inspect_safetensors_bytes(bytes: &[u8]) -> Result<WeightInspectionReport, K3Error> {
    let st = SafeTensorsFile::from_bytes(bytes)?;
    let meta = st.metadata();
    let mut seen = HashSet::new();
    let mut dtypes = Vec::new();
    for tensor in st.tensors() {
        let label = quant_label(tensor.quant());
        if seen.insert(label.clone()) {
            dtypes.push(label);
        }
    }
    let arch = st
        .metadata_value("architecture")
        .map(str::to_owned)
        .or_else(|| {
            st.metadata_value("model_type")
                .or_else(|| st.metadata_value("general.architecture"))
                .map(str::to_owned)
        });
    let resolved_arch = arch.or_else(|| {
        let fallback = st.architecture();
        if fallback.is_empty() {
            None
        } else {
            Some(fallback.to_owned())
        }
    });

    Ok(WeightInspectionReport {
        format: "safetensors".to_owned(),
        architecture: resolved_arch,
        tensor_count: meta.tensor_count(),
        total_weight_bytes: u64::try_from(meta.total_weight_bytes()).unwrap_or(u64::MAX),
        embedding_dim: meta.embedding_dim(),
        context_length: meta.context_length(),
        dtypes,
    })
}

/// Inspects an in-memory `GGUF` payload using [`GgufFile::parse`].
///
/// # Errors
/// Returns [`K3Error`] if GGUF magic, version, metadata, or tensor descriptors are malformed.
pub fn inspect_gguf_bytes(bytes: &[u8]) -> Result<WeightInspectionReport, K3Error> {
    let gguf = GgufFile::parse(bytes)?;
    let meta = gguf.metadata();
    let mut seen = HashSet::new();
    let mut dtypes = Vec::new();
    for tensor in meta.tensors() {
        let label = quant_label(tensor.quant());
        if seen.insert(label.clone()) {
            dtypes.push(label);
        }
    }
    let arch = gguf.architecture();
    Ok(WeightInspectionReport {
        format: "gguf".to_owned(),
        architecture: if arch.is_empty() {
            None
        } else {
            Some(arch.to_owned())
        },
        tensor_count: meta.tensor_count(),
        total_weight_bytes: u64::try_from(meta.total_weight_bytes()).unwrap_or(u64::MAX),
        embedding_dim: meta.embedding_dim(),
        context_length: meta.context_length(),
        dtypes,
    })
}

/// Validates and inspects a weight checkpoint file (`"safetensors"` or `"gguf"`) on disk.
///
/// # Errors
/// Returns [`K3Error::CapabilityMismatch`] or [`K3Error::TensorBoundsError`] on I/O error,
/// oversized checkpoint above [`MAX_WEIGHT_INSPECTION_BYTES`], unsupported format, or invalid binary
/// headers/tensors.
pub fn inspect_weight_file(path: &Path, format: &str) -> Result<WeightInspectionReport, K3Error> {
    let fmt_lower = format.to_ascii_lowercase();
    let file_len = fs::metadata(path)
        .map_err(|e| K3Error::CapabilityMismatch {
            architecture: fmt_lower.clone(),
            reason: format!("failed to stat weight file '{}': {e}", path.display()),
        })?
        .len();
    if file_len > MAX_WEIGHT_INSPECTION_BYTES {
        return Err(K3Error::CapabilityMismatch {
            architecture: fmt_lower,
            reason: format!(
                "weight file '{}' ({file_len} bytes) exceeds inspection buffer limit ({MAX_WEIGHT_INSPECTION_BYTES} bytes)",
                path.display()
            ),
        });
    }
    let bytes = fs::read(path).map_err(|e| K3Error::CapabilityMismatch {
        architecture: fmt_lower.clone(),
        reason: format!("failed to read weight file '{}': {e}", path.display()),
    })?;
    match fmt_lower.as_str() {
        "safetensors" | "safetensor" => inspect_safetensors_bytes(&bytes),
        "gguf" => inspect_gguf_bytes(&bytes),
        other => Err(K3Error::CapabilityMismatch {
            architecture: other.to_owned(),
            reason: format!("unsupported model weight format '{format}' for binary inspection"),
        }),
    }
}

/// Default system store root when none is configured via CLI or environment (ADR-036).
pub const DEFAULT_STORE_ROOT: &str = "/var/lib/jero-runner";

/// Legacy Phase 1 system store root fallback when `/var/lib/jero-runner` does not exist yet.
pub const LEGACY_DEFAULT_STORE_ROOT: &str = "/var/lib/k3";

/// Primary environment variable for overriding the content-addressed store root (US-18 Scenario 2).
pub const ENV_STORE_ROOT: &str = "JERO_STORE_ROOT";

/// Legacy Phase 1 environment variable fallback for overriding the content-addressed store root.
pub const LEGACY_ENV_STORE_ROOT: &str = "K3_STORE_ROOT";

/// Primary environment variable for overriding the `DuckDB` catalog path.
pub const ENV_CATALOG_PATH: &str = "JERO_CATALOG_PATH";

/// Legacy Phase 1 environment variable fallback for overriding the `DuckDB` catalog path.
pub const LEGACY_ENV_CATALOG_PATH: &str = "K3_CATALOG_PATH";

fn resolve_store_root_with(
    get_env: impl Fn(&str) -> Option<String>,
    exists: impl Fn(&Path) -> bool,
) -> PathBuf {
    for key in [ENV_STORE_ROOT, LEGACY_ENV_STORE_ROOT] {
        if let Some(val) = get_env(key) {
            let trimmed = val.trim();
            if !trimmed.is_empty() {
                return PathBuf::from(trimmed);
            }
        }
    }
    let canonical = Path::new(DEFAULT_STORE_ROOT);
    if exists(canonical) {
        return canonical.to_path_buf();
    }
    let legacy = Path::new(LEGACY_DEFAULT_STORE_ROOT);
    if exists(legacy) {
        return legacy.to_path_buf();
    }
    canonical.to_path_buf()
}

/// Resolves the default content-addressed store root path.
///
/// Precedence: `JERO_STORE_ROOT` -> `K3_STORE_ROOT` -> `/var/lib/jero-runner`
/// (preferring an existing `/var/lib/k3` if `/var/lib/jero-runner` does not exist yet).
#[must_use]
pub fn resolve_default_store_root() -> PathBuf {
    resolve_store_root_with(|k| std::env::var(k).ok(), Path::exists)
}

fn resolve_catalog_path_with(
    store_root: &Path,
    get_env: impl Fn(&str) -> Option<String>,
) -> PathBuf {
    for key in [ENV_CATALOG_PATH, LEGACY_ENV_CATALOG_PATH] {
        if let Some(val) = get_env(key) {
            let trimmed = val.trim();
            if !trimmed.is_empty() {
                return PathBuf::from(trimmed);
            }
        }
    }
    store_root.join("catalog.duckdb")
}

/// Resolves the `DuckDB` catalog file path, checking `JERO_CATALOG_PATH` ahead of
/// `K3_CATALOG_PATH`, and defaulting to `<store_root>/catalog.duckdb`.
#[must_use]
pub fn resolve_catalog_path(store_root: &Path) -> PathBuf {
    resolve_catalog_path_with(store_root, |k| std::env::var(k).ok())
}

/// Maximum allowed character length for any single path segment (source, namespace, name, tag).
pub const MAX_PATH_COMPONENT_LEN: usize = 128;

/// Validates that a path component does not contain directory traversal sequences,
/// path separators, null bytes, or dangerous control characters (Security R3).
///
/// # Errors
/// Returns [`K3Error::CapabilityMismatch`] if validation fails.
pub fn validate_path_component(component: &str) -> Result<(), K3Error> {
    if component.is_empty() {
        return Err(K3Error::CapabilityMismatch {
            architecture: "cas_path".to_owned(),
            reason: "path component cannot be empty".to_owned(),
        });
    }

    if component.len() > MAX_PATH_COMPONENT_LEN {
        return Err(K3Error::CapabilityMismatch {
            architecture: "cas_path".to_owned(),
            reason: format!(
                "path component exceeds maximum length of {MAX_PATH_COMPONENT_LEN} characters"
            ),
        });
    }

    if component == "." || component == ".." {
        return Err(K3Error::CapabilityMismatch {
            architecture: "cas_path".to_owned(),
            reason: format!("path traversal sequence '{component}' is prohibited"),
        });
    }

    if component.starts_with('-') {
        return Err(K3Error::CapabilityMismatch {
            architecture: "cas_path".to_owned(),
            reason: format!("leading dash in path component '{component}' is prohibited"),
        });
    }

    for ch in component.chars() {
        if ch == '/' || ch == '\\' || ch == '\0' || ch.is_control() {
            return Err(K3Error::CapabilityMismatch {
                architecture: "cas_path".to_owned(),
                reason: format!("illegal character '{ch}' in path component '{component}'"),
            });
        }
        if !ch.is_ascii_alphanumeric() && ch != '_' && ch != '-' && ch != '.' {
            return Err(K3Error::CapabilityMismatch {
                architecture: "cas_path".to_owned(),
                reason: format!("unpermitted character '{ch}' in path component '{component}'"),
            });
        }
    }

    Ok(())
}

/// Parses and validates a 4-part alias identifier into `(source, namespace, name, tag)`.
///
/// # Errors
/// Returns [`K3Error::CapabilityMismatch`] if the format is invalid or any component fails validation.
pub fn parse_alias_id(alias_id: &str) -> Result<(String, String, String, String), K3Error> {
    let clean = alias_id.strip_prefix("models/").unwrap_or(alias_id);
    let parts: Vec<&str> = clean.split('/').collect();
    if parts.len() != 4 {
        return Err(K3Error::CapabilityMismatch {
            architecture: "cas_alias".to_owned(),
            reason: format!(
                "alias identifier must be in the format '<source>/<namespace>/<name>/<tag>'; got '{alias_id}'"
            ),
        });
    }

    validate_path_component(parts[0])?;
    validate_path_component(parts[1])?;
    validate_path_component(parts[2])?;
    validate_path_component(parts[3])?;

    Ok((
        parts[0].to_owned(),
        parts[1].to_owned(),
        parts[2].to_owned(),
        parts[3].to_owned(),
    ))
}

/// Validates that a string is a strict 64-character lowercase hexadecimal SHA-256 digest (`^[a-f0-9]{64}$`).
///
/// # Errors
/// Returns [`K3Error::CapabilityMismatch`] if length is not 64 or any character is outside `[a-f0-9]`.
pub fn validate_sha256_digest(digest: &str) -> Result<(), K3Error> {
    if digest.len() != 64
        || !digest
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(K3Error::CapabilityMismatch {
            architecture: "cas_digest".to_owned(),
            reason: format!(
                "invalid SHA-256 digest '{digest}': must be exactly 64 lowercase hexadecimal characters (^[a-f0-9]{{64}}$)"
            ),
        });
    }
    Ok(())
}

pub use jero_core::ModelManifest;

/// Extension trait for computing SHA-256 digest on [`ModelManifest`].
pub trait ModelManifestDigestExt {
    /// Computes the SHA-256 digest of this manifest's canonical JSON.
    ///
    /// # Errors
    /// Returns [`serde_json::Error`] if serialization fails.
    fn manifest_digest(&self) -> Result<String, serde_json::Error>;
}

impl ModelManifestDigestExt for ModelManifest {
    fn manifest_digest(&self) -> Result<String, serde_json::Error> {
        let json = self.to_canonical_json()?;
        Ok(Sha256::digest(json.as_bytes()))
    }
}

const VALID_MACHINE_FEATURE_TYPES: [&str; 22] = [
    "f16", "f32", "f64", "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "bool", "float16",
    "float32", "float64", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32",
];

/// Declarative system-event translation rule bound to a machine-model output signal.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MachineEventRule {
    /// Deterministic control-plane event identifier emitted when triggered.
    pub event_code: String,
    /// Target subsystem identifier: `"ActionRunner"`, `"MuConDe"`, or `"MuIsolation"`.
    pub target_subsystem: String,
    /// Optional hysteresis enter threshold override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enter_threshold: Option<f32>,
    /// Optional hysteresis exit threshold override (`exit_threshold < enter_threshold`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exit_threshold: Option<f32>,
    /// Optional minimum cooldown window in milliseconds override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cooldown_ms: Option<u64>,
}

impl MachineEventRule {
    /// Validates event rule invariants upfront.
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] if `event_code` is empty, `target_subsystem` is
    /// unsupported, or threshold bounds are non-finite / inverted.
    pub fn validate(&self) -> Result<(), K3Error> {
        if self.event_code.trim().is_empty() {
            return Err(K3Error::CapabilityMismatch {
                architecture: "machine_schema".to_owned(),
                reason: "MachineEventRule.event_code must not be empty".to_owned(),
            });
        }
        validate_target_subsystem(&self.target_subsystem)?;
        if let Some(enter) = self.enter_threshold
            && !enter.is_finite()
        {
            return Err(K3Error::CapabilityMismatch {
                architecture: "machine_schema".to_owned(),
                reason: format!("MachineEventRule.enter_threshold must be finite, got {enter}"),
            });
        }
        if let Some(exit) = self.exit_threshold
            && !exit.is_finite()
        {
            return Err(K3Error::CapabilityMismatch {
                architecture: "machine_schema".to_owned(),
                reason: format!("MachineEventRule.exit_threshold must be finite, got {exit}"),
            });
        }
        if let (Some(enter), Some(exit)) = (self.enter_threshold, self.exit_threshold)
            && enter <= exit
        {
            return Err(K3Error::CapabilityMismatch {
                architecture: "machine_schema".to_owned(),
                reason: format!(
                    "MachineEventRule.enter_threshold ({enter}) must exceed exit_threshold ({exit})"
                ),
            });
        }
        Ok(())
    }
}

/// Declarative input feature normalization and event-mapping schema bound to a machine model.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MachineFeatureSchema {
    /// Ordered input column names matching incoming feature vectors and Arrow `RecordBatch` columns.
    pub feature_names: Vec<String>,
    /// Ordered feature scalar dtype descriptors (e.g., `"f32"`, `"f64"`, `"i32"`, `"i8"`).
    pub feature_types: Vec<String>,
    /// Ordered per-column z-score mean values used for deterministic online normalization.
    pub zscore_mean: Vec<f32>,
    /// Ordered per-column strictly positive standard deviations used for online normalization.
    pub zscore_std: Vec<f32>,
    /// Deterministic mapping from signal names (e.g., `"anomaly"`, `"default"`) to event sink rules.
    #[serde(default)]
    pub event_mappings: BTreeMap<String, MachineEventRule>,
}

impl MachineFeatureSchema {
    /// Validates structural consistency and finiteness of all feature columns, normalization
    /// statistics, and event mapping rules.
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] on empty/duplicate feature names, mismatched column
    /// lengths, unsupported feature types, non-finite statistics, `zscore_std <= 0.0`, or invalid rules.
    pub fn validate(&self) -> Result<(), K3Error> {
        let n = self.feature_names.len();
        if n == 0 {
            return Err(K3Error::CapabilityMismatch {
                architecture: "machine_schema".to_owned(),
                reason: "MachineFeatureSchema.feature_names must not be empty".to_owned(),
            });
        }
        if self.feature_types.len() != n
            || self.zscore_mean.len() != n
            || self.zscore_std.len() != n
        {
            return Err(K3Error::CapabilityMismatch {
                architecture: "machine_schema".to_owned(),
                reason: format!(
                    "MachineFeatureSchema column lengths disagree: names={n}, types={}, mean={}, std={}",
                    self.feature_types.len(),
                    self.zscore_mean.len(),
                    self.zscore_std.len()
                ),
            });
        }

        let mut seen = HashSet::with_capacity(n);
        for idx in 0..n {
            let name = &self.feature_names[idx];
            if name.trim().is_empty() || !seen.insert(name.as_str()) {
                return Err(K3Error::CapabilityMismatch {
                    architecture: "machine_schema".to_owned(),
                    reason: format!(
                        "MachineFeatureSchema.feature_names[{idx}] ('{name}') is empty or duplicate"
                    ),
                });
            }
            let ftype = self.feature_types[idx].trim().to_ascii_lowercase();
            let valid_type =
                VALID_MACHINE_FEATURE_TYPES.contains(&ftype.as_str()) || ftype == "uint64";
            if !valid_type {
                return Err(K3Error::CapabilityMismatch {
                    architecture: "machine_schema".to_owned(),
                    reason: format!(
                        "MachineFeatureSchema.feature_types[{idx}] ('{}') is unsupported",
                        self.feature_types[idx]
                    ),
                });
            }
            let mean = self.zscore_mean[idx];
            let std = self.zscore_std[idx];
            if !mean.is_finite() || !std.is_finite() || std <= 0.0 {
                return Err(K3Error::CapabilityMismatch {
                    architecture: "machine_schema".to_owned(),
                    reason: format!(
                        "MachineFeatureSchema z-score stats invalid for column '{name}': mean={mean}, std={std}"
                    ),
                });
            }
        }

        for (signal, rule) in &self.event_mappings {
            if signal.trim().is_empty() {
                return Err(K3Error::CapabilityMismatch {
                    architecture: "machine_schema".to_owned(),
                    reason: "MachineFeatureSchema.event_mappings key must not be empty".to_owned(),
                });
            }
            rule.validate()?;
        }

        Ok(())
    }

    /// Sanitizes and z-score normalizes row-major `raw` feature values against this schema,
    /// delegating to [`jero_core::sanitize_and_normalize_features`].
    ///
    /// # Errors
    /// Returns [`K3Error`] if this schema is invalid, `raw` contains `NaN`/`±Inf`, `raw.len()` is
    /// empty or not divisible by `feature_names.len()`, or `clip_sigma` is invalid.
    pub fn normalize_features(&self, raw: &[f32], clip_sigma: f32) -> Result<Vec<f32>, K3Error> {
        self.validate()?;
        let mut out = vec![0.0_f32; raw.len()];
        sanitize_and_normalize_features(
            &mut out,
            raw,
            &self.zscore_mean,
            &self.zscore_std,
            clip_sigma,
        )?;
        Ok(out)
    }

    /// Serializes this schema into deterministic pretty JSON (`BTreeMap`-ordered keys).
    ///
    /// # Errors
    /// Returns [`serde_json::Error`] if serialization fails.
    pub fn to_canonical_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Computes a deterministic 64-bit FNV-1a schema digest suitable for
    /// `FeatureInputSlotHeader.schema_hash`.
    #[must_use]
    pub fn schema_hash(&self) -> u64 {
        let bytes = self.to_canonical_json().unwrap_or_default().into_bytes();
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for b in bytes {
            hash ^= u64::from(b);
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
        hash
    }

    /// Returns the preferred primary event rule (`"default"`, `"anomaly"`, or first rule).
    #[must_use]
    pub fn primary_event_rule(&self) -> Option<(&str, &MachineEventRule)> {
        self.event_mappings
            .get_key_value("default")
            .or_else(|| self.event_mappings.get_key_value("anomaly"))
            .or_else(|| self.event_mappings.iter().next())
            .map(|(k, v)| (k.as_str(), v))
    }
}

/// Extended manifest pairing canonical [`ModelManifest`] metadata with an optional
/// [`MachineFeatureSchema`] for non-LLM machine-model control-plane deployments.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MachineModelManifest {
    /// Underlying core model manifest fields (flattened identically at JSON top-level).
    #[serde(flatten)]
    pub core: ModelManifest,
    /// Optional validated feature normalization and event-sink schema for machine models.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub machine_schema: Option<MachineFeatureSchema>,
}

impl MachineModelManifest {
    /// Constructs a validated [`MachineModelManifest`].
    ///
    /// # Errors
    /// Returns [`K3Error`] if `machine_schema` is `Some` and fails validation.
    pub fn new(
        core: ModelManifest,
        machine_schema: Option<MachineFeatureSchema>,
    ) -> Result<Self, K3Error> {
        if let Some(ref schema) = machine_schema {
            schema.validate()?;
        }
        Ok(Self {
            core,
            machine_schema,
        })
    }

    /// Wraps a plain [`ModelManifest`] with `machine_schema: None`.
    #[must_use]
    pub const fn from_core(core: ModelManifest) -> Self {
        Self {
            core,
            machine_schema: None,
        }
    }

    /// Serializes this manifest to canonical JSON.
    ///
    /// Matches [`ModelManifest::to_canonical_json`] byte-for-byte when `machine_schema` is `None`,
    /// and includes `"machine_schema"` when `Some` so `manifest_digest` cryptographically binds it.
    ///
    /// # Errors
    /// Returns [`serde_json::Error`] if serialization fails.
    pub fn to_canonical_json(&self) -> Result<String, serde_json::Error> {
        if self.machine_schema.is_some() {
            serde_json::to_string_pretty(self)
        } else {
            self.core.to_canonical_json()
        }
    }
}

impl Deref for MachineModelManifest {
    type Target = ModelManifest;

    fn deref(&self) -> &Self::Target {
        &self.core
    }
}

impl From<ModelManifest> for MachineModelManifest {
    fn from(core: ModelManifest) -> Self {
        Self::from_core(core)
    }
}

impl ModelManifestDigestExt for MachineModelManifest {
    fn manifest_digest(&self) -> Result<String, serde_json::Error> {
        let json = self.to_canonical_json()?;
        Ok(Sha256::digest(json.as_bytes()))
    }
}

/// Resolved model alias information linking to manifest and weights on disk (US-18 Scenario 3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedAlias {
    /// Canonical alias identifier (e.g. "huggingface/Qwen/Qwen2.5-7B/latest").
    pub alias_id: String,
    /// Deserialized manifest metadata.
    pub manifest: ModelManifest,
    /// Absolute filesystem path to canonical manifest file.
    pub manifest_path: PathBuf,
    /// Absolute filesystem path to deduplicated weights blob.
    pub weight_path: PathBuf,
}

/// Content-Addressed Store (CAS) management instance.
#[derive(Debug, Clone)]
pub struct ContentAddressedStore {
    root: PathBuf,
    catalog: Arc<dyn CasCatalog>,
}

impl ContentAddressedStore {
    /// Opens or initializes a Content-Addressed Store under `root`.
    ///
    /// Creates the directory hierarchy (`blobs/sha256`, `manifests/sha256`, `models`, `staging`)
    /// if it does not already exist.
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] if directory creation fails.
    pub fn open(root: impl AsRef<Path>, catalog: Arc<dyn CasCatalog>) -> Result<Self, K3Error> {
        let root_buf = root.as_ref().to_path_buf();
        let blobs_dir = root_buf.join("blobs").join("sha256");
        let manifests_dir = root_buf.join("manifests").join("sha256");
        let models_dir = root_buf.join("models");
        let staging_dir = root_buf.join("staging");

        fs::create_dir_all(&blobs_dir).map_err(|e| K3Error::CapabilityMismatch {
            architecture: "cas_init".to_owned(),
            reason: format!("failed to create blobs dir '{}': {e}", blobs_dir.display()),
        })?;

        fs::create_dir_all(&manifests_dir).map_err(|e| K3Error::CapabilityMismatch {
            architecture: "cas_init".to_owned(),
            reason: format!(
                "failed to create manifests dir '{}': {e}",
                manifests_dir.display()
            ),
        })?;

        fs::create_dir_all(&models_dir).map_err(|e| K3Error::CapabilityMismatch {
            architecture: "cas_init".to_owned(),
            reason: format!(
                "failed to create models dir '{}': {e}",
                models_dir.display()
            ),
        })?;

        fs::create_dir_all(&staging_dir).map_err(|e| K3Error::CapabilityMismatch {
            architecture: "cas_init".to_owned(),
            reason: format!(
                "failed to create staging dir '{}': {e}",
                staging_dir.display()
            ),
        })?;

        Ok(Self {
            root: root_buf,
            catalog,
        })
    }

    /// Opens the store at the configured default location, respecting `JERO_STORE_ROOT`
    /// and legacy `K3_STORE_ROOT` overrides.
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] on initialization failure.
    pub fn open_default(catalog: Arc<dyn CasCatalog>) -> Result<Self, K3Error> {
        let path = resolve_default_store_root();
        Self::open(&path, catalog)
    }

    /// Returns the root path of the store.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Returns the blobs directory (`<root>/blobs/sha256`).
    #[must_use]
    pub fn blobs_dir(&self) -> PathBuf {
        self.root.join("blobs").join("sha256")
    }

    /// Returns the manifests directory (`<root>/manifests/sha256`).
    #[must_use]
    pub fn manifests_dir(&self) -> PathBuf {
        self.root.join("manifests").join("sha256")
    }

    /// Returns the models alias directory (`<root>/models`).
    #[must_use]
    pub fn models_dir(&self) -> PathBuf {
        self.root.join("models")
    }

    /// Returns the staging directory (`<root>/staging`).
    #[must_use]
    pub fn staging_dir(&self) -> PathBuf {
        self.root.join("staging")
    }

    /// Returns a reference to the embedded `DuckDB` catalog.
    #[must_use]
    pub fn catalog(&self) -> &Arc<dyn CasCatalog> {
        &self.catalog
    }

    /// Creates a unique temporary file in `<root>/staging/<uuid>.part`.
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] if file creation fails.
    pub fn create_staging_file(&self) -> Result<(PathBuf, File), K3Error> {
        let seq = STAGING_COUNTER.fetch_add(1, Ordering::Relaxed);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let pid = std::process::id();

        let mut hasher = Sha256::new();
        hasher.update(&nanos.to_le_bytes());
        hasher.update(&pid.to_le_bytes());
        hasher.update(&seq.to_le_bytes());
        let digest = hasher.finalize();

        let mut uuid_bytes = [0u8; 16];
        uuid_bytes.copy_from_slice(&digest[0..16]);
        uuid_bytes[6] = (uuid_bytes[6] & 0x0f) | 0x40; // UUID Version 4
        uuid_bytes[8] = (uuid_bytes[8] & 0x3f) | 0x80; // Variant RFC 4122

        let unique_name = format!(
            "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}.part",
            uuid_bytes[0],
            uuid_bytes[1],
            uuid_bytes[2],
            uuid_bytes[3],
            uuid_bytes[4],
            uuid_bytes[5],
            uuid_bytes[6],
            uuid_bytes[7],
            uuid_bytes[8],
            uuid_bytes[9],
            uuid_bytes[10],
            uuid_bytes[11],
            uuid_bytes[12],
            uuid_bytes[13],
            uuid_bytes[14],
            uuid_bytes[15]
        );
        let path = self.staging_dir().join(unique_name);
        let file = File::options()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|e| K3Error::CapabilityMismatch {
                architecture: "cas_staging".to_owned(),
                reason: format!("failed to create staging file '{}': {e}", path.display()),
            })?;

        // Best effort: set restrictive 0600 permissions on unix
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = fs::set_permissions(&path, fs::Permissions::from_mode(0o600));
        }

        Ok((path, file))
    }

    /// Verifies the cryptographic digest and format header of a staged file, then
    /// atomically promotes it into `blobs/sha256/<digest>` (US-18, US-19, ADR-040, Feature 26).
    ///
    /// If an identical blob already exists, the staging file is unlinked and the existing
    /// blob is deduplicated without duplicating bytes on storage (US-18 Scenario 1).
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] if digest verification or format checking fails.
    #[allow(clippy::too_many_lines)]
    pub fn promote_staged_blob(
        &self,
        staging_path: &Path,
        expected_digest: &str,
        format: &str,
    ) -> Result<PathBuf, K3Error> {
        validate_sha256_digest(expected_digest)?;

        // 1. Open staging file with read/write access and take exclusive non-blocking advisory lock
        let mut file = File::options()
            .read(true)
            .write(true)
            .open(staging_path)
            .map_err(|e| K3Error::CapabilityMismatch {
                architecture: "cas_promote".to_owned(),
                reason: format!(
                    "failed to open staging file '{}': {e}",
                    staging_path.display()
                ),
            })?;

        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            let fd = file.as_raw_fd();
            // SAFETY: `fd` is an open, valid file descriptor owned by promote_staged_blob.
            let lock_res = unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) };
            if lock_res != 0 {
                return Err(K3Error::CapabilityMismatch {
                    architecture: "cas_promote".to_owned(),
                    reason: format!(
                        "staging file '{}' is locked by another process (flock contention)",
                        staging_path.display()
                    ),
                });
            }
        }

        // 2. Stream and compute SHA-256 over open descriptor
        file.seek(SeekFrom::Start(0)).map_err(|e| {
            let _ = fs::remove_file(staging_path);
            K3Error::CapabilityMismatch {
                architecture: "cas_promote".to_owned(),
                reason: format!("failed to seek staging file: {e}"),
            }
        })?;

        let mut hasher = Sha256::new();
        let mut buffer = vec![0u8; 64 * 1024];
        let mut total_bytes = 0u64;
        loop {
            match file.read(&mut buffer) {
                Ok(0) => break,
                Ok(n) => {
                    hasher.update(&buffer[..n]);
                    total_bytes = total_bytes.saturating_add(n as u64);
                }
                Err(e) => {
                    let _ = fs::remove_file(staging_path);
                    return Err(K3Error::CapabilityMismatch {
                        architecture: "cas_integrity".to_owned(),
                        reason: format!(
                            "failed to read staging file '{}': {e}",
                            staging_path.display()
                        ),
                    });
                }
            }
        }
        let actual_digest = hasher.finalize_hex();

        if actual_digest != expected_digest {
            let _ = fs::remove_file(staging_path);
            return Err(K3Error::CapabilityMismatch {
                architecture: "cas_integrity".to_owned(),
                reason: format!(
                    "integrity verification failed for staging blob: expected digest '{expected_digest}', got '{actual_digest}'"
                ),
            });
        }

        // 3. Format header verification from open descriptor
        file.seek(SeekFrom::Start(0)).map_err(|e| {
            let _ = fs::remove_file(staging_path);
            K3Error::CapabilityMismatch {
                architecture: "cas_promote".to_owned(),
                reason: format!("failed to seek staging file: {e}"),
            }
        })?;

        if let Err(e) = Self::verify_format_header_from_file(&mut file, format, total_bytes) {
            let _ = fs::remove_file(staging_path);
            return Err(e);
        }

        // 4. Target blob destination and atomic promotion
        let final_path = self.blobs_dir().join(&actual_digest);
        let size_bytes = total_bytes;

        if final_path.exists() {
            // Deduplication: unlinked staging file and reuse existing blob
            let _ = fs::remove_file(staging_path);
        } else {
            // Atomic rename into blobs/sha256/<digest>
            fs::rename(staging_path, &final_path).map_err(|e| K3Error::CapabilityMismatch {
                architecture: "cas_promote".to_owned(),
                reason: format!(
                    "failed to atomically promote staging file to '{}': {e}",
                    final_path.display()
                ),
            })?;

            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let _ = fs::set_permissions(&final_path, fs::Permissions::from_mode(0o444));
            }
        }

        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            // SAFETY: `file.as_raw_fd()` is a valid open file descriptor.
            let _ = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_UN) };
        }
        drop(file);

        // 5. Register in DuckDB catalog if not already tracked
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let blob_rec = ModelBlobRecord {
            digest: actual_digest,
            size_bytes,
            blob_type: "raw_weights".to_owned(),
            created_at: now,
            last_accessed_at: now,
            ref_count: 0,
        };

        self.catalog
            .save_blob(&blob_rec)
            .map_err(|e| K3Error::CapabilityMismatch {
                architecture: "cas_catalog".to_owned(),
                reason: format!("failed to register blob in catalog: {e}"),
            })?;

        Ok(final_path)
    }

    /// Verifies format header directly from an open, locked file descriptor.
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] on invalid header format or bounds violation.
    #[allow(clippy::too_many_lines)]
    pub fn verify_format_header_from_file(
        file: &mut File,
        format: &str,
        file_len: u64,
    ) -> Result<(), K3Error> {
        let fmt_lower = format.to_ascii_lowercase();
        if fmt_lower == "gguf" {
            if file_len < 24 {
                return Err(K3Error::CapabilityMismatch {
                    architecture: "gguf".to_owned(),
                    reason: format!("file is truncated for GGUF: length {file_len} < 24"),
                });
            }
            let mut magic = [0u8; 4];
            file.read_exact(&mut magic)
                .map_err(|e| K3Error::CapabilityMismatch {
                    architecture: "gguf".to_owned(),
                    reason: format!("failed to read GGUF magic: {e}"),
                })?;
            if &magic != b"GGUF" {
                return Err(K3Error::CapabilityMismatch {
                    architecture: "gguf".to_owned(),
                    reason: format!("file does not contain valid GGUF magic (found: {magic:?})"),
                });
            }
            let mut ver_bytes = [0u8; 4];
            file.read_exact(&mut ver_bytes)
                .map_err(|e| K3Error::CapabilityMismatch {
                    architecture: "gguf".to_owned(),
                    reason: format!("failed to read GGUF version: {e}"),
                })?;
            let version = u32::from_le_bytes(ver_bytes);
            if !(1..=3).contains(&version) {
                return Err(K3Error::CapabilityMismatch {
                    architecture: "gguf".to_owned(),
                    reason: format!("unsupported GGUF version {version}; supported: 1, 2, 3"),
                });
            }
            let mut tc_bytes = [0u8; 8];
            file.read_exact(&mut tc_bytes)
                .map_err(|e| K3Error::CapabilityMismatch {
                    architecture: "gguf".to_owned(),
                    reason: format!("failed to read GGUF tensor_count: {e}"),
                })?;
            let tensor_count = u64::from_le_bytes(tc_bytes);
            if tensor_count > 65_536 {
                return Err(K3Error::CapabilityMismatch {
                    architecture: "gguf".to_owned(),
                    reason: format!(
                        "GGUF tensor_count {tensor_count} exceeds maximum sanity limit of 65536"
                    ),
                });
            }
            let mut kc_bytes = [0u8; 8];
            file.read_exact(&mut kc_bytes)
                .map_err(|e| K3Error::CapabilityMismatch {
                    architecture: "gguf".to_owned(),
                    reason: format!("failed to read GGUF kv_count: {e}"),
                })?;
            let kv_count = u64::from_le_bytes(kc_bytes);
            if kv_count > 65_536 {
                return Err(K3Error::CapabilityMismatch {
                    architecture: "gguf".to_owned(),
                    reason: format!(
                        "GGUF kv_count {kv_count} exceeds maximum sanity limit of 65536"
                    ),
                });
            }
        } else if fmt_lower == "safetensors" {
            if file_len < 8 {
                return Err(K3Error::CapabilityMismatch {
                    architecture: "safetensors".to_owned(),
                    reason: format!("file is truncated for SafeTensors: length {file_len} < 8"),
                });
            }
            let mut len_bytes = [0u8; 8];
            file.read_exact(&mut len_bytes)
                .map_err(|e| K3Error::CapabilityMismatch {
                    architecture: "safetensors".to_owned(),
                    reason: format!("failed to read SafeTensors header length: {e}"),
                })?;
            let header_len = u64::from_le_bytes(len_bytes);
            if header_len == 0 || header_len > 100 * 1024 * 1024 {
                return Err(K3Error::CapabilityMismatch {
                    architecture: "safetensors".to_owned(),
                    reason: format!("invalid SafeTensors header length: {header_len}"),
                });
            }
            let total_min_len = (8u64).saturating_add(header_len);
            if file_len < total_min_len {
                return Err(K3Error::CapabilityMismatch {
                    architecture: "safetensors".to_owned(),
                    reason: format!(
                        "file is truncated: requires at least {total_min_len} bytes for SafeTensors header, got {file_len}"
                    ),
                });
            }
            let header_len_usize =
                usize::try_from(header_len).map_err(|_| K3Error::CapabilityMismatch {
                    architecture: "safetensors".to_owned(),
                    reason: "SafeTensors header length exceeds pointer address space".to_owned(),
                })?;
            let mut json_buf = vec![0u8; header_len_usize];
            file.read_exact(&mut json_buf)
                .map_err(|e| K3Error::CapabilityMismatch {
                    architecture: "safetensors".to_owned(),
                    reason: format!("failed to read SafeTensors JSON header: {e}"),
                })?;
            let value: serde_json::Value =
                serde_json::from_slice(&json_buf).map_err(|e| K3Error::CapabilityMismatch {
                    architecture: "safetensors".to_owned(),
                    reason: format!("invalid SafeTensors JSON metadata: {e}"),
                })?;
            if let Some(map) = value.as_object() {
                for (name, tensor_val) in map {
                    if name == "__metadata__" {
                        continue;
                    }
                    if let Some(offsets) = tensor_val.get("data_offsets").and_then(|v| v.as_array())
                    {
                        if offsets.len() != 2 {
                            return Err(K3Error::CapabilityMismatch {
                                architecture: "safetensors".to_owned(),
                                reason: format!(
                                    "tensor '{name}' data_offsets must be a 2-element array"
                                ),
                            });
                        }
                        let begin =
                            offsets[0]
                                .as_u64()
                                .ok_or_else(|| K3Error::CapabilityMismatch {
                                    architecture: "safetensors".to_owned(),
                                    reason: format!(
                                        "tensor '{name}' begin offset is not a valid u64"
                                    ),
                                })?;
                        let end =
                            offsets[1]
                                .as_u64()
                                .ok_or_else(|| K3Error::CapabilityMismatch {
                                    architecture: "safetensors".to_owned(),
                                    reason: format!(
                                        "tensor '{name}' end offset is not a valid u64"
                                    ),
                                })?;
                        if begin > end {
                            return Err(K3Error::CapabilityMismatch {
                                architecture: "safetensors".to_owned(),
                                reason: format!(
                                    "tensor '{name}' inverted data_offsets: begin ({begin}) > end ({end})"
                                ),
                            });
                        }
                        let total_extent = (8u64)
                            .checked_add(header_len)
                            .and_then(|h| h.checked_add(end));
                        match total_extent {
                            Some(ext) if ext <= file_len => {}
                            _ => {
                                return Err(K3Error::CapabilityMismatch {
                                    architecture: "safetensors".to_owned(),
                                    reason: format!(
                                        "tensor '{name}' offset {end} exceeds file length {file_len}"
                                    ),
                                });
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Verifies the format header of a weight file (GGUF magic / `SafeTensors` header).
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] on invalid header format.
    pub fn verify_format_header(path: &Path, format: &str) -> Result<(), K3Error> {
        let mut file = File::open(path).map_err(|e| K3Error::CapabilityMismatch {
            architecture: "cas_header".to_owned(),
            reason: format!("failed to open file '{}': {e}", path.display()),
        })?;
        let file_len = file.metadata().map_or(0, |m| m.len());
        Self::verify_format_header_from_file(&mut file, format, file_len)
    }

    /// Inspects a staged or promoted model weight file (`safetensors` or `gguf`) and cross-checks
    /// its parsed tensor layout and manifest against [`super::filter::CapabilityFilter`].
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] or [`K3Error::TensorBoundsError`] if the checkpoint
    /// binary fails validation, architecture metadata conflicts, or the manifest recipe/quant/format
    /// is not admitted.
    pub fn validate_safetensors_weights_and_admit(
        path: &Path,
        manifest: &ModelManifest,
    ) -> Result<(WeightInspectionReport, super::filter::CandidateSuitability), K3Error> {
        let report = inspect_weight_file(path, &manifest.format)?;
        if let Some(ref parsed_arch) = report.architecture {
            let trimmed = parsed_arch.trim();
            if !trimmed.is_empty()
                && !trimmed.eq_ignore_ascii_case("unknown")
                && !trimmed.eq_ignore_ascii_case(&manifest.architecture_id)
                && !manifest.format.eq_ignore_ascii_case("safetensors")
            {
                return Err(K3Error::CapabilityMismatch {
                    architecture: manifest.architecture_id.clone(),
                    reason: format!(
                        "weight header architecture '{trimmed}' does not match manifest architecture '{}'",
                        manifest.architecture_id
                    ),
                });
            }
        }
        let _desc = super::filter::CapabilityFilter::validate_model_manifest(manifest)?;
        let suitability = super::filter::CapabilityFilter::evaluate_candidate(manifest);
        if !suitability.executable {
            return Err(K3Error::CapabilityMismatch {
                architecture: manifest.architecture_id.clone(),
                reason: suitability
                    .reason
                    .unwrap_or_else(|| "model manifest rejected by capability filter".to_owned()),
            });
        }
        Ok((report, suitability))
    }

    fn persist_manifest_json(
        &self,
        manifest: &ModelManifest,
        canonical_json: String,
    ) -> Result<ModelManifestRecord, K3Error> {
        let manifest_digest = Sha256::digest(canonical_json.as_bytes());
        let manifest_path = self.manifests_dir().join(format!("{manifest_digest}.json"));

        // Write manifest atomically via temporary file
        let temp_manifest = self.manifests_dir().join(format!("{manifest_digest}.tmp"));
        let mut f = File::create(&temp_manifest).map_err(|e| K3Error::CapabilityMismatch {
            architecture: "cas_manifest".to_owned(),
            reason: format!(
                "failed to create temp manifest file '{}': {e}",
                temp_manifest.display()
            ),
        })?;

        f.write_all(canonical_json.as_bytes())
            .and_then(|()| f.flush())
            .map_err(|e| K3Error::CapabilityMismatch {
                architecture: "cas_manifest".to_owned(),
                reason: format!("failed to write manifest: {e}"),
            })?;

        fs::rename(&temp_manifest, &manifest_path).map_err(|e| K3Error::CapabilityMismatch {
            architecture: "cas_manifest".to_owned(),
            reason: format!(
                "failed to atomically promote manifest to '{}': {e}",
                manifest_path.display()
            ),
        })?;

        let record = ModelManifestRecord {
            manifest_digest: manifest_digest.clone(),
            weight_digest: manifest.weight_digest.clone(),
            format: manifest.format.clone(),
            architecture_id: manifest.architecture_id.clone(),
            recipe_version: manifest.recipe_version.clone(),
            quantization: manifest.quantization.clone(),
            context_length: manifest.context_length,
            license: manifest.license.clone(),
            origin: manifest.origin.clone(),
            manifest_json: canonical_json,
            created_at: manifest.created_at,
        };

        self.catalog
            .save_manifest(&record)
            .map_err(|e| K3Error::CapabilityMismatch {
                architecture: "cas_catalog".to_owned(),
                reason: format!("failed to save manifest in DuckDB catalog: {e}"),
            })?;

        // Record granular reference in DuckDB
        let ref_rec = BlobReferenceRecord {
            ref_id: format!("manifest:{manifest_digest}"),
            blob_digest: manifest.weight_digest.clone(),
            ref_type: "manifest".to_owned(),
            ref_source: manifest_digest,
            created_at: manifest.created_at,
        };
        self.catalog
            .add_blob_reference(&ref_rec)
            .map_err(|e| K3Error::CapabilityMismatch {
                architecture: "cas_catalog".to_owned(),
                reason: format!("failed to record blob reference: {e}"),
            })?;

        Ok(record)
    }

    /// Registers a model manifest in the store and `DuckDB` catalog (US-18).
    ///
    /// Writes the canonical manifest to `<root>/manifests/sha256/<manifest_digest>.json`.
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] on serialization, disk write, or catalog failure.
    pub fn register_manifest(
        &self,
        manifest: &ModelManifest,
    ) -> Result<ModelManifestRecord, K3Error> {
        validate_sha256_digest(&manifest.weight_digest)?;

        let canonical_json =
            manifest
                .to_canonical_json()
                .map_err(|e| K3Error::CapabilityMismatch {
                    architecture: "cas_manifest".to_owned(),
                    reason: format!("failed to serialize manifest to canonical JSON: {e}"),
                })?;

        self.persist_manifest_json(manifest, canonical_json)
    }

    /// Registers a [`MachineModelManifest`] (with optional validated [`MachineFeatureSchema`])
    /// in the store and `DuckDB` catalog, populating both `manifest_json` and `machine_schema_json`.
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] if schema validation, serialization, disk write, or
    /// catalog persistence fails.
    pub fn register_machine_manifest(
        &self,
        manifest: &MachineModelManifest,
    ) -> Result<ModelManifestRecord, K3Error> {
        validate_sha256_digest(&manifest.core.weight_digest)?;
        if let Some(ref schema) = manifest.machine_schema {
            schema.validate()?;
        }

        let canonical_json =
            manifest
                .to_canonical_json()
                .map_err(|e| K3Error::CapabilityMismatch {
                    architecture: "cas_manifest".to_owned(),
                    reason: format!("failed to serialize machine manifest to canonical JSON: {e}"),
                })?;

        self.persist_manifest_json(&manifest.core, canonical_json)
    }

    /// Loads and validates a [`MachineModelManifest`] by SHA-256 `manifest_digest`.
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] if `manifest_digest` is invalid, missing, corrupt,
    /// or carries an invalid [`MachineFeatureSchema`].
    pub fn load_manifest_with_schema(
        &self,
        manifest_digest: &str,
    ) -> Result<MachineModelManifest, K3Error> {
        validate_sha256_digest(manifest_digest)?;
        let record = self
            .catalog
            .get_manifest(manifest_digest)
            .map_err(|e| K3Error::CapabilityMismatch {
                architecture: "cas_resolve".to_owned(),
                reason: format!("catalog query failed for manifest '{manifest_digest}': {e}"),
            })?
            .ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: "cas_resolve".to_owned(),
                reason: format!("manifest '{manifest_digest}' not found in catalog"),
            })?;

        let parsed: MachineModelManifest =
            serde_json::from_str(&record.manifest_json).map_err(|e| {
                K3Error::CapabilityMismatch {
                    architecture: "cas_resolve".to_owned(),
                    reason: format!("corrupt machine manifest JSON in catalog: {e}"),
                }
            })?;
        if let Some(ref schema) = parsed.machine_schema {
            schema.validate()?;
        }
        Ok(parsed)
    }

    /// Resolves and validates the optional [`MachineFeatureSchema`] for a `manifest_digest` or
    /// `<source>/<namespace>/<name>/<tag>` alias identifier.
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] if the digest/alias cannot be resolved or fails schema
    /// validation.
    pub fn get_machine_schema(
        &self,
        manifest_digest_or_alias: &str,
    ) -> Result<Option<MachineFeatureSchema>, K3Error> {
        let digest = if manifest_digest_or_alias.contains('/') {
            let (source, namespace, name, tag) = parse_alias_id(manifest_digest_or_alias)?;
            let alias = self
                .catalog
                .find_alias(&source, &namespace, &name, &tag)
                .map_err(|e| K3Error::CapabilityMismatch {
                    architecture: "cas_resolve".to_owned(),
                    reason: format!(
                        "catalog query failed for alias '{manifest_digest_or_alias}': {e}"
                    ),
                })?
                .ok_or_else(|| K3Error::CapabilityMismatch {
                    architecture: "cas_resolve".to_owned(),
                    reason: format!("model alias '{manifest_digest_or_alias}' is not registered"),
                })?;
            alias.manifest_digest
        } else {
            validate_sha256_digest(manifest_digest_or_alias)?;
            manifest_digest_or_alias.to_owned()
        };

        Ok(self.load_manifest_with_schema(&digest)?.machine_schema)
    }

    /// Creates a human-browsable alias directory in `models/<source>/<namespace>/<name>/<tag>`
    /// containing relative symlinks to the manifest and weight blob (US-18 Scenario 3).
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] if path validation, directory creation, symlinking,
    /// or catalog recording fails.
    #[allow(clippy::too_many_lines)]
    pub fn create_alias(
        &self,
        source: &str,
        namespace: &str,
        name: &str,
        tag: &str,
        manifest_digest: &str,
    ) -> Result<ModelAliasRecord, K3Error> {
        validate_path_component(source)?;
        validate_path_component(namespace)?;
        validate_path_component(name)?;
        validate_path_component(tag)?;
        validate_sha256_digest(manifest_digest)?;

        let manifest = self
            .catalog
            .get_manifest(manifest_digest)
            .map_err(|e| K3Error::CapabilityMismatch {
                architecture: "cas_alias".to_owned(),
                reason: format!("failed to lookup manifest '{manifest_digest}': {e}"),
            })?
            .ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: "cas_alias".to_owned(),
                reason: format!("manifest '{manifest_digest}' not found in catalog"),
            })?;

        let alias_id = format!("{source}/{namespace}/{name}/{tag}");
        let alias_dir = self
            .models_dir()
            .join(source)
            .join(namespace)
            .join(name)
            .join(tag);

        fs::create_dir_all(&alias_dir).map_err(|e| K3Error::CapabilityMismatch {
            architecture: "cas_alias".to_owned(),
            reason: format!(
                "failed to create alias directory '{}': {e}",
                alias_dir.display()
            ),
        })?;

        // Create relative symlinks:
        // models/<source>/<namespace>/<name>/<tag>/manifest.json -> ../../../../../manifests/sha256/<digest>.json
        // models/<source>/<namespace>/<name>/<tag>/weights -> ../../../../../blobs/sha256/<digest>
        let manifest_target = PathBuf::from("../../../../../manifests/sha256")
            .join(format!("{manifest_digest}.json"));
        let weights_target =
            PathBuf::from("../../../../../blobs/sha256").join(&manifest.weight_digest);

        let manifest_link = alias_dir.join("manifest.json");
        let weights_link = alias_dir.join("weights");

        let _ = fs::remove_file(&manifest_link);
        let _ = fs::remove_file(&weights_link);

        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;
            symlink(&manifest_target, &manifest_link).map_err(|e| K3Error::CapabilityMismatch {
                architecture: "cas_alias".to_owned(),
                reason: format!(
                    "failed to create manifest symlink '{}' -> '{}': {e}",
                    manifest_link.display(),
                    manifest_target.display()
                ),
            })?;

            symlink(&weights_target, &weights_link).map_err(|e| K3Error::CapabilityMismatch {
                architecture: "cas_alias".to_owned(),
                reason: format!(
                    "failed to create weights symlink '{}' -> '{}': {e}",
                    weights_link.display(),
                    weights_target.display()
                ),
            })?;
        }

        #[cfg(not(unix))]
        {
            // Fallback for non-unix platforms
            fs::write(&manifest_link, manifest_target.to_string_lossy().as_bytes()).map_err(
                |e| K3Error::CapabilityMismatch {
                    architecture: "cas_alias".to_owned(),
                    reason: format!("failed to write fallback manifest link: {e}"),
                },
            )?;
            fs::write(&weights_link, weights_target.to_string_lossy().as_bytes()).map_err(|e| {
                K3Error::CapabilityMismatch {
                    architecture: "cas_alias".to_owned(),
                    reason: format!("failed to write fallback weights link: {e}"),
                }
            })?;
        }

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let alias_rel_path = format!("models/{source}/{namespace}/{name}/{tag}");

        let record = ModelAliasRecord {
            alias_id: alias_id.clone(),
            source: source.to_owned(),
            namespace: namespace.to_owned(),
            name: name.to_owned(),
            tag: tag.to_owned(),
            manifest_digest: manifest_digest.to_owned(),
            weight_digest: manifest.weight_digest.clone(),
            alias_path: alias_rel_path,
            created_at: now,
        };

        self.catalog
            .save_alias(&record)
            .map_err(|e| K3Error::CapabilityMismatch {
                architecture: "cas_catalog".to_owned(),
                reason: format!("failed to save alias in DuckDB catalog: {e}"),
            })?;

        // Add reference in DuckDB
        let ref_rec = BlobReferenceRecord {
            ref_id: format!("alias:{alias_id}"),
            blob_digest: manifest.weight_digest,
            ref_type: "alias".to_owned(),
            ref_source: alias_id,
            created_at: now,
        };
        self.catalog
            .add_blob_reference(&ref_rec)
            .map_err(|e| K3Error::CapabilityMismatch {
                architecture: "cas_catalog".to_owned(),
                reason: format!("failed to record alias blob reference: {e}"),
            })?;

        Ok(record)
    }

    /// Resolves a human-browsable alias into its manifest and canonical weights path (US-18 Scenario 3).
    ///
    /// Verifies symlinks resolve strictly within the store root to prevent path traversal attacks.
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] if alias resolution fails or points outside the store.
    #[allow(clippy::too_many_lines)]
    pub fn resolve_alias(&self, alias_str: &str) -> Result<ResolvedAlias, K3Error> {
        let (source, namespace, name, tag) = parse_alias_id(alias_str)?;
        let alias_id = format!("{source}/{namespace}/{name}/{tag}");

        let alias_dir = self
            .models_dir()
            .join(&source)
            .join(&namespace)
            .join(&name)
            .join(&tag);
        let canon_root = fs::canonicalize(&self.root).unwrap_or_else(|_| self.root.clone());

        // Symlink escape defense on physical alias directory
        if alias_dir.exists() {
            let weights_link = alias_dir.join("weights");
            if (weights_link.is_symlink() || weights_link.exists())
                && let Ok(canon_weights) = fs::canonicalize(&weights_link)
                && !canon_weights.starts_with(&canon_root)
            {
                return Err(K3Error::CapabilityMismatch {
                    architecture: "cas_security".to_owned(),
                    reason: format!("alias '{alias_id}' resolved outside configured store root"),
                });
            }
            let manifest_link = alias_dir.join("manifest.json");
            if (manifest_link.is_symlink() || manifest_link.exists())
                && let Ok(canon_manifest) = fs::canonicalize(&manifest_link)
                && !canon_manifest.starts_with(&canon_root)
            {
                return Err(K3Error::CapabilityMismatch {
                    architecture: "cas_security".to_owned(),
                    reason: format!("alias '{alias_id}' resolved outside configured store root"),
                });
            }
        }

        let alias_record = self
            .catalog
            .find_alias(&source, &namespace, &name, &tag)
            .map_err(|e| K3Error::CapabilityMismatch {
                architecture: "cas_resolve".to_owned(),
                reason: format!("catalog query failed for alias '{alias_id}': {e}"),
            })?
            .ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: "cas_resolve".to_owned(),
                reason: format!("model alias '{alias_id}' is not registered"),
            })?;

        let manifest_record = self
            .catalog
            .get_manifest(&alias_record.manifest_digest)
            .map_err(|e| K3Error::CapabilityMismatch {
                architecture: "cas_resolve".to_owned(),
                reason: format!(
                    "catalog query failed for manifest '{}': {e}",
                    alias_record.manifest_digest
                ),
            })?
            .ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: "cas_resolve".to_owned(),
                reason: format!(
                    "manifest '{}' for alias '{alias_id}' not found in catalog",
                    alias_record.manifest_digest
                ),
            })?;

        let manifest: ModelManifest = serde_json::from_str(&manifest_record.manifest_json)
            .map_err(|e| K3Error::CapabilityMismatch {
                architecture: "cas_resolve".to_owned(),
                reason: format!("corrupt manifest JSON in catalog: {e}"),
            })?;

        let weight_path = self.blobs_dir().join(&manifest.weight_digest);
        let manifest_path = self
            .manifests_dir()
            .join(format!("{}.json", alias_record.manifest_digest));

        if !weight_path.exists() {
            return Err(K3Error::CapabilityMismatch {
                architecture: "cas_resolve".to_owned(),
                reason: format!(
                    "weight blob '{}' referenced by alias '{alias_id}' is missing from storage",
                    weight_path.display()
                ),
            });
        }

        // Symlink traversal defense: verify resolved paths reside under canonical store root
        let canon_weight =
            fs::canonicalize(&weight_path).map_err(|e| K3Error::CapabilityMismatch {
                architecture: "cas_security".to_owned(),
                reason: format!(
                    "failed to canonicalize weight path '{}': {e}",
                    weight_path.display()
                ),
            })?;
        let canon_manifest =
            fs::canonicalize(&manifest_path).map_err(|e| K3Error::CapabilityMismatch {
                architecture: "cas_security".to_owned(),
                reason: format!(
                    "failed to canonicalize manifest path '{}': {e}",
                    manifest_path.display()
                ),
            })?;

        if !canon_weight.starts_with(&canon_root) || !canon_manifest.starts_with(&canon_root) {
            return Err(K3Error::CapabilityMismatch {
                architecture: "cas_security".to_owned(),
                reason: format!("alias '{alias_id}' resolved outside configured store root"),
            });
        }

        // Update access timestamp
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let _ = self
            .catalog
            .update_blob_access(&manifest.weight_digest, now);

        Ok(ResolvedAlias {
            alias_id,
            manifest,
            manifest_path,
            weight_path,
        })
    }

    /// Removes a model alias and unlinks its relative directory, decrementing reference counts
    /// in `DuckDB` without immediately deleting the underlying weight blob (US-22 Scenario 1).
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] on failure.
    pub fn remove_alias(
        &self,
        source: &str,
        namespace: &str,
        name: &str,
        tag: &str,
    ) -> Result<bool, K3Error> {
        validate_path_component(source)?;
        validate_path_component(namespace)?;
        validate_path_component(name)?;
        validate_path_component(tag)?;

        let alias_id = format!("{source}/{namespace}/{name}/{tag}");
        let alias_dir = self
            .models_dir()
            .join(source)
            .join(namespace)
            .join(name)
            .join(tag);

        // Remove from DuckDB catalog
        let alias_opt =
            self.catalog
                .get_alias(&alias_id)
                .map_err(|e| K3Error::CapabilityMismatch {
                    architecture: "cas_rm".to_owned(),
                    reason: format!("failed to query alias '{alias_id}': {e}"),
                })?;

        let Some(alias_rec) = alias_opt else {
            return Ok(false);
        };

        // Remove alias reference from blob_references (this decrements ref_count in DuckDB)
        let ref_id = format!("alias:{alias_id}");
        let _ = self.catalog.remove_blob_reference(&ref_id);

        // Delete from model_aliases table
        self.catalog
            .delete_alias(&alias_id)
            .map_err(|e| K3Error::CapabilityMismatch {
                architecture: "cas_rm".to_owned(),
                reason: format!("failed to delete alias from catalog: {e}"),
            })?;

        // Remove directory and symlinks from filesystem
        if alias_dir.exists() {
            let _ = fs::remove_dir_all(&alias_dir);
        }

        // Check if manifest is still referenced by any other alias
        let all_aliases = self.catalog.list_aliases().unwrap_or_default();
        let manifest_has_aliases = all_aliases
            .iter()
            .any(|a| a.manifest_digest == alias_rec.manifest_digest);

        if !manifest_has_aliases {
            // Unlink manifest file and remove manifest reference
            let manifest_path = self
                .manifests_dir()
                .join(format!("{}.json", alias_rec.manifest_digest));
            let _ = fs::remove_file(&manifest_path);

            let manifest_ref_id = format!("manifest:{}", alias_rec.manifest_digest);
            let _ = self.catalog.remove_blob_reference(&manifest_ref_id);
            let _ = self.catalog.delete_manifest(&alias_rec.manifest_digest);
        }

        // Clean up empty parent directories under models/
        let name_dir = self.models_dir().join(source).join(namespace).join(name);
        let _ = fs::remove_dir(&name_dir);
        let ns_dir = self.models_dir().join(source).join(namespace);
        let _ = fs::remove_dir(&ns_dir);
        let s_dir = self.models_dir().join(source);
        let _ = fs::remove_dir(&s_dir);

        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::DuckDbPersistenceStore;

    struct TempDirGuard {
        path: PathBuf,
    }

    impl TempDirGuard {
        fn new(prefix: &str) -> Self {
            let seq = STAGING_COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!("{prefix}_{}_{seq}", std::process::id()));
            fs::create_dir_all(&path).expect("create temp dir");
            Self { path }
        }

        fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for TempDirGuard {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    #[test]
    fn test_machine_feature_schema_and_manifest_canonical_json() {
        let mut mappings = BTreeMap::new();
        mappings.insert(
            "anomaly".to_owned(),
            MachineEventRule {
                event_code: "STAGE_RESCHEDULE".to_owned(),
                target_subsystem: "ActionRunner".to_owned(),
                enter_threshold: Some(0.82),
                exit_threshold: Some(0.40),
                cooldown_ms: Some(5000),
            },
        );
        let schema = MachineFeatureSchema {
            feature_names: vec!["cpu_util".to_owned(), "queue_depth".to_owned()],
            feature_types: vec!["f32".to_owned(), "f32".to_owned()],
            zscore_mean: vec![0.45, 4.0],
            zscore_std: vec![0.15, 2.0],
            event_mappings: mappings,
        };
        schema.validate().expect("valid schema");
        assert_ne!(schema.schema_hash(), 0);
        let normalized = schema
            .normalize_features(&[0.60, 8.0], 10.0)
            .expect("normalize features");
        assert!((normalized[0] - 1.0).abs() < 1e-5);
        assert!((normalized[1] - 2.0).abs() < 1e-5);

        let digest = "a".repeat(64);
        let core_plain = ModelManifest::new("gguf", "ttm", "1.0.0", "F32", 512, digest, 32);
        let wrapped_none = MachineModelManifest::from_core(core_plain.clone());
        assert_eq!(
            wrapped_none.to_canonical_json().unwrap(),
            core_plain.to_canonical_json().unwrap()
        );
        assert_eq!(
            wrapped_none.manifest_digest().unwrap(),
            core_plain.manifest_digest().unwrap()
        );

        let with_schema =
            MachineModelManifest::new(core_plain.clone(), Some(schema.clone())).expect("valid");
        let parsed: MachineModelManifest =
            serde_json::from_str(&with_schema.to_canonical_json().unwrap()).expect("roundtrip");
        assert_eq!(parsed, with_schema);
        assert_ne!(
            with_schema.manifest_digest().unwrap(),
            core_plain.manifest_digest().unwrap()
        );

        let bad_schema = MachineFeatureSchema {
            zscore_std: vec![0.15, 0.0],
            ..schema
        };
        assert!(MachineModelManifest::new(core_plain, Some(bad_schema)).is_err());
    }

    fn stage_and_promote_test_gguf(store: &ContentAddressedStore) -> (String, u64) {
        let mut weight_bytes = Vec::with_capacity(32);
        weight_bytes.extend_from_slice(b"GGUF");
        weight_bytes.extend_from_slice(&3u32.to_le_bytes());
        weight_bytes.extend_from_slice(&0u64.to_le_bytes());
        weight_bytes.extend_from_slice(&0u64.to_le_bytes());
        weight_bytes.extend_from_slice(b"TTM_DATA");
        let weight_digest = Sha256::digest(&weight_bytes);
        let weight_size_bytes = u64::try_from(weight_bytes.len()).expect("len u64");

        let (staging_path, mut staging_file) = store.create_staging_file().expect("staging file");
        staging_file.write_all(&weight_bytes).expect("write bytes");
        staging_file.flush().expect("flush");
        drop(staging_file);
        store
            .promote_staged_blob(&staging_path, &weight_digest, "gguf")
            .expect("promote blob");
        (weight_digest, weight_size_bytes)
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_model_manifest_machine_schema_roundtrip() {
        let dir = TempDirGuard::new("k3_machine_schema_cas_test");
        let db_path = dir.path().join("catalog.duckdb");
        let catalog = Arc::new(DuckDbPersistenceStore::open(&db_path).expect("open catalog"));
        let store =
            ContentAddressedStore::open(dir.path(), Arc::clone(&catalog) as Arc<dyn CasCatalog>)
                .expect("open cas");

        let (weight_digest, weight_size_bytes) = stage_and_promote_test_gguf(&store);
        let core_plain = ModelManifest::new(
            "gguf",
            "ttm",
            "1.0.0",
            "F32",
            512,
            weight_digest,
            weight_size_bytes,
        );

        let mut mappings = BTreeMap::new();
        mappings.insert(
            "anomaly".to_owned(),
            MachineEventRule {
                event_code: "STAGE_RESCHEDULE".to_owned(),
                target_subsystem: "ActionRunner".to_owned(),
                enter_threshold: Some(0.82),
                exit_threshold: Some(0.40),
                cooldown_ms: Some(5000),
            },
        );
        let schema = MachineFeatureSchema {
            feature_names: vec!["cpu_util".to_owned(), "queue_depth".to_owned()],
            feature_types: vec!["f32".to_owned(), "f32".to_owned()],
            zscore_mean: vec![0.45, 4.0],
            zscore_std: vec![0.15, 2.0],
            event_mappings: mappings,
        };
        let manifest =
            MachineModelManifest::new(core_plain.clone(), Some(schema.clone())).expect("valid");
        let rec = store
            .register_machine_manifest(&manifest)
            .expect("register machine manifest");
        assert_ne!(
            rec.manifest_digest,
            core_plain.manifest_digest().unwrap(),
            "machine_schema must participate in content-addressed manifest digest"
        );

        // Verify DuckDB machine_schema_json sidecar column was populated
        let schema_col = catalog
            .get_machine_schema_json(&rec.manifest_digest)
            .expect("query schema column")
            .expect("machine_schema_json column populated");
        let parsed_col: MachineFeatureSchema =
            serde_json::from_str(&schema_col).expect("parse machine_schema_json column");
        assert_eq!(parsed_col, schema);

        // Bind alias & verify lookup via digest and alias
        store
            .create_alias(
                "local",
                "telemetry",
                "ttm-guard",
                "latest",
                &rec.manifest_digest,
            )
            .expect("create alias");
        let loaded = store
            .load_manifest_with_schema(&rec.manifest_digest)
            .expect("load manifest with schema");
        assert_eq!(loaded, manifest);
        assert_eq!(
            store
                .get_machine_schema("local/telemetry/ttm-guard/latest")
                .expect("resolve schema by alias"),
            Some(schema.clone())
        );

        // Invalid schema (zscore_std <= 0) must fail upfront
        let bad_schema = MachineFeatureSchema {
            zscore_std: vec![0.15, 0.0],
            ..schema
        };
        assert!(MachineModelManifest::new(core_plain, Some(bad_schema)).is_err());

        // Reopen existing DuckDB catalog after drop and confirm schema persists cleanly
        drop(store);
        drop(catalog);
        let reopened_catalog =
            Arc::new(DuckDbPersistenceStore::open(&db_path).expect("reopen catalog"));
        let reopened_store = ContentAddressedStore::open(
            dir.path(),
            Arc::clone(&reopened_catalog) as Arc<dyn CasCatalog>,
        )
        .expect("reopen store");
        let reloaded = reopened_store
            .load_manifest_with_schema(&rec.manifest_digest)
            .expect("reload after restart");
        assert_eq!(reloaded, manifest);
        assert!(
            reopened_catalog
                .get_machine_schema_json(&rec.manifest_digest)
                .expect("query after reopen")
                .is_some()
        );
    }

    #[test]
    fn test_store_root_and_catalog_path_precedence() {
        use std::collections::HashMap;

        // 1. JERO_STORE_ROOT beats K3_STORE_ROOT and filesystem paths
        let env_both = HashMap::from([
            (ENV_STORE_ROOT, "/mnt/nvme/jero-store"),
            (LEGACY_ENV_STORE_ROOT, "/mnt/nvme/k3-store"),
        ]);
        assert_eq!(
            resolve_store_root_with(|k| env_both.get(k).map(|v| (*v).to_owned()), |_| true),
            PathBuf::from("/mnt/nvme/jero-store")
        );

        // 2. K3_STORE_ROOT fallback when JERO_STORE_ROOT is unset/blank
        let env_legacy = HashMap::from([
            (ENV_STORE_ROOT, "  "),
            (LEGACY_ENV_STORE_ROOT, "/mnt/nvme/k3-store"),
        ]);
        assert_eq!(
            resolve_store_root_with(|k| env_legacy.get(k).map(|v| (*v).to_owned()), |_| true),
            PathBuf::from("/mnt/nvme/k3-store")
        );

        // 3. No env: /var/lib/jero-runner preferred if it exists (even if /var/lib/k3 also exists)
        assert_eq!(
            resolve_store_root_with(|_| None, |_| true),
            PathBuf::from(DEFAULT_STORE_ROOT)
        );

        // 4. No env: falls back to /var/lib/k3 if only /var/lib/k3 exists
        assert_eq!(
            resolve_store_root_with(|_| None, |p| p == Path::new(LEGACY_DEFAULT_STORE_ROOT)),
            PathBuf::from(LEGACY_DEFAULT_STORE_ROOT)
        );

        // 5. No env and neither exists: defaults to /var/lib/jero-runner
        assert_eq!(
            resolve_store_root_with(|_| None, |_| false),
            PathBuf::from(DEFAULT_STORE_ROOT)
        );

        // 6. Catalog path precedence: JERO_CATALOG_PATH > K3_CATALOG_PATH > <store_root>/catalog.duckdb
        let store_root = Path::new("/var/lib/jero-runner");
        let cat_both = HashMap::from([
            (ENV_CATALOG_PATH, "/var/lib/jero-runner/custom.duckdb"),
            (LEGACY_ENV_CATALOG_PATH, "/var/lib/k3/legacy.duckdb"),
        ]);
        assert_eq!(
            resolve_catalog_path_with(store_root, |k| cat_both.get(k).map(|v| (*v).to_owned())),
            PathBuf::from("/var/lib/jero-runner/custom.duckdb")
        );

        let cat_legacy = HashMap::from([(LEGACY_ENV_CATALOG_PATH, "/var/lib/k3/legacy.duckdb")]);
        assert_eq!(
            resolve_catalog_path_with(store_root, |k| cat_legacy.get(k).map(|v| (*v).to_owned())),
            PathBuf::from("/var/lib/k3/legacy.duckdb")
        );

        assert_eq!(
            resolve_catalog_path_with(store_root, |_| None),
            PathBuf::from("/var/lib/jero-runner/catalog.duckdb")
        );
    }
}

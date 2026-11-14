//! Derived Weight Layout Caching & Stale Recipe Detection (US-21, ADR-037, ADR-038).
//!
//! Caches hardware-repacked (Cluster 1 A100 `8x16` `i8` IME tiles) or format-converted weights
//! as derived blobs keyed by `(source_digest, recipe_version, target_layout)`. Enforces
//! operator-controlled retention policies, memory admission accounting, and auto-rebuilding
//! of stale derivations upon runtime upgrades.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use jero_core::{K3Error, QuantType, TensorView};
use jero_kernels_rvv::ime::{TILE_K, pack_tiles};
use serde::{Deserialize, Serialize};

use super::cas::ContentAddressedStore;
use super::sha256::Sha256;
use crate::persistence::{BlobReferenceRecord, CasCatalog, DerivedBlobRecord, ModelBlobRecord};

/// Operator-configurable retention policy for original source weights after a derived layout is generated (US-21 Scenario 2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum RetentionPolicy {
    /// Retains original source weights indefinitely in storage alongside derived layouts (default).
    #[default]
    Keep,
    /// Immediately marks original source weights for reclamation once the derived layout is promoted.
    PurgeImmediate,
    /// Retains original source weights for an operator-configured grace period before marking for reclamation.
    PurgeAfterDelay {
        /// Grace period in seconds (e.g. 86400 for 24 hours).
        delay_secs: u64,
    },
}

impl RetentionPolicy {
    /// Returns the canonical policy string identifier.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Keep => "keep",
            Self::PurgeImmediate => "purge_immediate",
            Self::PurgeAfterDelay { .. } => "purge_after_delay",
        }
    }
}

/// Outcome of derived weight resolution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DerivedResolution {
    /// The derived weight layout was already cached on disk and in `DuckDB` catalog (US-21 Scenario 1).
    Cached {
        /// SHA-256 digest of derived layout blob.
        derived_digest: String,
        /// Filesystem path to cached derived blob.
        derived_path: PathBuf,
    },
    /// A new derived layout was freshly converted and registered.
    Created {
        /// SHA-256 digest of derived layout blob.
        derived_digest: String,
        /// Filesystem path to newly created derived blob.
        derived_path: PathBuf,
    },
}

/// Outcome of evaluating potentially stale derived layouts upon breaking runtime upgrades (US-21 Scenario 3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StaleRecipeOutcome {
    /// Layout was already valid and up-to-date with active recipe version.
    UpToDate { derived_digest: String },
    /// Layout was stale, but original source weights were present and layout was auto-rebuilt.
    Rebuilt {
        new_derived_digest: String,
        new_derived_path: PathBuf,
    },
    /// Layout was stale and original source weights were purged; requires operator advisory.
    SourcePurgedAdvisory {
        source_digest: String,
        stale_recipe_version: String,
        active_recipe_version: String,
        advisory_message: String,
    },
}

/// Enforces memory admission accounting for simultaneously loaded weight layouts (US-21, ADR-037).
///
/// Refuses load with [`K3Error::InsufficientLockableMemory`] if total required locked bytes
/// exceed available RAM or `RLIMIT_MEMLOCK` bound.
///
/// # Errors
/// Returns [`K3Error::InsufficientLockableMemory`] on ceiling violation.
pub fn check_derived_memory_admission(
    current_locked_bytes: u64,
    additional_layout_bytes: u64,
    kv_arena_bytes: u64,
    available_ram_bytes: u64,
    rlimit_memlock_bytes: u64,
) -> Result<(), K3Error> {
    let total_required = current_locked_bytes
        .saturating_add(additional_layout_bytes)
        .saturating_add(kv_arena_bytes);

    if (rlimit_memlock_bytes > 0 && total_required > rlimit_memlock_bytes)
        || (available_ram_bytes > 0 && total_required > available_ram_bytes)
    {
        return Err(K3Error::InsufficientLockableMemory {
            required_bytes: total_required,
            weight_bytes: current_locked_bytes.saturating_add(additional_layout_bytes),
            kv_bytes: kv_arena_bytes,
            available_bytes: available_ram_bytes,
            rlimit_memlock_bytes,
        });
    }

    Ok(())
}

/// Manager for derived weight layout caching, retention lifecycle, and stale recipe detection.
#[derive(Debug, Clone)]
pub struct DerivedBlobManager {
    store: Arc<ContentAddressedStore>,
    catalog: Arc<dyn CasCatalog>,
}

impl DerivedBlobManager {
    /// Constructs a new derived blob manager bound to a content-addressed store.
    #[must_use]
    pub fn new(store: Arc<ContentAddressedStore>) -> Self {
        let catalog = store.catalog().clone();
        Self { store, catalog }
    }

    /// Resolves or builds a hardware-repacked or format-converted derived blob (US-21 Scenario 1 & 2).
    ///
    /// Keyed by `(source_digest, recipe_version, target_layout)`.
    ///
    /// If already cached and valid, directly returns [`DerivedResolution::Cached`].
    /// Otherwise, invokes `converter` to populate a staging file, promotes it to `blobs/sha256/<derived_digest>`,
    /// registers in `DuckDB`, and applies the operator's configured `retention` policy to the source weights.
    ///
    /// # Errors
    /// Returns [`K3Error`] on conversion failure, catalog error, or missing source weights.
    #[allow(clippy::too_many_lines)]
    pub fn get_or_create_derived_layout<F>(
        &self,
        source_digest: &str,
        recipe_version: &str,
        target_layout: &str,
        retention: RetentionPolicy,
        converter: F,
    ) -> Result<DerivedResolution, K3Error>
    where
        F: FnOnce(&Path, &Path) -> Result<u64, K3Error>,
    {
        super::cas::validate_sha256_digest(source_digest)?;
        let source_clean = source_digest.to_owned();

        // 1. Check if valid derived blob already exists in DuckDB catalog and filesystem (Scenario 1)
        if let Ok(Some(existing)) =
            self.catalog
                .get_derived_blob(source_digest, recipe_version, target_layout)
            && existing.status == "valid"
        {
            let derived_path = self.store.blobs_dir().join(&existing.derived_digest);
            if derived_path.exists() {
                // Update access timestamp
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                let _ = self
                    .catalog
                    .update_blob_access(&existing.derived_digest, now);

                return Ok(DerivedResolution::Cached {
                    derived_digest: existing.derived_digest,
                    derived_path,
                });
            }
        }

        // 2. Validate that original source weights blob exists
        super::cas::validate_sha256_digest(source_digest)?;
        let blobs_dir = self.store.blobs_dir();
        let source_path = blobs_dir.join(source_digest);
        if !source_path.exists() {
            return Err(K3Error::CapabilityMismatch {
                architecture: "derived_weights".to_owned(),
                reason: format!(
                    "original source weights blob '{}' not found in store; cannot derive layout",
                    source_path.display()
                ),
            });
        }
        let canon_blobs = fs::canonicalize(&blobs_dir).unwrap_or(blobs_dir);
        let canon_source = fs::canonicalize(&source_path).unwrap_or_else(|_| source_path.clone());
        if !canon_source.starts_with(&canon_blobs) {
            return Err(K3Error::CapabilityMismatch {
                architecture: "derived_weights".to_owned(),
                reason: format!("path traversal detected in source_digest '{source_digest}'"),
            });
        }

        // 3. Create staging file for conversion
        let (staging_path, mut staging_file) = self.store.create_staging_file()?;

        // 4. Run conversion
        let derived_bytes = match converter(&source_path, &staging_path) {
            Ok(bytes) => bytes,
            Err(e) => {
                let _ = fs::remove_file(&staging_path);
                return Err(e);
            }
        };

        // Flush staging file
        staging_file
            .flush()
            .map_err(|e| K3Error::CapabilityMismatch {
                architecture: "derived_weights".to_owned(),
                reason: format!("failed to flush staging derived weights: {e}"),
            })?;
        drop(staging_file);

        // 5. Compute SHA-256 digest of converted derived weights
        let derived_digest = Sha256::digest_file(&staging_path).map_err(|e| {
            let _ = fs::remove_file(&staging_path);
            K3Error::CapabilityMismatch {
                architecture: "derived_weights".to_owned(),
                reason: format!("failed to hash derived staging weights: {e}"),
            }
        })?;

        // 6. Atomically promote staging file to blobs/sha256/<derived_digest>
        let final_derived_path = self.store.blobs_dir().join(&derived_digest);
        if final_derived_path.exists() {
            let _ = fs::remove_file(&staging_path);
        } else {
            fs::rename(&staging_path, &final_derived_path).map_err(|e| {
                let _ = fs::remove_file(&staging_path);
                K3Error::CapabilityMismatch {
                    architecture: "derived_weights".to_owned(),
                    reason: format!(
                        "failed to atomically promote derived weights to '{}': {e}",
                        final_derived_path.display()
                    ),
                }
            })?;

            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let _ = fs::set_permissions(&final_derived_path, fs::Permissions::from_mode(0o444));
            }
        }

        // 7. Register derived blob in DuckDB catalog
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let blob_rec = ModelBlobRecord {
            digest: derived_digest.clone(),
            size_bytes: derived_bytes,
            blob_type: "derived_weights".to_owned(),
            created_at: now,
            last_accessed_at: now,
            ref_count: 1,
        };
        self.catalog
            .save_blob(&blob_rec)
            .map_err(|e| K3Error::CapabilityMismatch {
                architecture: "derived_catalog".to_owned(),
                reason: format!("failed to save derived blob record: {e}"),
            })?;

        let derived_rec = DerivedBlobRecord {
            derived_digest: derived_digest.clone(),
            source_digest: source_clean.clone(),
            recipe_version: recipe_version.to_owned(),
            target_layout: target_layout.to_owned(),
            status: "valid".to_owned(),
            created_at: now,
        };
        self.catalog
            .save_derived_blob(&derived_rec)
            .map_err(|e| K3Error::CapabilityMismatch {
                architecture: "derived_catalog".to_owned(),
                reason: format!("failed to save derived association: {e}"),
            })?;

        // Add reference linking derived blob to source
        let ref_rec = BlobReferenceRecord {
            ref_id: format!("derived:{derived_digest}"),
            blob_digest: derived_digest.clone(),
            ref_type: "derived".to_owned(),
            ref_source: source_clean.clone(),
            created_at: now,
        };
        let _ = self.catalog.add_blob_reference(&ref_rec);

        // 8. Apply operator retention policy (Scenario 2)
        match retention {
            RetentionPolicy::Keep => {
                // Keep source blob indefinitely; no action needed
            }
            RetentionPolicy::PurgeImmediate => {
                // Mark source blob for immediate GC reclamation if unreferenced by other manifests
                let purge_key = format!("purge_immediate:{source_clean}");
                let _ = self.catalog.set_storage_config(&purge_key, "true");

                // Release manifest and alias references from blob_references
                if let Ok(refs) = self.catalog.list_blob_references(&source_clean) {
                    for r in refs {
                        if r.ref_type == "manifest" || r.ref_type == "alias" {
                            let _ = self.catalog.remove_blob_reference(&r.ref_id);
                        }
                    }
                }
                let _ = self.catalog.decrement_blob_ref(&source_clean);
            }
            RetentionPolicy::PurgeAfterDelay { delay_secs } => {
                let purge_timestamp = now.saturating_add(delay_secs);
                let purge_key = format!("purge_after:{source_clean}");
                let _ = self
                    .catalog
                    .set_storage_config(&purge_key, &purge_timestamp.to_string());
            }
        }

        Ok(DerivedResolution::Created {
            derived_digest,
            derived_path: final_derived_path,
        })
    }

    /// Evaluates existing derived weight layouts against the active runtime recipe version (US-21 Scenario 3).
    ///
    /// Upon breaking recipe upgrades:
    /// - Flags outdated derivations as `status = 'stale'`.
    /// - If original source weights are still present, automatically rebuilds the layout.
    /// - If original source weights were purged, returns [`StaleRecipeOutcome::SourcePurgedAdvisory`]
    ///   with an actionable operator advisory notice.
    ///
    /// # Errors
    /// Returns [`K3Error`] on rebuild error or catalog failure.
    pub fn detect_and_handle_stale_recipe<F>(
        &self,
        source_digest: &str,
        active_recipe_version: &str,
        target_layout: &str,
        model_name: &str,
        rebuilder: F,
    ) -> Result<StaleRecipeOutcome, K3Error>
    where
        F: FnOnce(&Path, &Path) -> Result<u64, K3Error>,
    {
        super::cas::validate_sha256_digest(source_digest)?;
        let source_clean = source_digest.to_owned();

        // Query existing derived blobs for this source and layout
        let all_derived = self
            .catalog
            .list_derived_for_source(source_digest)
            .map_err(|e| K3Error::CapabilityMismatch {
                architecture: "derived_stale".to_owned(),
                reason: format!("failed to list derived blobs for source: {e}"),
            })?;

        let matching_layout: Vec<DerivedBlobRecord> = all_derived
            .into_iter()
            .filter(|d| d.target_layout == target_layout)
            .collect();

        // If a valid derivation exists for the active recipe version, we are up to date
        if let Some(up_to_date) = matching_layout
            .iter()
            .find(|d| d.recipe_version == active_recipe_version && d.status == "valid")
        {
            return Ok(StaleRecipeOutcome::UpToDate {
                derived_digest: up_to_date.derived_digest.clone(),
            });
        }

        // Check if there is an outdated derivation
        let Some(stale_rec) = matching_layout
            .iter()
            .find(|d| d.recipe_version != active_recipe_version)
        else {
            // No prior derivation at all: not stale, simply unbuilt
            return Ok(StaleRecipeOutcome::UpToDate {
                derived_digest: String::new(),
            });
        };

        // Flag existing derived record as stale in DuckDB
        let _ = self
            .catalog
            .update_derived_status(&stale_rec.derived_digest, "stale");

        // Check whether the original source weights are still present on disk
        let source_path = self.store.blobs_dir().join(&source_clean);
        if !source_path.starts_with(self.store.blobs_dir()) {
            return Err(K3Error::CapabilityMismatch {
                architecture: "derived_stale".to_owned(),
                reason: format!("path traversal detected in source_digest '{source_digest}'"),
            });
        }
        if source_path.exists() {
            // Auto-rebuild the derived blob for the new recipe version (US-21 Scenario 3)
            let resolution = self.get_or_create_derived_layout(
                &source_clean,
                active_recipe_version,
                target_layout,
                RetentionPolicy::Keep,
                rebuilder,
            )?;

            match resolution {
                DerivedResolution::Cached {
                    derived_digest,
                    derived_path,
                }
                | DerivedResolution::Created {
                    derived_digest,
                    derived_path,
                } => Ok(StaleRecipeOutcome::Rebuilt {
                    new_derived_digest: derived_digest,
                    new_derived_path: derived_path,
                }),
            }
        } else {
            // Original source weights were purged; prompt operator to re-acquire (US-21 Scenario 3)
            let advisory = format!(
                "Derived weight layout for model '{model_name}' (source: {source_clean}) is stale \
                 (built for recipe version {}, current runtime is {active_recipe_version}), \
                 and original source weights have been purged. Please re-acquire the source model.",
                stale_rec.recipe_version
            );

            Ok(StaleRecipeOutcome::SourcePurgedAdvisory {
                source_digest: source_clean,
                stale_recipe_version: stale_rec.recipe_version.clone(),
                active_recipe_version: active_recipe_version.to_owned(),
                advisory_message: advisory,
            })
        }
    }
}

/// Pre-splits fused Phi-3/Phi-4 weights (e.g. `attn_qkv.weight` into Q, K, V and `ffn_up.weight`
/// into Gate, Up) using zero-copy row slicing and repacks each partition into Cluster 1 A100
/// `8x16` `i8` IME tiles.
///
/// # Errors
/// Returns [`K3Error::TensorBoundsError`] if:
/// - `src` has fewer than 2 dimensions,
/// - the sum of `partition_rows` does not equal `src.shape()[1]`,
/// - `src.shape()[0]` is not a multiple of `TILE_K` (16),
/// - or if slice row bounds or byte ranges fail validation.
pub fn repack_phi3_fused_tensor_ime_8x16(
    src: &TensorView<'_>,
    partition_rows: &[usize],
) -> Result<Vec<Vec<i8>>, K3Error> {
    let shape = src.shape();
    if shape.len() < 2 {
        return Err(K3Error::TensorBoundsError {
            tensor_name: src.name().to_owned(),
            offset: src.offset(),
            byte_len: src.byte_len(),
            buffer_len: src.byte_len(),
            reason: format!(
                "repack_phi3_fused_tensor_ime_8x16 requires at least 2 dimensions, found shape with {} dimension(s)",
                shape.len()
            ),
        });
    }

    let total_rows = shape[1];
    let partition_sum: usize = partition_rows.iter().copied().sum();
    if partition_sum != total_rows {
        return Err(K3Error::TensorBoundsError {
            tensor_name: src.name().to_owned(),
            offset: src.offset(),
            byte_len: src.byte_len(),
            buffer_len: src.byte_len(),
            reason: format!(
                "partition rows sum ({partition_sum}) does not match tensor row count ({total_rows})"
            ),
        });
    }

    let k = shape[0];
    if k % TILE_K != 0 {
        return Err(K3Error::TensorBoundsError {
            tensor_name: src.name().to_owned(),
            offset: src.offset(),
            byte_len: src.byte_len(),
            buffer_len: src.byte_len(),
            reason: format!("tensor column dimension {k} must be a multiple of TILE_K ({TILE_K})"),
        });
    }

    let mut results = Vec::with_capacity(partition_rows.len());
    let mut current_row = 0usize;

    for &p_rows in partition_rows {
        let sub_view = src.slice_rows(current_row, p_rows)?;
        current_row += p_rows;

        let i8_vec: Vec<i8> = if sub_view.quant() == QuantType::Q8_0 {
            let mut v = Vec::with_capacity(p_rows * k);
            for row in 0..p_rows {
                let row_blocks = k / 32;
                let row_start_byte = row * row_blocks * 34;
                for b in 0..row_blocks {
                    let block_start = row_start_byte + b * 34;
                    let qs = &sub_view.as_bytes()[block_start + 2..block_start + 34];
                    // SAFETY: u8 and i8 have identical 1-byte size, alignment, and all bit patterns are valid.
                    // Length is guaranteed to be exactly 32 bytes from the sub-slice above.
                    let qs_i8: &[i8] =
                        unsafe { core::slice::from_raw_parts(qs.as_ptr().cast::<i8>(), 32) };
                    v.extend_from_slice(qs_i8);
                }
            }
            v
        } else {
            let raw = sub_view.as_bytes();
            let expected = p_rows * k;
            if raw.len() == expected {
                // SAFETY: u8 and i8 have identical 1-byte size, alignment, and valid bit patterns;
                // slice length is verified to match raw.len().
                let i8_slice: &[i8] =
                    unsafe { core::slice::from_raw_parts(raw.as_ptr().cast::<i8>(), raw.len()) };
                i8_slice.to_vec()
            } else {
                return Err(K3Error::TensorBoundsError {
                    tensor_name: sub_view.name().to_owned(),
                    offset: sub_view.offset(),
                    byte_len: sub_view.byte_len(),
                    buffer_len: sub_view.byte_len(),
                    reason: format!(
                        "unsupported quant {:?} or unexpected byte length {} for {} elements",
                        sub_view.quant(),
                        raw.len(),
                        expected
                    ),
                });
            }
        };

        let packed = pack_tiles(&i8_vec, p_rows, k);
        results.push(packed);
    }

    Ok(results)
}

/// Repacks a 3D stacked `MoE` expert tensor (`ffn_{gate,up,down}_exps.weight`) into
/// contiguous per-expert `8x16` `i8` IME tiles aligned to 64-byte boundaries.
///
/// # Errors
/// Returns [`K3Error::TensorBoundsError`] if:
/// - `num_experts == 0`,
/// - `stacked_exps` fails 3D expert slicing via [`TensorView::slice_expert`],
/// - each expert's column dimension `k` is not a multiple of `TILE_K` (`16`),
/// - or `stacked_exps` uses an unsupported quantization layout or unexpected byte length.
pub fn repack_stacked_moe_experts_ime_8x16(
    stacked_exps: &TensorView<'_>,
    num_experts: usize,
) -> Result<Vec<u8>, K3Error> {
    const EXPERT_ALIGN_BYTES: usize = 64;

    if num_experts == 0 {
        return Err(K3Error::TensorBoundsError {
            tensor_name: stacked_exps.name().to_owned(),
            offset: stacked_exps.offset(),
            byte_len: stacked_exps.byte_len(),
            buffer_len: stacked_exps.byte_len(),
            reason: "num_experts must be greater than zero".to_owned(),
        });
    }

    let mut out = Vec::new();
    for expert_idx in 0..num_experts {
        let expert_view = stacked_exps.slice_expert(expert_idx, num_experts)?;
        let shape = expert_view.shape();
        let (k, rows) = (shape[0], shape[1]);
        if k % TILE_K != 0 {
            return Err(K3Error::TensorBoundsError {
                tensor_name: expert_view.name().to_owned(),
                offset: expert_view.offset(),
                byte_len: expert_view.byte_len(),
                buffer_len: expert_view.byte_len(),
                reason: format!(
                    "expert column dimension {k} must be a multiple of TILE_K ({TILE_K})"
                ),
            });
        }

        let i8_vec: Vec<i8> = if expert_view.quant() == QuantType::Q8_0 {
            let mut v = Vec::with_capacity(rows * k);
            let row_blocks = k / 32;
            let raw = expert_view.as_bytes();
            for row in 0..rows {
                let row_start_byte = row * row_blocks * 34;
                for block in 0..row_blocks {
                    let block_start = row_start_byte + block * 34;
                    let qs = &raw[block_start + 2..block_start + 34];
                    v.extend(qs.iter().map(|&b| i8::from_ne_bytes([b])));
                }
            }
            v
        } else {
            let raw = expert_view.as_bytes();
            let expected = rows * k;
            if raw.len() == expected {
                raw.iter().map(|&b| i8::from_ne_bytes([b])).collect()
            } else {
                return Err(K3Error::TensorBoundsError {
                    tensor_name: expert_view.name().to_owned(),
                    offset: expert_view.offset(),
                    byte_len: expert_view.byte_len(),
                    buffer_len: expert_view.byte_len(),
                    reason: format!(
                        "unsupported quant {:?} or unexpected byte length {} for {} elements",
                        expert_view.quant(),
                        raw.len(),
                        expected
                    ),
                });
            }
        };

        let packed = pack_tiles(&i8_vec, rows, k);
        out.reserve(packed.len() + EXPERT_ALIGN_BYTES);
        out.extend(packed.iter().map(|&x| x.to_ne_bytes()[0]));
        let padded_len = out.len().next_multiple_of(EXPERT_ALIGN_BYTES);
        out.resize(padded_len, 0);
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "row index in test fits in u8"
    )]
    fn test_repack_phi3_fused_qkv_pre_split() {
        // Fused QKV tensor with shape [32 cols, 8 rows]:
        // Q: 4 rows, K: 2 rows, V: 2 rows (sum = 8 rows)
        let cols = 32;
        let total_rows = 8;
        let shape = [cols, total_rows];
        let mut q8_raw = vec![0_u8; 34 * total_rows];
        for r in 0..total_rows {
            let start = r * 34;
            // f16 scale = 1.0 (0x3C00)
            q8_raw[start..start + 2].copy_from_slice(&[0x00, 0x3C]);
            // 32 i8 values
            q8_raw[start + 2..start + 34].fill((r as u8) + 1);
        }

        let q8_view =
            TensorView::from_bytes("blk.0.attn_qkv.weight", QuantType::Q8_0, &shape, 0, &q8_raw)
                .expect("valid Q8_0 view");

        let partitions = [4, 2, 2];
        let repacked = repack_phi3_fused_tensor_ime_8x16(&q8_view, &partitions)
            .expect("repack fused qkv succeeds");

        assert_eq!(repacked.len(), 3);
        // Q partition: 4 rows padded to 8 rows * (32 / 16 = 2 chunks) * 128 bytes = 256 bytes
        assert_eq!(repacked[0].len(), 256);
        // K partition: 2 rows padded to 8 rows * 2 chunks * 128 bytes = 256 bytes
        assert_eq!(repacked[1].len(), 256);
        // V partition: 2 rows padded to 8 rows * 2 chunks * 128 bytes = 256 bytes
        assert_eq!(repacked[2].len(), 256);

        // Verify first tile of Q partition contains marker 1 in first 16 bytes
        assert_eq!(repacked[0][0], 1);
        // Verify K partition contains marker 5 (row 4)
        assert_eq!(repacked[1][0], 5);
        // Verify V partition contains marker 7 (row 6)
        assert_eq!(repacked[2][0], 7);
    }

    #[test]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "row index in test fits in u8"
    )]
    fn test_repack_phi3_fused_gate_up_pre_split() {
        // Fused Gate-Up tensor: Gate (6 rows), Up (6 rows), total = 12 rows, cols = 32
        let cols_32 = 32;
        let total_rows = 12;
        let shape = [cols_32, total_rows];
        let mut q8_raw = vec![0_u8; 34 * total_rows];
        for r in 0..total_rows {
            let start = r * 34;
            q8_raw[start..start + 2].copy_from_slice(&[0x00, 0x3C]);
            q8_raw[start + 2..start + 34].fill((r as u8) + 10);
        }

        let q8_view =
            TensorView::from_bytes("blk.0.ffn_up.weight", QuantType::Q8_0, &shape, 0, &q8_raw)
                .expect("valid Q8_0 view");

        let partitions = [6, 6];
        let repacked = repack_phi3_fused_tensor_ime_8x16(&q8_view, &partitions)
            .expect("repack gate-up succeeds");

        assert_eq!(repacked.len(), 2);
        // 6 rows padded to 8 rows * 2 chunks * 128 = 256 bytes per partition
        assert_eq!(repacked[0].len(), 256);
        assert_eq!(repacked[1].len(), 256);
        assert_eq!(repacked[0][0], 10);
        assert_eq!(repacked[1][0], 16);
    }

    #[test]
    fn test_repack_phi3_fused_rejections() {
        let cols = 32;
        let total_rows = 8;
        let shape = [cols, total_rows];
        let q8_raw = vec![0_u8; 34 * total_rows];
        let q8_view =
            TensorView::from_bytes("blk.0.attn_qkv.weight", QuantType::Q8_0, &shape, 0, &q8_raw)
                .unwrap();

        // 1. Partition rows sum mismatch
        let err_sum = repack_phi3_fused_tensor_ime_8x16(&q8_view, &[4, 2])
            .expect_err("partition sum mismatch");
        assert!(matches!(err_sum, K3Error::TensorBoundsError { .. }));
        assert!(
            err_sum
                .to_string()
                .contains("partition rows sum (6) does not match")
        );

        // 2. Column dimension not divisible by TILE_K (16)
        // A 1D tensor
        let shape_1d = [32];
        let raw_1d = vec![0u8; 128];
        let view_1d = TensorView::from_bytes("1d", QuantType::F32, &shape_1d, 0, &raw_1d).unwrap();
        let err_1d = repack_phi3_fused_tensor_ime_8x16(&view_1d, &[32]).expect_err("requires 2D");
        assert!(matches!(err_1d, K3Error::TensorBoundsError { .. }));
    }

    #[test]
    fn test_qwen3moe_derived_blob_alignment() {
        let num_experts = 128;
        let rows = 8;
        let cols = 32; // Q8_0 block aligned & TILE_K=16 aligned
        let block_bytes = 34;
        let expert_bytes = rows * (cols / 32) * block_bytes;
        let mut raw = vec![0_u8; num_experts * expert_bytes];
        for e in 0..num_experts {
            let base = e * expert_bytes;
            for r in 0..rows {
                let row_off = base + r * block_bytes;
                raw[row_off..row_off + 2].copy_from_slice(&0x3C00_u16.to_le_bytes());
                let val = u8::try_from((e + r) % 120 + 1).expect("fits u8");
                raw[row_off + 2..row_off + 34].fill(val);
            }
        }

        let shape_3d = [cols, rows, num_experts];
        let view = TensorView::from_bytes(
            "blk.0.ffn_gate_exps.weight",
            QuantType::Q8_0,
            &shape_3d,
            0,
            &raw,
        )
        .expect("3D Q8_0 view");

        let packed_blob =
            repack_stacked_moe_experts_ime_8x16(&view, num_experts).expect("repack 128 experts");
        let bytes_per_expert = rows.div_ceil(8) * (cols / 16) * 128; // 256 bytes, multiple of 64B and 128B
        assert_eq!(bytes_per_expert % 64, 0);
        assert_eq!(bytes_per_expert % 128, 0);
        assert_eq!(packed_blob.len(), num_experts * bytes_per_expert);
        for e in 0..num_experts {
            let off = e * bytes_per_expert;
            assert_eq!(off % 64, 0);
            let expected_first = u8::try_from(e % 120 + 1).expect("fits u8");
            assert_eq!(packed_blob[off], expected_first);
        }

        // Rejection guards
        let err_zero =
            repack_stacked_moe_experts_ime_8x16(&view, 0).expect_err("reject zero experts");
        assert!(matches!(err_zero, K3Error::TensorBoundsError { .. }));

        let shape_2d = [cols, rows];
        let view_2d =
            TensorView::from_bytes("2d", QuantType::Q8_0, &shape_2d, 0, &raw[..expert_bytes])
                .expect("2D view");
        let err_2d = repack_stacked_moe_experts_ime_8x16(&view_2d, 1).expect_err("reject 2D");
        assert!(matches!(err_2d, K3Error::TensorBoundsError { .. }));
    }

    #[test]
    fn test_repack_stacked_moe_experts_contiguous() {
        let num_experts = 4;
        let rows = 8;
        let cols = 32;
        let expert_elems = rows * cols;
        let expert_bytes = rows * 34;

        // Standard GGUF [cols, rows, num_experts] ordering
        let mut raw_col_major = vec![0_u8; num_experts * expert_bytes];
        let mut expected_all = Vec::new();
        for e in 0..num_experts {
            let mut expert_i8 = Vec::with_capacity(expert_elems);
            for r in 0..rows {
                let row_off = e * expert_bytes + r * 34;
                raw_col_major[row_off..row_off + 2].copy_from_slice(&0x3C00_u16.to_le_bytes());
                for k in 0..cols {
                    let q = i8::try_from(((e * 17 + r * 3 + k) % 97) + 1).expect("fits i8");
                    raw_col_major[row_off + 2 + k] = q.to_ne_bytes()[0];
                    expert_i8.push(q);
                }
            }
            let packed = pack_tiles(&expert_i8, rows, cols);
            expected_all.extend(packed.iter().map(|&v| v.to_ne_bytes()[0]));
        }

        let shape_gguf = [cols, rows, num_experts];
        let view_gguf = TensorView::from_bytes(
            "blk.1.ffn_down_exps.weight",
            QuantType::Q8_0,
            &shape_gguf,
            0,
            &raw_col_major,
        )
        .expect("gguf 3D view");
        let actual_gguf =
            repack_stacked_moe_experts_ime_8x16(&view_gguf, num_experts).expect("repack gguf 3D");
        assert_eq!(actual_gguf, expected_all);

        // Synthetic [num_experts, rows, cols] ordering when shape[2] != num_experts
        let shape_synth = [num_experts, rows, cols];
        let view_synth = TensorView::from_bytes(
            "blk.1.ffn_up_exps.weight",
            QuantType::Q8_0,
            &shape_synth,
            0,
            &raw_col_major,
        )
        .expect("synthetic 3D view");
        let actual_synth =
            repack_stacked_moe_experts_ime_8x16(&view_synth, num_experts).expect("repack synth 3D");
        assert_eq!(actual_synth, expected_all);
    }

    #[test]
    fn test_qwen3moe_expert_tile_alignment() {
        let num_experts = 128;
        let rows = 10; // spans 2 TILE_M=8 tiles with zero-padded rows 10..16
        let cols = 32;
        let block_bytes = 34;
        let expert_bytes = rows * block_bytes;
        let mut raw = vec![0_u8; num_experts * expert_bytes];
        for e in 0..num_experts {
            for r in 0..rows {
                let row_off = e * expert_bytes + r * block_bytes;
                raw[row_off..row_off + 2].copy_from_slice(&0x3C00_u16.to_le_bytes());
                let marker = i8::try_from((e % 50) + 1).expect("fits i8").to_ne_bytes()[0];
                raw[row_off + 2..row_off + 34].fill(marker);
            }
        }

        let shape = [cols, rows, num_experts];
        let view =
            TensorView::from_bytes("blk.2.ffn_up_exps.weight", QuantType::Q8_0, &shape, 0, &raw)
                .expect("qwen3moe 3D view");
        let packed =
            repack_stacked_moe_experts_ime_8x16(&view, num_experts).expect("repack tiled experts");

        // tile_count(10) = 2 row tiles, k_chunks = 2, tile bytes = 128 => 512 bytes/expert
        let expected_stride = 2 * 2 * 128;
        assert_eq!(expected_stride % 64, 0);
        assert_eq!(packed.len(), num_experts * expected_stride);
        for e in 0..num_experts {
            let expert_slice = &packed[e * expected_stride..(e + 1) * expected_stride];
            // Rows 10..15 in the second row tile (tile offset 256) must be zero-padded
            let tile1_k0 = &expert_slice[256..256 + 128];
            assert!(tile1_k0[2 * 16..].iter().all(|&b| b == 0));
        }
    }
}

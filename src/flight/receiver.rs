//! Arrow Flight Receiver & Anonymous Memfd Arena Population (US-19, ADR-037).
//!
//! Receives streamed Apache Arrow Flight tensor batches, writes them directly into
//! an anonymous shared memory arena (`memfd_create`), locks pages in physical RAM (`mlock()`),
//! applies `F_SEAL_WRITE` sealing, and hands off the sealed descriptor to worker processes
//! over Unix Domain Sockets using `SCM_RIGHTS`.

#![allow(
    clippy::pedantic,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::missing_panics_doc,
    clippy::missing_errors_doc,
    clippy::doc_markdown,
    clippy::needless_pass_by_value
)]

use std::os::fd::RawFd;
use std::os::unix::net::UnixStream;

use arrow_array::types::{
    Float16Type, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type,
    UInt16Type, UInt32Type, UInt64Type,
};
use arrow_array::{
    Array, ArrowPrimitiveType, BinaryArray, BooleanArray, PrimitiveArray, RecordBatch, UInt64Array,
};
use arrow_schema::DataType;
use jero_core::K3Error;

use super::memfd::{FlightError, MemfdArena};

/// Stream container for incoming Flight tensor batches.
#[derive(Debug, Clone)]
pub struct FlightStream {
    /// Ordered list of Arrow tensor batches.
    pub batches: Vec<RecordBatch>,
    /// Total expected memory footprint in bytes.
    pub total_bytes: usize,
}

impl FlightStream {
    /// Creates a new flight stream wrapping a collection of batches.
    #[must_use]
    pub fn new(batches: Vec<RecordBatch>) -> Self {
        let mut max_extent = 0;
        let mut sum_lengths = 0;
        for batch in &batches {
            if batch.num_columns() >= 6 {
                let offsets = batch.column(3).as_any().downcast_ref::<UInt64Array>();
                let weights = batch.column(5).as_any().downcast_ref::<BinaryArray>();

                if let (Some(offsets), Some(weights)) = (offsets, weights) {
                    let rows = offsets.len().min(weights.len());
                    for i in 0..rows {
                        let offset = offsets.value(i) as usize;
                        let len = weights.value(i).len();
                        sum_lengths += len;
                        let extent = offset.saturating_add(len);
                        if extent > max_extent {
                            max_extent = extent;
                        }
                    }
                }
            }
        }
        let total_bytes = max_extent.max(sum_lengths);
        Self {
            batches,
            total_bytes,
        }
    }
}

/// Node bridge receiver that populates anonymous `memfd` arenas from Arrow Flight batches.
#[derive(Debug)]
pub struct FlightReceiver {
    arena: MemfdArena,
    bytes_written: usize,
}

impl FlightReceiver {
    /// Creates a new receiver with an anonymous `memfd` sized to `capacity_bytes`.
    ///
    /// # Errors
    /// Returns [`FlightError::AllocationFailed`] if arena creation fails.
    pub fn new(name: &str, capacity_bytes: usize) -> Result<Self, FlightError> {
        let arena = MemfdArena::new(name, capacity_bytes)?;
        Ok(Self {
            arena,
            bytes_written: 0,
        })
    }

    /// Returns a reference to the underlying `MemfdArena`.
    #[must_use]
    pub fn arena(&self) -> &MemfdArena {
        &self.arena
    }

    /// Returns a mutable reference to the underlying `MemfdArena`.
    pub fn arena_mut(&mut self) -> &mut MemfdArena {
        &mut self.arena
    }

    /// Writes an incoming Arrow [`RecordBatch`] directly into the anonymous `memfd`.
    ///
    /// # Invariant
    /// Extracts `byte_offset` and raw `weights` binary data and copies directly
    /// into the shared memory segment without intermediate allocations.
    ///
    /// # Errors
    /// Returns [`FlightError`] if parsing or writing fails.
    pub fn write_batch(&mut self, batch: &RecordBatch) -> Result<usize, FlightError> {
        let num_rows = batch.num_rows();
        if num_rows == 0 {
            return Ok(0);
        }

        if batch.num_columns() < 6 {
            return Err(FlightError::InvalidBatch(format!(
                "record batch requires at least 6 columns, got {}",
                batch.num_columns()
            )));
        }

        let offsets = batch
            .column(3)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| FlightError::InvalidBatch("column 3 is not UInt64Array".to_owned()))?;

        let weights = batch
            .column(5)
            .as_any()
            .downcast_ref::<BinaryArray>()
            .ok_or_else(|| FlightError::InvalidBatch("column 5 is not BinaryArray".to_owned()))?;

        let mut batch_bytes = 0;

        for i in 0..num_rows {
            let offset = offsets.value(i) as usize;
            let slice = weights.value(i);

            self.arena.write_at(offset, slice)?;
            batch_bytes += slice.len();
        }

        self.bytes_written += batch_bytes;
        Ok(batch_bytes)
    }

    /// Consumes the receiver and returns the underlying open `RawFd`.
    #[must_use]
    pub fn into_raw_fd(self) -> RawFd {
        self.arena.into_raw_fd()
    }

    /// Locks arena memory in RAM (`mlock`) and applies `F_SEAL_WRITE`.
    ///
    /// # Errors
    /// Returns [`FlightError`] if memlock ceiling is breached or sealing fails.
    pub fn finalize(&mut self, ceiling_bytes: Option<u64>) -> Result<(), FlightError> {
        // 1. Lock physical pages in RAM
        self.arena.mlock(ceiling_bytes)?;

        // 2. Apply write seal
        self.arena.seal_write()?;

        Ok(())
    }

    /// Transfers the sealed `memfd` descriptor to a compute worker over UDS via `SCM_RIGHTS`.
    ///
    /// # Errors
    /// Returns [`FlightError::UdsError`] on transfer failure.
    pub fn transfer_to_worker(
        &self,
        stream: &UnixStream,
        model_id: &str,
    ) -> Result<(), FlightError> {
        self.arena.transfer_via_uds(stream, model_id.as_bytes())
    }

    /// Implements the `PROJECT.md` interface contract:
    /// `FlightReceiver::receive_to_memfd(flight_stream: FlightStream) -> Result<RawFd, K3Error>`
    ///
    /// Receives all tensor batches, writes them directly into a sealed `memfd`,
    /// calls `mlock()`, applies `F_SEAL_WRITE`, and returns the raw file descriptor.
    ///
    /// # Errors
    /// Returns [`K3Error`] on failure.
    pub fn receive_to_memfd(flight_stream: FlightStream) -> Result<RawFd, K3Error> {
        let mut max_extent = flight_stream.total_bytes;
        for batch in &flight_stream.batches {
            if batch.num_columns() >= 6
                && let (Some(offsets), Some(weights)) = (
                    batch.column(3).as_any().downcast_ref::<UInt64Array>(),
                    batch.column(5).as_any().downcast_ref::<BinaryArray>(),
                )
            {
                let rows = offsets.len().min(weights.len());
                for i in 0..rows {
                    let extent = (offsets.value(i) as usize).saturating_add(weights.value(i).len());
                    if extent > max_extent {
                        max_extent = extent;
                    }
                }
            }
        }

        let capacity = max_extent.max(4096);
        let mut receiver = Self::new("flight_received_arena", capacity).map_err(K3Error::from)?;

        for batch in &flight_stream.batches {
            receiver.write_batch(batch).map_err(K3Error::from)?;
        }

        receiver.finalize(None).map_err(K3Error::from)?;

        Ok(receiver.into_raw_fd())
    }
}

/// Row-major `f32` feature matrix extracted from an Apache Arrow [`RecordBatch`].
#[derive(Debug, Clone, PartialEq)]
pub struct ExtractedFeatureMatrix {
    /// Row-major tensor values with shape `[num_rows, num_cols]`.
    pub values: Vec<f32>,
    /// Number of feature rows.
    pub num_rows: usize,
    /// Number of feature columns.
    pub num_cols: usize,
    /// Schema column names in extraction order.
    pub column_names: Vec<String>,
}

fn extract_primitive_column<T: ArrowPrimitiveType>(
    col: &dyn Array,
    col_name: &str,
    convert: fn(T::Native) -> Option<f32>,
) -> Result<Vec<f32>, FlightError> {
    let arr = col
        .as_any()
        .downcast_ref::<PrimitiveArray<T>>()
        .ok_or_else(|| {
            FlightError::InvalidBatch(format!(
                "column '{col_name}' failed primitive downcast for {:?}",
                col.data_type()
            ))
        })?;
    let mut out = Vec::with_capacity(arr.len());
    for (row_idx, &raw) in arr.values().iter().enumerate() {
        let converted = convert(raw).filter(|v| v.is_finite()).ok_or_else(|| {
            FlightError::InvalidBatch(format!(
                "column '{col_name}' row {row_idx} produced non-finite or overflowed f32 value"
            ))
        })?;
        out.push(converted);
    }
    Ok(out)
}

fn extract_numeric_column(col: &dyn Array, col_name: &str) -> Result<Vec<f32>, FlightError> {
    if col.null_count() > 0 {
        return Err(FlightError::InvalidBatch(format!(
            "column '{col_name}' contains {} null value(s); machine feature tensors must be dense",
            col.null_count()
        )));
    }

    match col.data_type() {
        DataType::Float32 => extract_primitive_column::<Float32Type>(col, col_name, Some),
        DataType::Float64 => extract_primitive_column::<Float64Type>(col, col_name, |v| {
            if v.is_finite() { Some(v as f32) } else { None }
        }),
        DataType::Float16 => {
            extract_primitive_column::<Float16Type>(col, col_name, |v| Some(v.to_f32()))
        }
        DataType::Int8 => {
            extract_primitive_column::<Int8Type>(col, col_name, |v| Some(f32::from(v)))
        }
        DataType::Int16 => {
            extract_primitive_column::<Int16Type>(col, col_name, |v| Some(f32::from(v)))
        }
        DataType::Int32 => extract_primitive_column::<Int32Type>(col, col_name, |v| Some(v as f32)),
        DataType::Int64 => extract_primitive_column::<Int64Type>(col, col_name, |v| Some(v as f32)),
        DataType::UInt8 => {
            extract_primitive_column::<UInt8Type>(col, col_name, |v| Some(f32::from(v)))
        }
        DataType::UInt16 => {
            extract_primitive_column::<UInt16Type>(col, col_name, |v| Some(f32::from(v)))
        }
        DataType::UInt32 => {
            extract_primitive_column::<UInt32Type>(col, col_name, |v| Some(v as f32))
        }
        DataType::UInt64 => {
            extract_primitive_column::<UInt64Type>(col, col_name, |v| Some(v as f32))
        }
        DataType::Boolean => {
            let arr = col.as_any().downcast_ref::<BooleanArray>().ok_or_else(|| {
                FlightError::InvalidBatch(format!(
                    "column '{col_name}' failed BooleanArray downcast"
                ))
            })?;
            Ok((0..arr.len())
                .map(|row_idx| if arr.value(row_idx) { 1.0 } else { 0.0 })
                .collect())
        }
        other => Err(FlightError::InvalidBatch(format!(
            "column '{col_name}' has unsupported Arrow DataType {other:?} for feature matrix extraction"
        ))),
    }
}

/// Validates an Apache Arrow [`RecordBatch`] and extracts its numeric columns into a row-major
/// dense `f32` matrix `[num_rows, num_cols]`.
///
/// Enforces fail-closed validation against empty batches, column schema mismatches (when
/// `expected_columns` is provided), null entries, unsupported non-numeric types, and `NaN`/`±Inf`.
pub fn extract_feature_matrix_from_record_batch(
    batch: &RecordBatch,
    expected_columns: Option<&[&str]>,
) -> Result<ExtractedFeatureMatrix, FlightError> {
    let num_rows = batch.num_rows();
    let num_cols = batch.num_columns();
    if num_rows == 0 || num_cols == 0 {
        return Err(FlightError::InvalidBatch(format!(
            "feature RecordBatch must be non-empty, got shape [{num_rows}, {num_cols}]"
        )));
    }

    let schema = batch.schema();
    if let Some(expected) = expected_columns {
        if expected.len() != num_cols {
            return Err(FlightError::InvalidBatch(format!(
                "feature column count mismatch: expected {}, got {}",
                expected.len(),
                num_cols
            )));
        }
        for (idx, &expected_name) in expected.iter().enumerate() {
            let actual_name = schema.field(idx).name();
            if actual_name != expected_name {
                return Err(FlightError::InvalidBatch(format!(
                    "feature column mismatch at index {idx}: expected '{expected_name}', got '{actual_name}'"
                )));
            }
        }
    }

    let mut column_names = Vec::with_capacity(num_cols);
    let mut column_data = Vec::with_capacity(num_cols);
    for col_idx in 0..num_cols {
        let col_name = schema.field(col_idx).name().clone();
        let extracted = extract_numeric_column(batch.column(col_idx).as_ref(), &col_name)?;
        column_names.push(col_name);
        column_data.push(extracted);
    }

    let total_elems = num_rows.checked_mul(num_cols).ok_or_else(|| {
        FlightError::InvalidBatch("feature matrix dimensions overflow".to_owned())
    })?;
    let mut values = vec![0.0_f32; total_elems];
    for (col_idx, col_vals) in column_data.iter().enumerate() {
        for (row_idx, &val) in col_vals.iter().enumerate() {
            values[row_idx * num_cols + col_idx] = val;
        }
    }

    Ok(ExtractedFeatureMatrix {
        values,
        num_rows,
        num_cols,
        column_names,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Float32Array, Float64Array, Int32Array, StringArray};
    use arrow_schema::{Field, Schema};

    use super::*;

    #[test]
    fn test_extract_feature_matrix_from_record_batch_mixed_types_and_fail_closed() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("cpu_util", DataType::Float32, false),
            Field::new("mem_ratio", DataType::Float64, false),
            Field::new("queue_len", DataType::Int32, false),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Float32Array::from(vec![0.25, 0.90])),
                Arc::new(Float64Array::from(vec![0.40, 0.85])),
                Arc::new(Int32Array::from(vec![3, 19])),
            ],
        )
        .expect("build batch");

        let extracted = extract_feature_matrix_from_record_batch(
            &batch,
            Some(&["cpu_util", "mem_ratio", "queue_len"]),
        )
        .expect("extract matrix");
        assert_eq!(extracted.num_rows, 2);
        assert_eq!(extracted.num_cols, 3);
        assert_eq!(
            extracted.column_names,
            vec!["cpu_util", "mem_ratio", "queue_len"]
        );
        assert_eq!(extracted.values, vec![0.25, 0.40, 3.0, 0.90, 0.85, 19.0]);

        // Schema name mismatch rejected
        assert!(
            extract_feature_matrix_from_record_batch(
                &batch,
                Some(&["cpu_util", "wrong_col", "queue_len"])
            )
            .is_err()
        );

        // NaN value fails closed
        let nan_batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float32Array::from(vec![0.25, f32::NAN])),
                Arc::new(Float64Array::from(vec![0.40, 0.85])),
                Arc::new(Int32Array::from(vec![3, 19])),
            ],
        )
        .expect("build nan batch");
        assert!(extract_feature_matrix_from_record_batch(&nan_batch, None).is_err());

        // Unsupported string column rejected
        let str_schema = Arc::new(Schema::new(vec![Field::new("tag", DataType::Utf8, false)]));
        let str_batch =
            RecordBatch::try_new(str_schema, vec![Arc::new(StringArray::from(vec!["bad"]))])
                .expect("build string batch");
        assert!(extract_feature_matrix_from_record_batch(&str_batch, None).is_err());
    }
}

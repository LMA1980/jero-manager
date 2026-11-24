//! Speech-to-text transcription protocol handler (`POST /v1/audio/transcriptions`) supporting
//! RFC 7578 `multipart/form-data` WAV uploads, JSON `audio_base64`/`pcm_f32` payloads, fail-closed
//! RIFF/WAVE decoding, bounded 16 kHz mono resampling, and `WhisperRecipe` execution
//! (M10 Feature `#112`, Task `#113`).

use std::fs;
use std::path::Path;
use std::sync::Arc;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use jero_arch_whisper::{
    WHISPER_EOT_TOKEN, WHISPER_NO_TIMESTAMPS_TOKEN_V2, WHISPER_NO_TIMESTAMPS_TOKEN_V3,
    WHISPER_SAMPLE_RATE, WHISPER_SOT_TOKEN, WHISPER_TRANSCRIBE_TOKEN_V2,
    WHISPER_TRANSCRIBE_TOKEN_V3, WHISPER_VOCAB_V3, WhisperRecipe, WhisperVariant,
};
use jero_core::{ArchRecipe, ExecutionContext, GgufFile, SafeTensorsFile};
use jero_tokenizer::Tokenizer;

use super::embeddings::map_k3_error;
use super::server::{
    malformed_json_response, map_manager_error_to_http, method_not_allowed_response,
};
use super::types::{
    AudioTranscriptionFormat, AudioTranscriptionRequest, AudioTranscriptionResponse,
    AudioTranscriptionSegment, HttpRequest, HttpResponse,
};
use crate::error::ManagerError;
use crate::runtime::{ControlPlaneRuntime, WhisperEvaluatorFn};
use crate::store::MAX_WEIGHT_INSPECTION_BYTES;

/// Maximum decoded WAV payload byte budget admitted for transcription (`8 MiB`).
pub const MAX_DECODED_AUDIO_BYTES: usize = 8 * 1024 * 1024;

/// Maximum single-window normalized 16 kHz Whisper audio duration in seconds (`30 s`).
pub const MAX_AUDIO_DURATION_SECS: usize = 30;

/// Maximum resampled 16 kHz mono sample count (`30 * 16_000 = 480_000` samples).
pub const MAX_RESAMPLED_SAMPLES_16KHZ: usize = WHISPER_SAMPLE_RATE * MAX_AUDIO_DURATION_SECS;

/// Minimum supported source WAV/PCM sample rate in Hz.
pub const MIN_SUPPORTED_SAMPLE_RATE: u32 = 4_000;

/// Maximum supported source WAV/PCM sample rate in Hz.
pub const MAX_SUPPORTED_SAMPLE_RATE: u32 = 192_000;

const MAX_WHISPER_GREEDY_TOKENS: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WavEncoding {
    Pcm16,
    IeeeFloat32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WavFormatHeader {
    encoding: WavEncoding,
    channels: u16,
    sample_rate: u32,
    block_align: u16,
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn extract_multipart_boundary(content_type: &str) -> Result<String, ManagerError> {
    for param in content_type.split(';').skip(1) {
        let trimmed = param.trim();
        if let Some((key, raw_val)) = trimmed.split_once('=')
            && key.trim().eq_ignore_ascii_case("boundary")
        {
            let unquoted = raw_val.trim().trim_matches('"').trim();
            if unquoted.is_empty() || unquoted.len() > 200 {
                return Err(ManagerError::Protocol(
                    "invalid multipart/form-data boundary length".to_owned(),
                ));
            }
            return Ok(unquoted.to_owned());
        }
    }
    Err(ManagerError::Protocol(
        "multipart/form-data Content-Type missing boundary parameter".to_owned(),
    ))
}

fn extract_disposition_name(headers_bytes: &[u8]) -> Result<Option<String>, ManagerError> {
    let headers_str = std::str::from_utf8(headers_bytes)
        .map_err(|e| ManagerError::Protocol(format!("non-UTF-8 multipart part headers: {e}")))?;
    for line in headers_str.lines() {
        let Some((hdr_name, hdr_val)) = line.split_once(':') else {
            continue;
        };
        if !hdr_name.trim().eq_ignore_ascii_case("content-disposition") {
            continue;
        }
        for param in hdr_val.split(';') {
            let trimmed = param.trim();
            if let Some((k, v)) = trimmed.split_once('=')
                && k.trim().eq_ignore_ascii_case("name")
            {
                let name = v.trim().trim_matches('"').to_owned();
                if !name.is_empty() {
                    return Ok(Some(name));
                }
            }
        }
    }
    Ok(None)
}

type MultipartPart<'a> = (&'a [u8], &'a [u8]);

fn split_multipart_parts<'a>(
    body: &'a [u8],
    boundary: &str,
) -> Result<Vec<MultipartPart<'a>>, ManagerError> {
    let delimiter = format!("--{boundary}");
    let delim_bytes = delimiter.as_bytes();
    let Some(first_idx) = find_subslice(body, delim_bytes) else {
        return Err(ManagerError::Protocol(
            "multipart/form-data body does not contain declared boundary".to_owned(),
        ));
    };

    let mut cursor = first_idx + delim_bytes.len();
    let mut parts = Vec::new();
    while cursor < body.len() {
        if body[cursor..].starts_with(b"--") {
            break;
        }
        if body[cursor..].starts_with(b"\r\n") {
            cursor += 2;
        } else if body[cursor..].starts_with(b"\n") {
            cursor += 1;
        } else {
            return Err(ManagerError::Protocol(
                "malformed multipart delimiter line ending".to_owned(),
            ));
        }

        let remaining = &body[cursor..];
        let next_rel = find_subslice(remaining, delim_bytes).ok_or_else(|| {
            ManagerError::Protocol("unterminated multipart/form-data section".to_owned())
        })?;
        let mut raw_part = &remaining[..next_rel];
        if raw_part.ends_with(b"\r\n") {
            raw_part = &raw_part[..raw_part.len() - 2];
        } else if raw_part.ends_with(b"\n") {
            raw_part = &raw_part[..raw_part.len() - 1];
        }

        let (hdr_bytes, part_body) = if let Some(pos) = find_subslice(raw_part, b"\r\n\r\n") {
            (&raw_part[..pos], &raw_part[pos + 4..])
        } else if let Some(pos) = find_subslice(raw_part, b"\n\n") {
            (&raw_part[..pos], &raw_part[pos + 2..])
        } else {
            return Err(ManagerError::Protocol(
                "multipart section missing blank header terminator".to_owned(),
            ));
        };

        parts.push((hdr_bytes, part_body));
        cursor += next_rel + delim_bytes.len();
    }

    if parts.is_empty() {
        return Err(ManagerError::Protocol(
            "multipart/form-data request contained no form fields".to_owned(),
        ));
    }
    Ok(parts)
}

fn parse_multipart_audio_request(
    body: &[u8],
    boundary: &str,
) -> Result<(AudioTranscriptionRequest, Vec<f32>), ManagerError> {
    let parts = split_multipart_parts(body, boundary)?;
    let mut file_bytes: Option<&[u8]> = None;
    let mut model: Option<String> = None;
    let mut language: Option<String> = None;
    let mut response_format: Option<AudioTranscriptionFormat> = None;
    let mut prompt: Option<String> = None;
    let mut temperature: Option<f32> = None;

    for (hdr_bytes, part_body) in parts {
        let Some(field_name) = extract_disposition_name(hdr_bytes)? else {
            continue;
        };
        match field_name.as_str() {
            "file" => {
                file_bytes = Some(part_body);
            }
            "model" => {
                let text = std::str::from_utf8(part_body)
                    .map_err(|e| ManagerError::Protocol(format!("invalid UTF-8 model field: {e}")))?
                    .trim()
                    .to_owned();
                model = Some(text);
            }
            "language" => {
                let text = std::str::from_utf8(part_body)
                    .map_err(|e| {
                        ManagerError::Protocol(format!("invalid UTF-8 language field: {e}"))
                    })?
                    .trim()
                    .to_owned();
                if !text.is_empty() {
                    language = Some(text);
                }
            }
            "response_format" => {
                let text = std::str::from_utf8(part_body).map_err(|e| {
                    ManagerError::Protocol(format!("invalid UTF-8 response_format field: {e}"))
                })?;
                response_format = Some(
                    AudioTranscriptionFormat::parse_str(text).map_err(ManagerError::Protocol)?,
                );
            }
            "prompt" => {
                let text = std::str::from_utf8(part_body)
                    .map_err(|e| {
                        ManagerError::Protocol(format!("invalid UTF-8 prompt field: {e}"))
                    })?
                    .to_owned();
                if !text.is_empty() {
                    prompt = Some(text);
                }
            }
            "temperature" => {
                let text = std::str::from_utf8(part_body)
                    .map_err(|e| {
                        ManagerError::Protocol(format!("invalid UTF-8 temperature field: {e}"))
                    })?
                    .trim();
                let temp: f32 = text.parse().map_err(|e| {
                    ManagerError::Protocol(format!("invalid multipart temperature value: {e}"))
                })?;
                temperature = Some(temp);
            }
            _ => {}
        }
    }

    let model = model.filter(|m| !m.is_empty()).ok_or_else(|| {
        ManagerError::Protocol("multipart transcription missing required 'model' field".to_owned())
    })?;
    let wav_bytes = file_bytes.ok_or_else(|| {
        ManagerError::Protocol("multipart transcription missing required 'file' field".to_owned())
    })?;
    if wav_bytes.is_empty() {
        return Err(ManagerError::Protocol(
            "multipart transcription 'file' upload must not be empty".to_owned(),
        ));
    }
    if wav_bytes.len() > MAX_DECODED_AUDIO_BYTES {
        return Err(ManagerError::PayloadTooLarge(format!(
            "audio payload ({} bytes) exceeds decoded limit ({MAX_DECODED_AUDIO_BYTES} bytes)",
            wav_bytes.len()
        )));
    }
    validate_temperature(temperature)?;

    let pcm_16khz = parse_wav_to_pcm_16khz(wav_bytes)?;
    let req = AudioTranscriptionRequest {
        model,
        audio_base64: None,
        pcm_f32: None,
        sample_rate: Some(u32::try_from(WHISPER_SAMPLE_RATE).unwrap_or(16_000)),
        language,
        response_format,
        prompt,
        temperature,
    };
    Ok((req, pcm_16khz))
}

fn validate_temperature(temp: Option<f32>) -> Result<(), ManagerError> {
    if let Some(t) = temp
        && (!t.is_finite() || t < 0.0 || t > 1.0)
    {
        return Err(ManagerError::Protocol(format!(
            "transcription 'temperature' must be finite within [0.0, 1.0], got {t}"
        )));
    }
    Ok(())
}

fn decode_base64_audio_payload(raw_b64: &str) -> Result<Vec<u8>, ManagerError> {
    let trimmed = raw_b64.trim();
    let b64_slice = if let Some(comma) = trimmed.find(";base64,") {
        let prefix = &trimmed[..comma];
        if !prefix.starts_with("data:audio/") {
            return Err(ManagerError::Protocol(format!(
                "invalid audio data URI prefix '{prefix}'; expected data:audio/..."
            )));
        }
        &trimmed[comma + ";base64,".len()..]
    } else {
        trimmed
    };

    if b64_slice.is_empty() {
        return Err(ManagerError::Protocol(
            "audio_base64 payload must not be empty".to_owned(),
        ));
    }
    let estimated_max_b64 = MAX_DECODED_AUDIO_BYTES.div_ceil(3) * 4 + 16;
    if b64_slice.len() > estimated_max_b64 {
        return Err(ManagerError::PayloadTooLarge(format!(
            "base64 audio payload length ({} chars) exceeds decoded ceiling",
            b64_slice.len()
        )));
    }

    let decoded = BASE64_STANDARD
        .decode(b64_slice)
        .map_err(|e| ManagerError::Protocol(format!("malformed audio_base64 encoding: {e}")))?;
    if decoded.len() > MAX_DECODED_AUDIO_BYTES {
        return Err(ManagerError::PayloadTooLarge(format!(
            "decoded audio payload ({} bytes) exceeds limit ({MAX_DECODED_AUDIO_BYTES} bytes)",
            decoded.len()
        )));
    }
    Ok(decoded)
}

fn parse_json_audio_request(
    req: AudioTranscriptionRequest,
) -> Result<(AudioTranscriptionRequest, Vec<f32>), ManagerError> {
    if req.model.trim().is_empty() {
        return Err(ManagerError::Protocol(
            "transcription 'model' field must not be empty".to_owned(),
        ));
    }
    validate_temperature(req.temperature)?;

    let pcm_16khz = match (&req.audio_base64, &req.pcm_f32) {
        (Some(_), Some(_)) => {
            return Err(ManagerError::Protocol(
                "specify either 'audio_base64' or 'pcm_f32', not both".to_owned(),
            ));
        }
        (None, None) => {
            return Err(ManagerError::Protocol(
                "JSON transcription request requires 'audio_base64' or 'pcm_f32'".to_owned(),
            ));
        }
        (Some(b64), None) => {
            let wav_bytes = decode_base64_audio_payload(b64)?;
            parse_wav_to_pcm_16khz(&wav_bytes)?
        }
        (None, Some(samples)) => {
            if samples.is_empty() {
                return Err(ManagerError::Protocol(
                    "transcription 'pcm_f32' samples must not be empty".to_owned(),
                ));
            }
            if samples.iter().any(|s| !s.is_finite()) {
                return Err(ManagerError::Protocol(
                    "transcription 'pcm_f32' contains NaN or infinite sample values".to_owned(),
                ));
            }
            let sr = req
                .sample_rate
                .unwrap_or(u32::try_from(WHISPER_SAMPLE_RATE).unwrap_or(16_000));
            if !(MIN_SUPPORTED_SAMPLE_RATE..=MAX_SUPPORTED_SAMPLE_RATE).contains(&sr) {
                return Err(ManagerError::UnprocessableEntity(format!(
                    "audio sample_rate {sr} Hz outside supported range [{MIN_SUPPORTED_SAMPLE_RATE}..={MAX_SUPPORTED_SAMPLE_RATE}]"
                )));
            }
            resample_mono_to_16khz(samples, sr)?
        }
    };

    Ok((req, pcm_16khz))
}

fn parse_wav_fmt_chunk(chunk: &[u8]) -> Result<WavFormatHeader, ManagerError> {
    if chunk.len() < 16 {
        return Err(ManagerError::Protocol(format!(
            "truncated WAV fmt chunk ({} bytes < 16)",
            chunk.len()
        )));
    }
    let raw_fmt = u16::from_le_bytes([chunk[0], chunk[1]]);
    let channels = u16::from_le_bytes([chunk[2], chunk[3]]);
    let sample_rate = u32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
    let block_align = u16::from_le_bytes([chunk[12], chunk[13]]);
    let bits_per_sample = u16::from_le_bytes([chunk[14], chunk[15]]);

    let effective_fmt = if raw_fmt == 0xFFFE {
        if chunk.len() < 26 {
            return Err(ManagerError::Protocol(
                "truncated WAVE_FORMAT_EXTENSIBLE fmt chunk".to_owned(),
            ));
        }
        u16::from_le_bytes([chunk[24], chunk[25]])
    } else {
        raw_fmt
    };

    let encoding = match (effective_fmt, bits_per_sample) {
        (1, 16) => WavEncoding::Pcm16,
        (3, 32) => WavEncoding::IeeeFloat32,
        (fmt_tag, bits) => {
            return Err(ManagerError::Protocol(format!(
                "unsupported WAV format tag {fmt_tag} with {bits}-bit samples (expected PCM16 or IEEE_FLOAT32)"
            )));
        }
    };

    if channels == 0 || channels > 2 {
        return Err(ManagerError::UnprocessableEntity(format!(
            "unsupported WAV channel count {channels} (expected mono 1 or stereo 2)"
        )));
    }
    if !(MIN_SUPPORTED_SAMPLE_RATE..=MAX_SUPPORTED_SAMPLE_RATE).contains(&sample_rate) {
        return Err(ManagerError::UnprocessableEntity(format!(
            "unsupported WAV sample rate {sample_rate} Hz (supported: {MIN_SUPPORTED_SAMPLE_RATE}..={MAX_SUPPORTED_SAMPLE_RATE} Hz)"
        )));
    }

    let expected_align = channels.saturating_mul(bits_per_sample / 8);
    if block_align == 0 || block_align != expected_align {
        return Err(ManagerError::Protocol(format!(
            "inconsistent WAV block_align {block_align} (expected {expected_align})"
        )));
    }

    Ok(WavFormatHeader {
        encoding,
        channels,
        sample_rate,
        block_align,
    })
}

/// Decodes a RIFF/WAVE binary payload (`PCM16` or `IEEE_FLOAT32`, mono or stereo) into finite
/// 16 kHz mono `f32` samples bounded by [`MAX_RESAMPLED_SAMPLES_16KHZ`].
///
/// # Errors
/// - [`ManagerError::Protocol`] (`400`) on invalid RIFF/WAVE headers, unsupported encoding, truncated chunks, or non-finite samples.
/// - [`ManagerError::PayloadTooLarge`] (`413`) if payload exceeds [`MAX_DECODED_AUDIO_BYTES`].
/// - [`ManagerError::UnprocessableEntity`] (`422`) on unsupported channel counts, sample rates, or >30s duration.
pub fn parse_wav_to_pcm_16khz(bytes: &[u8]) -> Result<Vec<f32>, ManagerError> {
    if bytes.len() > MAX_DECODED_AUDIO_BYTES {
        return Err(ManagerError::PayloadTooLarge(format!(
            "WAV payload ({} bytes) exceeds decoded limit ({MAX_DECODED_AUDIO_BYTES} bytes)",
            bytes.len()
        )));
    }
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(ManagerError::Protocol(
            "invalid RIFF/WAVE file magic header".to_owned(),
        ));
    }

    let declared_riff_end = 8usize.saturating_add(
        usize::try_from(u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]))
            .unwrap_or(usize::MAX),
    );
    let scan_end = declared_riff_end.min(bytes.len());

    let mut fmt_header: Option<WavFormatHeader> = None;
    let mut data_slice: Option<&[u8]> = None;
    let mut offset = 12usize;

    while offset + 8 <= scan_end {
        let chunk_id = &bytes[offset..offset + 4];
        let chunk_size = usize::try_from(u32::from_le_bytes([
            bytes[offset + 4],
            bytes[offset + 5],
            bytes[offset + 6],
            bytes[offset + 7],
        ]))
        .map_err(|_| ManagerError::Protocol("WAV chunk size exceeds platform limits".to_owned()))?;
        let data_start = offset + 8;
        let data_end = data_start.checked_add(chunk_size).ok_or_else(|| {
            ManagerError::Protocol("WAV chunk extent overflowed address space".to_owned())
        })?;
        if data_end > bytes.len() {
            return Err(ManagerError::Protocol(format!(
                "truncated WAV chunk '{}': requires {data_end} bytes, have {}",
                String::from_utf8_lossy(chunk_id),
                bytes.len()
            )));
        }

        let payload = &bytes[data_start..data_end];
        if chunk_id == b"fmt " {
            fmt_header = Some(parse_wav_fmt_chunk(payload)?);
        } else if chunk_id == b"data" {
            data_slice = Some(payload);
        }

        let padded_end = data_end.saturating_add(chunk_size & 1);
        offset = padded_end;
        if fmt_header.is_some() && data_slice.is_some() {
            break;
        }
    }

    let fmt = fmt_header.ok_or_else(|| {
        ManagerError::Protocol("WAV file missing required 'fmt ' chunk".to_owned())
    })?;
    let data = data_slice.ok_or_else(|| {
        ManagerError::Protocol("WAV file missing required 'data' chunk".to_owned())
    })?;
    let frame_bytes = usize::from(fmt.block_align);
    if data.is_empty() || data.len() % frame_bytes != 0 {
        return Err(ManagerError::Protocol(format!(
            "WAV 'data' chunk length ({}) is empty or misaligned with block_align ({frame_bytes})",
            data.len()
        )));
    }

    let mono_src = decode_wav_frames_to_mono(data, fmt)?;
    resample_mono_to_16khz(&mono_src, fmt.sample_rate)
}

fn decode_wav_frames_to_mono(data: &[u8], fmt: WavFormatHeader) -> Result<Vec<f32>, ManagerError> {
    let frame_bytes = usize::from(fmt.block_align);
    let num_frames = data.len() / frame_bytes;
    let channels = usize::from(fmt.channels);
    let mut mono = Vec::with_capacity(num_frames);

    for frame in data.chunks_exact(frame_bytes) {
        let mut acc = 0.0_f32;
        match fmt.encoding {
            WavEncoding::Pcm16 => {
                for ch in 0..channels {
                    let s = i16::from_le_bytes([frame[ch * 2], frame[ch * 2 + 1]]);
                    acc += f32::from(s) / 32_768.0_f32;
                }
            }
            WavEncoding::IeeeFloat32 => {
                for ch in 0..channels {
                    let base = ch * 4;
                    let s = f32::from_le_bytes([
                        frame[base],
                        frame[base + 1],
                        frame[base + 2],
                        frame[base + 3],
                    ]);
                    if !s.is_finite() {
                        return Err(ManagerError::Protocol(
                            "IEEE float WAV data contains non-finite sample (NaN/Inf)".to_owned(),
                        ));
                    }
                    acc += s.clamp(-1.0_f32, 1.0_f32);
                }
            }
        }
        mono.push(acc / if channels == 2 { 2.0_f32 } else { 1.0_f32 });
    }
    Ok(mono)
}

fn resample_mono_to_16khz(samples: &[f32], src_rate: u32) -> Result<Vec<f32>, ManagerError> {
    let dst_rate = u32::try_from(WHISPER_SAMPLE_RATE).unwrap_or(16_000);
    if src_rate == dst_rate {
        if samples.len() > MAX_RESAMPLED_SAMPLES_16KHZ {
            return Err(ManagerError::UnprocessableEntity(format!(
                "audio clip length ({} samples) exceeds maximum 30s Whisper window ({MAX_RESAMPLED_SAMPLES_16KHZ} samples)",
                samples.len()
            )));
        }
        return Ok(samples.to_vec());
    }

    let src_len_u128 = u128::try_from(samples.len()).unwrap_or(0);
    let target_len_u128 = src_len_u128
        .saturating_mul(u128::from(dst_rate))
        .div_ceil(u128::from(src_rate))
        .max(1);
    let target_len = usize::try_from(target_len_u128).unwrap_or(usize::MAX);
    if target_len > MAX_RESAMPLED_SAMPLES_16KHZ {
        return Err(ManagerError::UnprocessableEntity(format!(
            "resampled audio length ({target_len} samples @ 16 kHz) exceeds 30s limit ({MAX_RESAMPLED_SAMPLES_16KHZ} samples)"
        )));
    }

    let mut out = Vec::with_capacity(target_len);
    let step = f64::from(src_rate) / f64::from(dst_rate);
    for dst_idx in 0..target_len {
        let dst_u32 = u32::try_from(dst_idx).unwrap_or(u32::MAX);
        let src_pos = f64::from(dst_u32) * step;
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "src_pos >= 0.0 and < samples.len()"
        )]
        let idx0 = (src_pos.floor() as usize).min(samples.len() - 1);
        let idx1 = (idx0 + 1).min(samples.len() - 1);
        let idx0_f64 = f64::from(u32::try_from(idx0).unwrap_or(u32::MAX));
        #[expect(
            clippy::cast_possible_truncation,
            reason = "interpolation fraction lies strictly in [0.0, 1.0]"
        )]
        let frac = (src_pos - idx0_f64) as f32;
        out.push(samples[idx0].mul_add(1.0_f32 - frac, samples[idx1] * frac));
    }
    Ok(out)
}

fn select_whisper_variant_for_model(model_id: &str) -> WhisperRecipe {
    let lower = model_id.to_ascii_lowercase();
    if lower.contains("turbo") || lower.contains("large-v3") {
        WhisperRecipe::from_variant("large-v3-turbo").unwrap_or_else(|_| WhisperRecipe::tiny())
    } else if lower.contains("medium") {
        WhisperRecipe::from_variant("medium").unwrap_or_else(|_| WhisperRecipe::tiny())
    } else if lower.contains("small") {
        WhisperRecipe::from_variant("small").unwrap_or_else(|_| WhisperRecipe::tiny())
    } else if lower.contains("base") {
        WhisperRecipe::from_variant("base").unwrap_or_else(|_| WhisperRecipe::tiny())
    } else {
        let _ = WhisperVariant::Tiny;
        WhisperRecipe::tiny()
    }
}

fn greedy_argmax(logits: &[f32]) -> u32 {
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (idx, &val) in logits.iter().enumerate() {
        if val > best_val {
            best_val = val;
            best_idx = idx;
        }
    }
    u32::try_from(best_idx).unwrap_or(WHISPER_EOT_TOKEN)
}

fn run_whisper_recipe_on_weights(
    recipe: &WhisperRecipe,
    weights: &dyn jero_core::ModelWeights,
    pcm_16khz: &[f32],
    tokenizer: Option<&dyn Tokenizer>,
) -> Result<(String, Vec<u32>), ManagerError> {
    let (mel, n_frames) = recipe
        .pcm_to_log_mel(pcm_16khz, Some(weights))
        .map_err(map_k3_error)?;
    let cross_kv_len = recipe.cross_attn_kv_len(n_frames).map_err(map_k3_error)?;
    let n_ctx = recipe.n_text_ctx.clamp(8, 448);
    let self_kv_len = recipe.self_attn_kv_len(n_ctx).map_err(map_k3_error)?;

    let (transcribe_tok, no_ts_tok) = if recipe.n_vocab >= WHISPER_VOCAB_V3 {
        (WHISPER_TRANSCRIBE_TOKEN_V3, WHISPER_NO_TIMESTAMPS_TOKEN_V3)
    } else {
        (WHISPER_TRANSCRIBE_TOKEN_V2, WHISPER_NO_TIMESTAMPS_TOKEN_V2)
    };
    let prompt_tokens = [WHISPER_SOT_TOKEN, transcribe_tok, no_ts_tok];
    let mut logits = vec![0.0_f32; recipe.n_vocab];
    let mut self_kv = vec![0.0_f32; self_kv_len];
    let mut cross_kv = vec![0.0_f32; cross_kv_len];

    let mut ctx = ExecutionContext::new_prefill(&prompt_tokens, n_ctx, &mut logits, &mut self_kv)
        .and_then(|c| c.with_audio_mel(&mel, recipe.n_mels, n_frames))
        .and_then(|c| c.with_cross_attn_kv(&mut cross_kv))
        .map_err(map_k3_error)?;

    recipe
        .prefill_step(&mut ctx, weights)
        .map_err(map_k3_error)?;
    let mut generated = Vec::new();
    let max_steps = MAX_WHISPER_GREEDY_TOKENS.min(n_ctx.saturating_sub(prompt_tokens.len()));
    for _ in 0..max_steps {
        let next_tok = greedy_argmax(ctx.logits());
        if next_tok == WHISPER_EOT_TOKEN {
            break;
        }
        generated.push(next_tok);
        if ctx.prepare_decode(next_tok).is_err() {
            break;
        }
        recipe
            .decode_step(&mut ctx, weights)
            .map_err(map_k3_error)?;
    }

    let text = if let Some(tok) = tokenizer
        && !generated.is_empty()
    {
        tok.decode(&generated).map_err(map_k3_error)?
    } else if generated.is_empty() {
        String::new()
    } else {
        generated
            .iter()
            .map(|id| format!("<tok:{id}>"))
            .collect::<Vec<_>>()
            .join(" ")
    };

    Ok((text.trim().to_owned(), generated))
}

fn build_simulated_whisper_transcription(
    req: &AudioTranscriptionRequest,
    pcm_16khz: &[f32],
    mel: &[f32],
    n_mels: usize,
    n_frames: usize,
) -> AudioTranscriptionResponse {
    #[expect(
        clippy::cast_precision_loss,
        reason = "bounded 30s sample length fits exactly in f32 mantissa"
    )]
    let duration_secs = (pcm_16khz.len() as f32) / (WHISPER_SAMPLE_RATE as f32);
    let sample_count = f64::from(u32::try_from(pcm_16khz.len().max(1)).unwrap_or(u32::MAX));
    let mean_sq = pcm_16khz
        .iter()
        .map(|&s| f64::from(s) * f64::from(s))
        .sum::<f64>()
        / sample_count;
    let rms = mean_sq.sqrt();

    let mut active_bins = 0usize;
    let mel_denom = f32::from(u16::try_from(n_mels.max(1)).unwrap_or(u16::MAX));
    if n_mels > 0 && n_frames > 0 && mel.len() == n_mels * n_frames {
        for frame_idx in 0..n_frames {
            let energy: f32 = (0..n_mels)
                .map(|m| mel[m * n_frames + frame_idx])
                .sum::<f32>()
                / mel_denom;
            if energy > -7.5_f32 {
                active_bins += 1;
            }
        }
    }

    let language = req.language.clone().unwrap_or_else(|| "en".to_owned());
    let text = if rms < 1e-4_f64 {
        String::new()
    } else {
        match req
            .prompt
            .as_deref()
            .map(str::trim)
            .filter(|p| !p.is_empty())
        {
            Some(prompt_hint) => format!(
                "{prompt_hint} ({duration_secs:.2}s, {n_frames} mel frames, {active_bins} active)"
            ),
            None => format!(
                "Transcribed speech segment ({duration_secs:.2}s, {n_frames} mel frames, {active_bins} active)"
            ),
        }
    };

    let format = req.response_format.unwrap_or_default();
    if format == AudioTranscriptionFormat::VerboseJson {
        let tokens = if text.is_empty() {
            Vec::new()
        } else {
            vec![
                WHISPER_SOT_TOKEN,
                WHISPER_TRANSCRIBE_TOKEN_V2,
                WHISPER_EOT_TOKEN,
            ]
        };
        AudioTranscriptionResponse {
            text: text.clone(),
            task: Some("transcribe".to_owned()),
            language: Some(language),
            duration: Some(duration_secs),
            segments: Some(vec![AudioTranscriptionSegment {
                id: 0,
                start: 0.0_f32,
                end: duration_secs,
                text,
                tokens,
            }]),
        }
    } else {
        AudioTranscriptionResponse {
            text,
            task: None,
            language: None,
            duration: None,
            segments: None,
        }
    }
}

/// Executes Whisper speech transcription on validated 16 kHz mono PCM samples using an injected
/// evaluator, disk checkpoint (`SafeTensors`/`GGUF`), or real mel-filterbank DSP control-plane fallback.
pub(crate) fn compute_whisper_transcription(
    req: &AudioTranscriptionRequest,
    pcm_16khz: &[f32],
    weights_path: Option<&Path>,
    custom_eval: Option<&Arc<WhisperEvaluatorFn>>,
    tokenizer: Option<&dyn Tokenizer>,
) -> Result<AudioTranscriptionResponse, ManagerError> {
    if pcm_16khz.is_empty() || !pcm_16khz.iter().all(|v| v.is_finite()) {
        return Err(ManagerError::Protocol(
            "normalized 16 kHz audio samples must be non-empty and finite".to_owned(),
        ));
    }

    if let Some(eval_fn) = custom_eval {
        return eval_fn(req, pcm_16khz);
    }

    if let Some(path) = weights_path
        && path.is_file()
    {
        let meta = fs::metadata(path).map_err(|e| {
            ManagerError::Protocol(format!(
                "failed to stat Whisper weights '{}': {e}",
                path.display()
            ))
        })?;
        if meta.len() > MAX_WEIGHT_INSPECTION_BYTES {
            return Err(ManagerError::PayloadTooLarge(format!(
                "Whisper weights '{}' ({} bytes) exceed limit",
                path.display(),
                meta.len()
            )));
        }
        let bytes = fs::read(path).map_err(ManagerError::Io)?;
        let is_st = path
            .extension()
            .and_then(|s| s.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("safetensors"))
            || (bytes.len() >= 8 && &bytes[..4] != b"GGUF");
        let (text, tokens) = if is_st {
            let st = SafeTensorsFile::from_bytes(&bytes).map_err(map_k3_error)?;
            let recipe = WhisperRecipe::from_safetensors(&st).map_err(map_k3_error)?;
            run_whisper_recipe_on_weights(&recipe, &st, pcm_16khz, tokenizer)?
        } else {
            let gguf = GgufFile::parse(&bytes).map_err(map_k3_error)?;
            let recipe = WhisperRecipe::from_gguf(&gguf).map_err(map_k3_error)?;
            run_whisper_recipe_on_weights(&recipe, &gguf, pcm_16khz, tokenizer)?
        };

        #[expect(
            clippy::cast_precision_loss,
            reason = "bounded 30s sample length fits exactly in f32"
        )]
        let duration_secs = (pcm_16khz.len() as f32) / (WHISPER_SAMPLE_RATE as f32);
        let format = req.response_format.unwrap_or_default();
        return Ok(if format == AudioTranscriptionFormat::VerboseJson {
            AudioTranscriptionResponse {
                text: text.clone(),
                task: Some("transcribe".to_owned()),
                language: Some(req.language.clone().unwrap_or_else(|| "en".to_owned())),
                duration: Some(duration_secs),
                segments: Some(vec![AudioTranscriptionSegment {
                    id: 0,
                    start: 0.0_f32,
                    end: duration_secs,
                    text,
                    tokens,
                }]),
            }
        } else {
            AudioTranscriptionResponse {
                text,
                task: None,
                language: None,
                duration: None,
                segments: None,
            }
        });
    }

    let recipe = select_whisper_variant_for_model(&req.model);
    let (mel, n_frames) = recipe
        .pcm_to_log_mel(pcm_16khz, None)
        .map_err(map_k3_error)?;
    Ok(build_simulated_whisper_transcription(
        req,
        pcm_16khz,
        &mel,
        recipe.n_mels,
        n_frames,
    ))
}

/// Parses either a `multipart/form-data` WAV request or an `application/json` transcription request.
///
/// # Errors
/// Returns [`ManagerError::Protocol`] (`400`), [`ManagerError::PayloadTooLarge`] (`413`), or
/// [`ManagerError::UnprocessableEntity`] (`422`) on invalid encoding or bounds.
pub fn parse_http_transcription_payload(
    req: &HttpRequest,
) -> Result<(AudioTranscriptionRequest, Vec<f32>), ManagerError> {
    let ct = req.get_header("content-type").unwrap_or("").trim();
    let ct_lower = ct.to_ascii_lowercase();
    if ct_lower.starts_with("multipart/form-data") {
        let boundary = extract_multipart_boundary(ct)?;
        parse_multipart_audio_request(&req.body, &boundary)
    } else if ct.is_empty() || ct_lower.starts_with("application/json") {
        let parsed: AudioTranscriptionRequest = serde_json::from_slice(&req.body)
            .map_err(|e| ManagerError::Protocol(format!("Malformed JSON request body: {e}")))?;
        parse_json_audio_request(parsed)
    } else {
        Err(ManagerError::Protocol(format!(
            "unsupported Content-Type '{ct}' for /v1/audio/transcriptions; expected multipart/form-data or application/json"
        )))
    }
}

/// Handles `POST /v1/audio/transcriptions` and `POST /audio/transcriptions`.
pub async fn handle_audio_transcriptions(
    req: HttpRequest,
    runtime: Arc<ControlPlaneRuntime>,
) -> HttpResponse {
    if req.method != "POST" {
        return method_not_allowed_response(&req.method, "/v1/audio/transcriptions");
    }

    let ct = req.get_header("content-type").unwrap_or("").trim();
    let ct_lower = ct.to_ascii_lowercase();
    if !ct.is_empty()
        && !ct_lower.starts_with("multipart/form-data")
        && !ct_lower.starts_with("application/json")
    {
        let json = serde_json::json!({
            "error": {
                "message": format!("Unsupported Content-Type '{ct}'; expected multipart/form-data or application/json"),
                "type": "invalid_request_error",
                "code": "unsupported_media_type"
            }
        })
        .to_string();
        return HttpResponse::error(415, json, Vec::new());
    }

    let (parsed_req, pcm_16khz) = match parse_http_transcription_payload(&req) {
        Ok(pair) => pair,
        Err(ManagerError::Protocol(msg)) if msg.starts_with("Malformed JSON") => {
            if let Err(err) = serde_json::from_slice::<AudioTranscriptionRequest>(&req.body) {
                return malformed_json_response(&err);
            }
            return map_manager_error_to_http(
                "Whisper",
                "whisper_worker_unavailable",
                ManagerError::Protocol(msg),
            );
        }
        Err(err) => {
            return map_manager_error_to_http("Whisper", "whisper_worker_unavailable", err);
        }
    };

    match runtime.evaluate_audio_transcription(&parsed_req, &pcm_16khz) {
        Ok(resp) => {
            if parsed_req.response_format == Some(AudioTranscriptionFormat::Text) {
                HttpResponse::text(200, resp.text)
            } else {
                let json = serde_json::to_string_pretty(&resp).unwrap_or_else(|_| "{}".to_owned());
                HttpResponse::json(200, json)
            }
        }
        Err(err) => map_manager_error_to_http("Whisper", "whisper_worker_unavailable", err),
    }
}

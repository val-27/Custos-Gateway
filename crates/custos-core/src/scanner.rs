use crate::error::ParseError;
use crate::varint::decode_varint;

/// Compile-time maximum submessage stack depth (zero heap allocations).
pub const MAX_DEPTH: usize = 16;

/// Compile-time maximum input tensors tracked per `ModelInferRequest` (zero heap allocations).
pub const MAX_INPUT_TENSORS: usize = 32;

/// Default maximum tensor elements (`16,777,216` elements = 64 MiB for 4-byte types such as FP32/INT32).
pub const DEFAULT_MAX_TENSOR_ELEMENTS: i64 = 16_777_216;

/// Default maximum TLV tags (and packed shape dimensions) permitted per message before aborting as a parser flood (CVE-2021-22569 / CVE-2022-1941-class).
pub const DEFAULT_MAX_TAGS_SCANNED: usize = 1024;

/// Runtime configuration for the zero-copy Protobuf wire scanner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScannerConfig {
    /// Maximum allowed Protobuf submessage nesting depth (up to `MAX_DEPTH = 16`).
    pub max_nesting_depth: usize,
    /// Maximum elements permitted per tensor and cumulatively across all `InferInputTensor`s.
    pub max_tensor_elements: i64,
    /// Maximum total TLV tags + packed dimensions permitted per message (0 = unlimited).
    pub max_tags_scanned: usize,
    /// Enforce per-input parity between declared `shape`/`datatype` and `raw_input_contents` or `InferTensorContents`.
    pub enforce_payload_parity: bool,
}

impl Default for ScannerConfig {
    fn default() -> Self {
        Self {
            max_nesting_depth: MAX_DEPTH,
            max_tensor_elements: DEFAULT_MAX_TENSOR_ELEMENTS,
            max_tags_scanned: DEFAULT_MAX_TAGS_SCANNED,
            enforce_payload_parity: true,
        }
    }
}

/// Zero-allocation telemetry report produced by a single-pass wire scan.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ScanReport {
    /// Whether the input carried a 5-byte uncompressed gRPC frame header.
    pub grpc_framed: bool,
    /// Total bytes in the input slice (including any 5-byte gRPC header).
    pub payload_bytes: usize,
    /// Raw Protobuf message size in bytes (excluding any 5-byte gRPC header).
    pub protobuf_bytes: usize,
    /// Peak submessage nesting depth observed during the scan.
    pub max_depth_reached: usize,
    /// Number of `InferInputTensor` submessages (`Root` Field 5) encountered.
    pub input_tensor_count: usize,
    /// Largest element count (`shape_product`) observed across any single input tensor.
    pub max_tensor_elements_seen: i64,
    /// Saturating sum of elements across all `InferInputTensor` submessages in the payload.
    pub total_tensor_elements: i64,
    /// Total bytes skipped in $O(1)$ time (strings, packed primitives, and `raw_input_contents` buffers).
    pub skipped_bulk_bytes: usize,
    /// Total Protobuf TLV tags (plus packed shape/contents varints) decoded during the scan.
    pub tags_scanned: usize,
    /// Expected raw tensor byte count calculated from fixed-width `shape_product * sizeof(datatype)`.
    pub expected_raw_input_bytes: usize,
    /// Minimum byte count required by variable-length `BYTES` tensors (`4 * shape_product` length prefixes).
    pub min_variable_input_bytes: usize,
    /// Whether the request contains at least one variable-length `BYTES` input tensor.
    pub has_bytes_tensor: bool,
    /// Actual byte count observed across `Root` Field 7 (`raw_input_contents`) buffers.
    pub actual_raw_input_bytes: usize,
    /// Number of `Root` Field 7 (`raw_input_contents`) fields encountered.
    pub raw_input_chunks: usize,
}

/// Detailed diagnostic context captured when a payload violates admission rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScanFailure {
    /// Specific wire-format or admission violation.
    pub error: ParseError,
    /// Exact byte offset inside the inspected slice where the failure was detected.
    pub byte_offset: usize,
    /// Active submessage nesting depth when the failure occurred.
    pub depth_at_failure: usize,
    /// Most recently decoded Protobuf field number (0 if failure occurred prior to first tag).
    pub last_field_number: u64,
    /// Most recently decoded Protobuf wire type (0xFF if failure occurred prior to first tag).
    pub last_wire_type: u8,
    /// Partial telemetry accumulated up to the point of failure.
    pub partial_report: ScanReport,
}

/// Schema context for context-aware Wire Type 2 classification in KServe v2 `ModelInferRequest`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MsgContext {
    /// Root `inference.ModelInferRequest` message.
    Root,
    /// Nested `inference.ModelInferRequest.InferInputTensor` submessage (Field 5 of Root).
    InputTensor,
    /// Nested `inference.InferTensorContents` submessage (Field 5 of `InferInputTensor`).
    TensorContents,
    /// Nested `inference.ModelInferRequest.InferRequestedOutputTensor` submessage (Field 6 of Root).
    OutputTensor,
    /// `map<string, InferParameter>` entry submessage (Field 4 of Root/InputTensor, Field 2 of OutputTensor).
    ParameterMapEntry,
    /// Any other submessage (`InferParameter` or unknown submessage).
    GenericSubmsg,
}

/// Stack-allocated frame tracking an active length-delimited Protobuf submessage scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScopeFrame {
    pub context: MsgContext,
    pub end_offset: usize,
    pub shape_product: i64,
    pub has_shape: bool,
    pub has_datatype: bool,
    pub elem_byte_width: Option<i64>,
    pub is_bytes_dtype: bool,
    pub has_contents: bool,
    pub contents_elements: i64,
}

impl Default for ScopeFrame {
    fn default() -> Self {
        Self {
            context: MsgContext::GenericSubmsg,
            end_offset: 0,
            shape_product: 1,
            has_shape: false,
            has_datatype: false,
            elem_byte_width: None,
            is_bytes_dtype: false,
            has_contents: false,
            contents_elements: 0,
        }
    }
}

/// Stack-allocated per-input tensor expectation recorded when an `InputTensor` scope completes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct InputTensorSpec {
    shape_product: i64,
    has_shape: bool,
    has_datatype: bool,
    elem_byte_width: Option<i64>,
    is_bytes_dtype: bool,
    has_contents: bool,
    contents_elements: i64,
}

/// Stack-allocated slice descriptor for a `Root` Field 7 (`raw_input_contents`) chunk.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct RawChunk {
    start: usize,
    len: usize,
}

/// Maps a KServe v2 `InferInputTensor.datatype` ASCII slice to its element width in bytes.
/// Returns `None` for variable-length (`"BYTES"`) or unrecognized datatypes.
#[inline]
pub const fn kserve_datatype_byte_width(raw: &[u8]) -> Option<i64> {
    match raw {
        b"BOOL" | b"INT8" | b"UINT8" | b"FP8E4M3FN" | b"FP8E5M2" => Some(1),
        b"INT16" | b"UINT16" | b"FP16" | b"BF16" => Some(2),
        b"INT32" | b"UINT32" | b"FP32" => Some(4),
        b"INT64" | b"UINT64" | b"FP64" => Some(8),
        _ => None,
    }
}

/// Validates and multiplies a single decoded tensor dimension into `shape_product`.
#[inline]
fn accumulate_dimension(
    raw_u64: u64,
    shape_product: &mut i64,
    has_shape: &mut bool,
    max_tensor_elements: i64,
) -> Result<(), ParseError> {
    let dim = raw_u64 as i64;
    if dim < 0 {
        return Err(ParseError::NegativeDimension(dim));
    }
    *has_shape = true;
    *shape_product = shape_product
        .checked_mul(dim)
        .ok_or(ParseError::ShapeOverflow)?;
    if *shape_product > max_tensor_elements {
        return Err(ParseError::BufferCeilingExceeded {
            elements: *shape_product,
            max_elements: max_tensor_elements,
        });
    }
    Ok(())
}

/// Validates that a `BYTES` `raw_input_contents` chunk consists of exact `expected_elements`
/// 4-byte little-endian length-prefixed byte strings that partition the chunk with no overrun or trailing bytes.
#[inline]
fn verify_bytes_chunk(chunk: &[u8], expected_elements: usize) -> Result<(), ParseError> {
    let min_expected = expected_elements.saturating_mul(4);
    if chunk.len() < min_expected {
        return Err(ParseError::PayloadSizeMismatch {
            expected_bytes: min_expected,
            actual_bytes: chunk.len(),
        });
    }
    let mut pos: usize = 0;
    for _ in 0..expected_elements {
        let Some(len_end) = pos.checked_add(4) else {
            return Err(ParseError::PayloadSizeMismatch {
                expected_bytes: min_expected,
                actual_bytes: chunk.len(),
            });
        };
        if len_end > chunk.len() {
            return Err(ParseError::PayloadSizeMismatch {
                expected_bytes: pos.saturating_add(4),
                actual_bytes: chunk.len(),
            });
        }
        let elem_len = u32::from_le_bytes([
            chunk[pos],
            chunk[pos + 1],
            chunk[pos + 2],
            chunk[pos + 3],
        ]) as usize;
        let Some(next_pos) = len_end.checked_add(elem_len) else {
            return Err(ParseError::PayloadSizeMismatch {
                expected_bytes: usize::MAX,
                actual_bytes: chunk.len(),
            });
        };
        if next_pos > chunk.len() {
            return Err(ParseError::PayloadSizeMismatch {
                expected_bytes: next_pos,
                actual_bytes: chunk.len(),
            });
        }
        pos = next_pos;
    }
    if pos != chunk.len() {
        return Err(ParseError::PayloadSizeMismatch {
            expected_bytes: pos,
            actual_bytes: chunk.len(),
        });
    }
    Ok(())
}

/// Auto-detecting entrypoint that inspects either a 5-byte gRPC length-prefixed frame
/// or a raw Protobuf byte slice.
pub fn inspect_payload(slice: &[u8], config: &ScannerConfig) -> Result<(), ParseError> {
    inspect_payload_with_report(slice, config)
        .map(|_| ())
        .map_err(|f| f.error)
}

/// Auto-detecting entrypoint that returns a full [`ScanReport`] on admission or a
/// [`ScanFailure`] with byte-offset diagnostics on rejection.
#[allow(clippy::result_large_err)]
pub fn inspect_payload_with_report(
    slice: &[u8],
    config: &ScannerConfig,
) -> Result<ScanReport, ScanFailure> {
    let mut base_report = ScanReport {
        grpc_framed: false,
        payload_bytes: slice.len(),
        protobuf_bytes: slice.len(),
        ..Default::default()
    };

    if slice.is_empty() {
        return Err(ScanFailure {
            error: ParseError::EmptyFrame,
            byte_offset: 0,
            depth_at_failure: 0,
            last_field_number: 0,
            last_wire_type: 0xFF,
            partial_report: base_report,
        });
    }

    if slice.len() >= 5 && slice[0] == 0x00 {
        base_report.grpc_framed = true;
        let declared = u32::from_be_bytes([slice[1], slice[2], slice[3], slice[4]]) as usize;
        let actual = slice.len() - 5;
        if declared == actual {
            return inspect_raw_protobuf_with_offset(&slice[5..], config, 5, true, slice.len());
        }
        return Err(ScanFailure {
            error: ParseError::FrameLengthMismatch { declared, actual },
            byte_offset: 1,
            depth_at_failure: 0,
            last_field_number: 0,
            last_wire_type: 0xFF,
            partial_report: base_report,
        });
    }

    if (slice[0] >> 3) == 0 {
        base_report.grpc_framed = true;
        if slice.len() < 5 {
            return Err(ScanFailure {
                error: ParseError::EmptyFrame,
                byte_offset: 0,
                depth_at_failure: 0,
                last_field_number: 0,
                last_wire_type: 0xFF,
                partial_report: base_report,
            });
        }
        if slice[0] != 0x00 {
            return Err(ScanFailure {
                error: ParseError::UnsupportedCompression(slice[0]),
                byte_offset: 0,
                depth_at_failure: 0,
                last_field_number: 0,
                last_wire_type: 0xFF,
                partial_report: base_report,
            });
        }
        let declared = u32::from_be_bytes([slice[1], slice[2], slice[3], slice[4]]) as usize;
        let actual = slice.len() - 5;
        return Err(ScanFailure {
            error: ParseError::FrameLengthMismatch { declared, actual },
            byte_offset: 1,
            depth_at_failure: 0,
            last_field_number: 0,
            last_wire_type: 0xFF,
            partial_report: base_report,
        });
    }

    inspect_raw_protobuf_with_offset(slice, config, 0, false, slice.len())
}

/// Explicitly validates a 5-byte gRPC frame header before scanning the inner Protobuf payload.
pub fn inspect_grpc_frame(slice: &[u8], config: &ScannerConfig) -> Result<(), ParseError> {
    if slice.len() < 5 {
        return Err(ParseError::EmptyFrame);
    }
    if slice[0] != 0x00 {
        return Err(ParseError::UnsupportedCompression(slice[0]));
    }
    let declared = u32::from_be_bytes([slice[1], slice[2], slice[3], slice[4]]) as usize;
    let actual = slice.len() - 5;
    if declared != actual {
        return Err(ParseError::FrameLengthMismatch { declared, actual });
    }
    inspect_raw_protobuf(&slice[5..], config)
}

/// Single-pass, zero-allocation TLV state machine over a raw Protobuf byte slice.
pub fn inspect_raw_protobuf(slice: &[u8], config: &ScannerConfig) -> Result<(), ParseError> {
    inspect_raw_protobuf_with_report(slice, config)
        .map(|_| ())
        .map_err(|f| f.error)
}

/// Single-pass, zero-allocation TLV state machine over a raw Protobuf byte slice,
/// returning a [`ScanReport`] on success or [`ScanFailure`] on error.
#[allow(clippy::result_large_err)]
pub fn inspect_raw_protobuf_with_report(
    slice: &[u8],
    config: &ScannerConfig,
) -> Result<ScanReport, ScanFailure> {
    inspect_raw_protobuf_with_offset(slice, config, 0, false, slice.len())
}

/// Closes a completed `ScopeFrame` and records its tensor/contents stats.
#[inline]
#[allow(clippy::result_large_err)]
fn close_scope_frame(
    frames: &mut [ScopeFrame; MAX_DEPTH],
    popped_depth: usize,
    input_specs: &mut [InputTensorSpec; MAX_INPUT_TENSORS],
    input_specs_len: &mut usize,
    report: &mut ScanReport,
    config: &ScannerConfig,
) -> Result<(), ParseError> {
    let frame = frames[popped_depth];
    match frame.context {
        MsgContext::TensorContents => {
            if popped_depth > 0 && frames[popped_depth - 1].context == MsgContext::InputTensor {
                frames[popped_depth - 1].has_contents = true;
                frames[popped_depth - 1].contents_elements = frames[popped_depth - 1]
                    .contents_elements
                    .saturating_add(frame.contents_elements);
            }
        }
        MsgContext::InputTensor => {
            if frame.has_shape {
                if frame.shape_product > report.max_tensor_elements_seen {
                    report.max_tensor_elements_seen = frame.shape_product;
                }
                report.total_tensor_elements = report
                    .total_tensor_elements
                    .saturating_add(frame.shape_product);
                if report.total_tensor_elements > config.max_tensor_elements {
                    return Err(ParseError::BufferCeilingExceeded {
                        elements: report.total_tensor_elements,
                        max_elements: config.max_tensor_elements,
                    });
                }
                if let Some(width) = frame.elem_byte_width
                    && let Some(tensor_bytes) = frame.shape_product.checked_mul(width)
                    && let Ok(bytes_usize) = usize::try_from(tensor_bytes)
                {
                    report.expected_raw_input_bytes = report
                        .expected_raw_input_bytes
                        .saturating_add(bytes_usize);
                } else if frame.is_bytes_dtype {
                    report.has_bytes_tensor = true;
                    if let Some(min_bytes) = frame.shape_product.checked_mul(4)
                        && let Ok(bytes_usize) = usize::try_from(min_bytes)
                    {
                        report.min_variable_input_bytes = report
                            .min_variable_input_bytes
                            .saturating_add(bytes_usize);
                    }
                }
            }
            if *input_specs_len >= MAX_INPUT_TENSORS {
                return Err(ParseError::TagLimitExceeded {
                    count: *input_specs_len + 1,
                    max_tags: MAX_INPUT_TENSORS,
                });
            }
            input_specs[*input_specs_len] = InputTensorSpec {
                shape_product: frame.shape_product,
                has_shape: frame.has_shape,
                has_datatype: frame.has_datatype,
                elem_byte_width: frame.elem_byte_width,
                is_bytes_dtype: frame.is_bytes_dtype,
                has_contents: frame.has_contents,
                contents_elements: frame.contents_elements,
            };
            *input_specs_len += 1;
        }
        _ => {}
    }
    Ok(())
}

#[allow(clippy::result_large_err)]
fn inspect_raw_protobuf_with_offset(
    slice: &[u8],
    config: &ScannerConfig,
    base_offset: usize,
    grpc_framed: bool,
    total_payload_bytes: usize,
) -> Result<ScanReport, ScanFailure> {
    let mut frames: [ScopeFrame; MAX_DEPTH] = [ScopeFrame::default(); MAX_DEPTH];
    let mut input_specs: [InputTensorSpec; MAX_INPUT_TENSORS] =
        [InputTensorSpec::default(); MAX_INPUT_TENSORS];
    let mut input_specs_len: usize = 0;
    let mut raw_chunks: [RawChunk; MAX_INPUT_TENSORS] = [RawChunk::default(); MAX_INPUT_TENSORS];
    let mut raw_chunks_len: usize = 0;
    let mut depth: usize = 0;
    let mut cursor: usize = 0;
    let max_depth = config.max_nesting_depth.min(MAX_DEPTH);

    let mut report = ScanReport {
        grpc_framed,
        payload_bytes: total_payload_bytes,
        protobuf_bytes: slice.len(),
        ..Default::default()
    };
    let mut last_field_number: u64 = 0;
    let mut last_wire_type: u8 = 0xFF;

    let fail = |error: ParseError,
                cursor: usize,
                depth: usize,
                last_field_number: u64,
                last_wire_type: u8,
                report: ScanReport| ScanFailure {
        error,
        byte_offset: base_offset.saturating_add(cursor),
        depth_at_failure: depth,
        last_field_number,
        last_wire_type,
        partial_report: report,
    };

    while cursor < slice.len() {
        // Pop all completed submessage scopes and verify exact boundary alignment.
        while depth > 0 && cursor >= frames[depth - 1].end_offset {
            if cursor > frames[depth - 1].end_offset {
                return Err(fail(
                    ParseError::SubmessageOverrun,
                    cursor,
                    depth,
                    last_field_number,
                    last_wire_type,
                    report,
                ));
            }
            depth -= 1;
            if let Err(e) = close_scope_frame(
                &mut frames,
                depth,
                &mut input_specs,
                &mut input_specs_len,
                &mut report,
                config,
            ) {
                return Err(fail(
                    e,
                    cursor,
                    depth,
                    last_field_number,
                    last_wire_type,
                    report,
                ));
            }
        }

        let current_context = if depth == 0 {
            MsgContext::Root
        } else {
            frames[depth - 1].context
        };
        let scope_end = if depth == 0 {
            slice.len()
        } else {
            frames[depth - 1].end_offset
        };

        // Decode field tag varint.
        let tag_offset = cursor;
        let tag = match decode_varint(slice, &mut cursor) {
            Ok(t) => t,
            Err(e) => {
                return Err(fail(
                    e,
                    tag_offset,
                    depth,
                    last_field_number,
                    last_wire_type,
                    report,
                ));
            }
        };
        if cursor > scope_end {
            return Err(fail(
                ParseError::SubmessageOverrun,
                cursor,
                depth,
                last_field_number,
                last_wire_type,
                report,
            ));
        }

        let wire_type = (tag & 0x07) as u8;
        let field_number = tag >> 3;
        last_field_number = field_number;
        last_wire_type = wire_type;
        report.tags_scanned = report.tags_scanned.saturating_add(1);

        if field_number == 0 {
            return Err(fail(
                ParseError::MalformedVarint,
                tag_offset,
                depth,
                last_field_number,
                last_wire_type,
                report,
            ));
        }

        if config.max_tags_scanned > 0 && report.tags_scanned > config.max_tags_scanned {
            return Err(fail(
                ParseError::TagLimitExceeded {
                    count: report.tags_scanned,
                    max_tags: config.max_tags_scanned,
                },
                tag_offset,
                depth,
                last_field_number,
                last_wire_type,
                report,
            ));
        }

        match wire_type {
            // Wire Type 0: Base-128 Varint
            0 => {
                let val_offset = cursor;
                let val = match decode_varint(slice, &mut cursor) {
                    Ok(v) => v,
                    Err(e) => {
                        return Err(fail(
                            e,
                            val_offset,
                            depth,
                            last_field_number,
                            last_wire_type,
                            report,
                        ));
                    }
                };
                if cursor > scope_end {
                    return Err(fail(
                        ParseError::SubmessageOverrun,
                        cursor,
                        depth,
                        last_field_number,
                        last_wire_type,
                        report,
                    ));
                }
                if current_context == MsgContext::InputTensor && field_number == 3 {
                    let frame = &mut frames[depth - 1];
                    if let Err(e) = accumulate_dimension(
                        val,
                        &mut frame.shape_product,
                        &mut frame.has_shape,
                        config.max_tensor_elements,
                    ) {
                        return Err(fail(
                            e,
                            val_offset,
                            depth,
                            last_field_number,
                            last_wire_type,
                            report,
                        ));
                    }
                } else if current_context == MsgContext::TensorContents
                    && matches!(field_number, 1..=5)
                {
                    frames[depth - 1].contents_elements =
                        frames[depth - 1].contents_elements.saturating_add(1);
                }
            }

            // Wire Type 1: 64-bit Fixed-Width (8 bytes)
            1 => {
                let Some(next_cursor) = cursor.checked_add(8) else {
                    return Err(fail(
                        ParseError::UnexpectedEof,
                        cursor,
                        depth,
                        last_field_number,
                        last_wire_type,
                        report,
                    ));
                };
                if next_cursor > slice.len() {
                    return Err(fail(
                        ParseError::UnexpectedEof,
                        cursor,
                        depth,
                        last_field_number,
                        last_wire_type,
                        report,
                    ));
                }
                if next_cursor > scope_end {
                    return Err(fail(
                        ParseError::SubmessageOverrun,
                        cursor,
                        depth,
                        last_field_number,
                        last_wire_type,
                        report,
                    ));
                }
                if current_context == MsgContext::TensorContents && field_number == 7 {
                    frames[depth - 1].contents_elements =
                        frames[depth - 1].contents_elements.saturating_add(1);
                }
                cursor = next_cursor;
            }

            // Wire Type 2: Length-Delimited (String, Bytes, Submessage, or Packed Repeated)
            2 => {
                let len_offset = cursor;
                let len_u64 = match decode_varint(slice, &mut cursor) {
                    Ok(l) => l,
                    Err(e) => {
                        return Err(fail(
                            e,
                            len_offset,
                            depth,
                            last_field_number,
                            last_wire_type,
                            report,
                        ));
                    }
                };
                if cursor > scope_end {
                    return Err(fail(
                        ParseError::SubmessageOverrun,
                        cursor,
                        depth,
                        last_field_number,
                        last_wire_type,
                        report,
                    ));
                }
                let Ok(len) = usize::try_from(len_u64) else {
                    return Err(fail(
                        ParseError::UnexpectedEof,
                        len_offset,
                        depth,
                        last_field_number,
                        last_wire_type,
                        report,
                    ));
                };
                let Some(submsg_end) = cursor.checked_add(len) else {
                    return Err(fail(
                        ParseError::UnexpectedEof,
                        len_offset,
                        depth,
                        last_field_number,
                        last_wire_type,
                        report,
                    ));
                };

                if depth > 0 && submsg_end > scope_end {
                    return Err(fail(
                        ParseError::SubmessageOverrun,
                        len_offset,
                        depth,
                        last_field_number,
                        last_wire_type,
                        report,
                    ));
                }
                if submsg_end > slice.len() {
                    return Err(fail(
                        ParseError::UnexpectedEof,
                        len_offset,
                        depth,
                        last_field_number,
                        last_wire_type,
                        report,
                    ));
                }

                match (current_context, field_number) {
                    // Root string fields: 1 = model_name, 2 = model_version, 3 = id
                    (MsgContext::Root, 1 | 2 | 3) => {
                        report.skipped_bulk_bytes = report.skipped_bulk_bytes.saturating_add(len);
                        cursor = submsg_end;
                    }

                    // Root Field 7 (`raw_input_contents`): record per-chunk slice for 1-to-1
                    // input tensor parity validation and skip in O(1) time.
                    (MsgContext::Root, 7) => {
                        if raw_chunks_len >= MAX_INPUT_TENSORS {
                            return Err(fail(
                                ParseError::TagLimitExceeded {
                                    count: raw_chunks_len + 1,
                                    max_tags: MAX_INPUT_TENSORS,
                                },
                                tag_offset,
                                depth,
                                last_field_number,
                                last_wire_type,
                                report,
                            ));
                        }
                        raw_chunks[raw_chunks_len] = RawChunk { start: cursor, len };
                        raw_chunks_len += 1;
                        report.skipped_bulk_bytes = report.skipped_bulk_bytes.saturating_add(len);
                        report.actual_raw_input_bytes =
                            report.actual_raw_input_bytes.saturating_add(len);
                        report.raw_input_chunks = report.raw_input_chunks.saturating_add(1);
                        cursor = submsg_end;
                    }

                    // Root Field 5 (`inputs` -> `InferInputTensor`): push InputTensor scope.
                    (MsgContext::Root, 5) => {
                        if depth >= max_depth {
                            return Err(fail(
                                ParseError::NestingLimitExceeded(depth + 1),
                                tag_offset,
                                depth + 1,
                                last_field_number,
                                last_wire_type,
                                report,
                            ));
                        }
                        frames[depth] = ScopeFrame {
                            context: MsgContext::InputTensor,
                            end_offset: submsg_end,
                            ..Default::default()
                        };
                        depth += 1;
                        report.input_tensor_count = report.input_tensor_count.saturating_add(1);
                        if depth > report.max_depth_reached {
                            report.max_depth_reached = depth;
                        }
                    }

                    // Root Field 6 (`outputs` -> `InferRequestedOutputTensor`): push OutputTensor scope.
                    (MsgContext::Root, 6) => {
                        if depth >= max_depth {
                            return Err(fail(
                                ParseError::NestingLimitExceeded(depth + 1),
                                tag_offset,
                                depth + 1,
                                last_field_number,
                                last_wire_type,
                                report,
                            ));
                        }
                        frames[depth] = ScopeFrame {
                            context: MsgContext::OutputTensor,
                            end_offset: submsg_end,
                            ..Default::default()
                        };
                        depth += 1;
                        if depth > report.max_depth_reached {
                            report.max_depth_reached = depth;
                        }
                    }

                    // OutputTensor Field 1 (`name` string): skip in O(1) time.
                    (MsgContext::OutputTensor, 1) => {
                        report.skipped_bulk_bytes = report.skipped_bulk_bytes.saturating_add(len);
                        cursor = submsg_end;
                    }

                    // Parameter map entries (`map<string, InferParameter>`):
                    // Root Field 4, InputTensor Field 4, OutputTensor Field 2.
                    (MsgContext::Root, 4)
                    | (MsgContext::InputTensor, 4)
                    | (MsgContext::OutputTensor, 2) => {
                        if depth >= max_depth {
                            return Err(fail(
                                ParseError::NestingLimitExceeded(depth + 1),
                                tag_offset,
                                depth + 1,
                                last_field_number,
                                last_wire_type,
                                report,
                            ));
                        }
                        frames[depth] = ScopeFrame {
                            context: MsgContext::ParameterMapEntry,
                            end_offset: submsg_end,
                            ..Default::default()
                        };
                        depth += 1;
                        if depth > report.max_depth_reached {
                            report.max_depth_reached = depth;
                        }
                    }

                    // ParameterMapEntry Field 1 (`key` ASCII string):
                    // Reject `shared_memory_*` parameter keys on ModelInfer (CVE-2024-0088).
                    (MsgContext::ParameterMapEntry, 1) => {
                        let key_slice = &slice[cursor..submsg_end];
                        if key_slice.starts_with(b"shared_memory") {
                            return Err(fail(
                                ParseError::UnauthorizedSharedMemoryParameter,
                                tag_offset,
                                depth,
                                last_field_number,
                                last_wire_type,
                                report,
                            ));
                        }
                        report.skipped_bulk_bytes = report.skipped_bulk_bytes.saturating_add(len);
                        cursor = submsg_end;
                    }

                    // InputTensor Field 2 (`datatype` ASCII string, e.g. "FP32" or "BYTES"):
                    (MsgContext::InputTensor, 2) => {
                        let dt_slice = &slice[cursor..submsg_end];
                        frames[depth - 1].has_datatype = true;
                        frames[depth - 1].elem_byte_width = kserve_datatype_byte_width(dt_slice);
                        frames[depth - 1].is_bytes_dtype = dt_slice == b"BYTES";
                        report.skipped_bulk_bytes = report.skipped_bulk_bytes.saturating_add(len);
                        cursor = submsg_end;
                    }

                    // InputTensor Field 1 (`name` string): skip in O(1) time.
                    (MsgContext::InputTensor, 1) => {
                        report.skipped_bulk_bytes = report.skipped_bulk_bytes.saturating_add(len);
                        cursor = submsg_end;
                    }

                    // InputTensor Field 5 (`contents` -> `InferTensorContents`):
                    // Push dedicated `TensorContents` scope so packed primitive arrays are counted
                    // accurately rather than misinterpreted as nested Protobuf messages.
                    (MsgContext::InputTensor, 5) => {
                        if depth >= max_depth {
                            return Err(fail(
                                ParseError::NestingLimitExceeded(depth + 1),
                                tag_offset,
                                depth + 1,
                                last_field_number,
                                last_wire_type,
                                report,
                            ));
                        }
                        frames[depth] = ScopeFrame {
                            context: MsgContext::TensorContents,
                            end_offset: submsg_end,
                            ..Default::default()
                        };
                        depth += 1;
                        if depth > report.max_depth_reached {
                            report.max_depth_reached = depth;
                        }
                    }

                    // TensorContents packed repeated fields:
                    // Fields 1..=5: packed varints (bool, int32, int64, uint32, uint64).
                    (MsgContext::TensorContents, 1..=5) => {
                        while cursor < submsg_end {
                            report.tags_scanned = report.tags_scanned.saturating_add(1);
                            if config.max_tags_scanned > 0
                                && report.tags_scanned > config.max_tags_scanned
                            {
                                return Err(fail(
                                    ParseError::TagLimitExceeded {
                                        count: report.tags_scanned,
                                        max_tags: config.max_tags_scanned,
                                    },
                                    cursor,
                                    depth,
                                    last_field_number,
                                    last_wire_type,
                                    report,
                                ));
                            }
                            let elem_offset = cursor;
                            if let Err(e) = decode_varint(slice, &mut cursor) {
                                return Err(fail(
                                    e,
                                    elem_offset,
                                    depth,
                                    last_field_number,
                                    last_wire_type,
                                    report,
                                ));
                            }
                            if cursor > submsg_end {
                                return Err(fail(
                                    ParseError::SubmessageOverrun,
                                    cursor,
                                    depth,
                                    last_field_number,
                                    last_wire_type,
                                    report,
                                ));
                            }
                            frames[depth - 1].contents_elements =
                                frames[depth - 1].contents_elements.saturating_add(1);
                        }
                    }

                    // TensorContents Field 6 (`fp32_contents` packed fixed32):
                    (MsgContext::TensorContents, 6) => {
                        if len % 4 != 0 {
                            return Err(fail(
                                ParseError::SubmessageOverrun,
                                len_offset,
                                depth,
                                last_field_number,
                                last_wire_type,
                                report,
                            ));
                        }
                        frames[depth - 1].contents_elements = frames[depth - 1]
                            .contents_elements
                            .saturating_add((len / 4) as i64);
                        report.skipped_bulk_bytes = report.skipped_bulk_bytes.saturating_add(len);
                        cursor = submsg_end;
                    }

                    // TensorContents Field 7 (`fp64_contents` packed fixed64):
                    (MsgContext::TensorContents, 7) => {
                        if len % 8 != 0 {
                            return Err(fail(
                                ParseError::SubmessageOverrun,
                                len_offset,
                                depth,
                                last_field_number,
                                last_wire_type,
                                report,
                            ));
                        }
                        frames[depth - 1].contents_elements = frames[depth - 1]
                            .contents_elements
                            .saturating_add((len / 8) as i64);
                        report.skipped_bulk_bytes = report.skipped_bulk_bytes.saturating_add(len);
                        cursor = submsg_end;
                    }

                    // TensorContents Field 8 (`bytes_contents` repeated bytes):
                    (MsgContext::TensorContents, 8) => {
                        frames[depth - 1].contents_elements =
                            frames[depth - 1].contents_elements.saturating_add(1);
                        report.skipped_bulk_bytes = report.skipped_bulk_bytes.saturating_add(len);
                        cursor = submsg_end;
                    }

                    // InputTensor Field 3 (`shape` encoded as Packed Repeated Varints, Tag 0x1A):
                    // Each packed dimension counts toward `tags_scanned` so a 16 MiB packed shape
                    // of `0x01` dimensions is bounded by `max_tags_scanned`.
                    (MsgContext::InputTensor, 3) => {
                        while cursor < submsg_end {
                            report.tags_scanned = report.tags_scanned.saturating_add(1);
                            if config.max_tags_scanned > 0
                                && report.tags_scanned > config.max_tags_scanned
                            {
                                return Err(fail(
                                    ParseError::TagLimitExceeded {
                                        count: report.tags_scanned,
                                        max_tags: config.max_tags_scanned,
                                    },
                                    cursor,
                                    depth,
                                    last_field_number,
                                    last_wire_type,
                                    report,
                                ));
                            }
                            let dim_offset = cursor;
                            let raw_dim = match decode_varint(slice, &mut cursor) {
                                Ok(d) => d,
                                Err(e) => {
                                    return Err(fail(
                                        e,
                                        dim_offset,
                                        depth,
                                        last_field_number,
                                        last_wire_type,
                                        report,
                                    ));
                                }
                            };
                            if cursor > submsg_end {
                                return Err(fail(
                                    ParseError::SubmessageOverrun,
                                    cursor,
                                    depth,
                                    last_field_number,
                                    last_wire_type,
                                    report,
                                ));
                            }
                            let frame = &mut frames[depth - 1];
                            if let Err(e) = accumulate_dimension(
                                raw_dim,
                                &mut frame.shape_product,
                                &mut frame.has_shape,
                                config.max_tensor_elements,
                            ) {
                                return Err(fail(
                                    e,
                                    dim_offset,
                                    depth,
                                    last_field_number,
                                    last_wire_type,
                                    report,
                                ));
                            }
                        }
                    }

                    // All other Wire Type 2 fields at any depth are stepped into as GenericSubmsg.
                    _ => {
                        if depth >= max_depth {
                            return Err(fail(
                                ParseError::NestingLimitExceeded(depth + 1),
                                tag_offset,
                                depth + 1,
                                last_field_number,
                                last_wire_type,
                                report,
                            ));
                        }
                        frames[depth] = ScopeFrame {
                            context: MsgContext::GenericSubmsg,
                            end_offset: submsg_end,
                            ..Default::default()
                        };
                        depth += 1;
                        if depth > report.max_depth_reached {
                            report.max_depth_reached = depth;
                        }
                    }
                }
            }

            // Wire Type 5: 32-bit Fixed-Width (4 bytes)
            5 => {
                let Some(next_cursor) = cursor.checked_add(4) else {
                    return Err(fail(
                        ParseError::UnexpectedEof,
                        cursor,
                        depth,
                        last_field_number,
                        last_wire_type,
                        report,
                    ));
                };
                if next_cursor > slice.len() {
                    return Err(fail(
                        ParseError::UnexpectedEof,
                        cursor,
                        depth,
                        last_field_number,
                        last_wire_type,
                        report,
                    ));
                }
                if next_cursor > scope_end {
                    return Err(fail(
                        ParseError::SubmessageOverrun,
                        cursor,
                        depth,
                        last_field_number,
                        last_wire_type,
                        report,
                    ));
                }
                if current_context == MsgContext::TensorContents && field_number == 6 {
                    frames[depth - 1].contents_elements =
                        frames[depth - 1].contents_elements.saturating_add(1);
                }
                cursor = next_cursor;
            }

            // Deprecated or undefined wire types (3, 4, 6, 7)
            wt => {
                return Err(fail(
                    ParseError::InvalidWireType(wt),
                    tag_offset,
                    depth,
                    last_field_number,
                    last_wire_type,
                    report,
                ));
            }
        }
    }

    // Pop any frames that end_offset == slice.len(); if any active frame expected more bytes,
    // report an overrun or unexpected EOF.
    while depth > 0 {
        let end_offset = frames[depth - 1].end_offset;
        if cursor > end_offset {
            return Err(fail(
                ParseError::SubmessageOverrun,
                cursor,
                depth,
                last_field_number,
                last_wire_type,
                report,
            ));
        }
        if cursor < end_offset {
            return Err(fail(
                ParseError::UnexpectedEof,
                cursor,
                depth,
                last_field_number,
                last_wire_type,
                report,
            ));
        }
        depth -= 1;
        if let Err(e) = close_scope_frame(
            &mut frames,
            depth,
            &mut input_specs,
            &mut input_specs_len,
            &mut report,
            config,
        ) {
            return Err(fail(
                e,
                cursor,
                depth,
                last_field_number,
                last_wire_type,
                report,
            ));
        }
    }

    // Per-input tensor parity validation:
    // 1. Every `InferInputTensor` must declare a valid `shape` and known `datatype` (`elem_byte_width` or `BYTES`).
    // 2. If `InferTensorContents` (`InputTensor` Field 5) is used, `raw_input_contents` (Root Field 7) must not be
    //    mixed in, and `contents_elements` must equal `shape_product`.
    // 3. Otherwise, `raw_input_contents` (Root Field 7) must provide one chunk per `InferInputTensor` in exact input
    //    order:
    //    - Fixed-width tensors (`FP32`, `INT64`, etc.) require `chunk.len == shape_product * elem_byte_width`.
    //    - Variable-width `BYTES` tensors require `shape_product` valid 4-byte little-endian length-prefixed strings
    //      that partition `chunk` with zero overrun or trailing bytes.
    if config.enforce_payload_parity && (input_specs_len > 0 || raw_chunks_len > 0) {
        if input_specs_len == 0 && raw_chunks_len > 0 {
            return Err(fail(
                ParseError::PayloadSizeMismatch {
                    expected_bytes: 0,
                    actual_bytes: report.actual_raw_input_bytes,
                },
                cursor,
                0,
                last_field_number,
                last_wire_type,
                report,
            ));
        }

        let any_inline_contents = (0..input_specs_len).any(|i| input_specs[i].has_contents);
        if any_inline_contents && raw_chunks_len > 0 {
            return Err(fail(
                ParseError::PayloadSizeMismatch {
                    expected_bytes: report.expected_raw_input_bytes,
                    actual_bytes: report.actual_raw_input_bytes,
                },
                cursor,
                0,
                last_field_number,
                last_wire_type,
                report,
            ));
        }

        if !any_inline_contents && raw_chunks_len != input_specs_len {
            let expected_total = if report.has_bytes_tensor {
                report
                    .expected_raw_input_bytes
                    .saturating_add(report.min_variable_input_bytes)
            } else {
                report.expected_raw_input_bytes
            };
            return Err(fail(
                ParseError::PayloadSizeMismatch {
                    expected_bytes: expected_total.max(input_specs_len),
                    actual_bytes: report.actual_raw_input_bytes,
                },
                cursor,
                0,
                last_field_number,
                last_wire_type,
                report,
            ));
        }

        for i in 0..input_specs_len {
            let spec = input_specs[i];
            if !spec.has_shape
                || !spec.has_datatype
                || (spec.elem_byte_width.is_none() && !spec.is_bytes_dtype)
            {
                return Err(fail(
                    ParseError::PayloadSizeMismatch {
                        expected_bytes: 0,
                        actual_bytes: report.actual_raw_input_bytes,
                    },
                    cursor,
                    0,
                    last_field_number,
                    last_wire_type,
                    report,
                ));
            }

            if any_inline_contents {
                let width = spec.elem_byte_width.unwrap_or(4) as usize;
                let expected_elems = usize::try_from(spec.shape_product).unwrap_or(usize::MAX);
                let actual_elems = usize::try_from(spec.contents_elements).unwrap_or(0);
                if !spec.has_contents || actual_elems != expected_elems {
                    return Err(fail(
                        ParseError::PayloadSizeMismatch {
                            expected_bytes: expected_elems.saturating_mul(width),
                            actual_bytes: actual_elems.saturating_mul(width),
                        },
                        cursor,
                        0,
                        last_field_number,
                        last_wire_type,
                        report,
                    ));
                }
            } else {
                let chunk = raw_chunks[i];
                if let Some(width) = spec.elem_byte_width {
                    let expected_chunk_bytes = spec
                        .shape_product
                        .checked_mul(width)
                        .and_then(|b| usize::try_from(b).ok())
                        .unwrap_or(usize::MAX);
                    if chunk.len != expected_chunk_bytes {
                        return Err(fail(
                            ParseError::PayloadSizeMismatch {
                                expected_bytes: expected_chunk_bytes,
                                actual_bytes: chunk.len,
                            },
                            cursor,
                            0,
                            last_field_number,
                            last_wire_type,
                            report,
                        ));
                    }
                } else if spec.is_bytes_dtype {
                    let expected_elems = usize::try_from(spec.shape_product).unwrap_or(usize::MAX);
                    let chunk_slice = &slice[chunk.start..chunk.start + chunk.len];
                    if let Err(e) = verify_bytes_chunk(chunk_slice, expected_elems) {
                        return Err(fail(
                            e,
                            chunk.start,
                            0,
                            last_field_number,
                            last_wire_type,
                            report,
                        ));
                    }
                }
            }
        }
    }

    Ok(report)
}

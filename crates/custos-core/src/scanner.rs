use crate::error::ParseError;
use crate::varint::decode_varint;

/// Compile-time maximum submessage stack depth (zero heap allocations).
pub const MAX_DEPTH: usize = 16;

/// Default maximum tensor elements (`16,777,216` elements = 64 MiB for FP32/INT32).
pub const DEFAULT_MAX_TENSOR_ELEMENTS: i64 = 16_777_216;

/// Runtime configuration for the zero-copy Protobuf wire scanner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScannerConfig {
    /// Maximum allowed Protobuf submessage nesting depth (up to `MAX_DEPTH = 16`).
    pub max_nesting_depth: usize,
    /// Maximum total elements permitted per `InferInputTensor` shape product.
    pub max_tensor_elements: i64,
}

impl Default for ScannerConfig {
    fn default() -> Self {
        Self {
            max_nesting_depth: MAX_DEPTH,
            max_tensor_elements: DEFAULT_MAX_TENSOR_ELEMENTS,
        }
    }
}

/// Zero-allocation telemetry report produced by a single-pass wire scan.
///
/// Useful for operational logging, threshold tuning, and capacity planning in production clusters.
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
    /// Total bytes skipped in $O(1)$ time (strings and `raw_input_contents` float buffers).
    pub skipped_bulk_bytes: usize,
    /// Total Protobuf TLV tags decoded during the scan.
    pub tags_scanned: usize,
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
    /// Any other submessage (`InferRequestedOutputTensor`, `InferParameter`, or unknown submessage).
    GenericSubmsg,
}

/// Stack-allocated frame tracking an active length-delimited Protobuf submessage scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScopeFrame {
    pub context: MsgContext,
    pub end_offset: usize,
    pub shape_product: i64,
    pub has_shape: bool,
}

impl Default for ScopeFrame {
    fn default() -> Self {
        Self {
            context: MsgContext::GenericSubmsg,
            end_offset: 0,
            shape_product: 1,
            has_shape: false,
        }
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
    if dim <= 0 {
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

/// Records a completed `InputTensor` frame's shape statistics into the `ScanReport`.
#[inline]
fn finalize_frame_stats(frame: &ScopeFrame, report: &mut ScanReport) {
    if frame.context == MsgContext::InputTensor && frame.has_shape {
        if frame.shape_product > report.max_tensor_elements_seen {
            report.max_tensor_elements_seen = frame.shape_product;
        }
        report.total_tensor_elements = report
            .total_tensor_elements
            .saturating_add(frame.shape_product);
    }
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
///
/// Note: `ScanFailure` is returned by value on the stack because `custos-core` is `#![no_std]`
/// and performs zero heap allocations (no `Box`).
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

    // Protobuf field number 0 is never legal on the wire; a leading byte with `byte >> 3 == 0`
    // indicates a malformed, truncated, or compressed gRPC frame header.
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

#[allow(clippy::result_large_err)]
fn inspect_raw_protobuf_with_offset(
    slice: &[u8],
    config: &ScannerConfig,
    base_offset: usize,
    grpc_framed: bool,
    total_payload_bytes: usize,
) -> Result<ScanReport, ScanFailure> {
    let mut frames: [ScopeFrame; MAX_DEPTH] = [ScopeFrame::default(); MAX_DEPTH];
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
            finalize_frame_stats(&frames[depth - 1], &mut report);
            depth -= 1;
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

                // Check against parent scope boundary first if inside a submessage,
                // or overall slice boundary if at Root.
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
                    // Root string/bytes fields:
                    // 1 = model_name, 2 = model_version, 3 = id, 7 = raw_input_contents
                    // Skip in O(1) time without parsing interior bytes as Protobuf tags.
                    (MsgContext::Root, 1 | 2 | 3 | 7) => {
                        report.skipped_bulk_bytes = report.skipped_bulk_bytes.saturating_add(len);
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
                            shape_product: 1,
                            has_shape: false,
                        };
                        depth += 1;
                        report.input_tensor_count = report.input_tensor_count.saturating_add(1);
                        if depth > report.max_depth_reached {
                            report.max_depth_reached = depth;
                        }
                    }

                    // InputTensor string/contents fields:
                    // 1 = name, 2 = datatype, 5 = contents
                    // Skip in O(1) time.
                    (MsgContext::InputTensor, 1 | 2 | 5) => {
                        report.skipped_bulk_bytes = report.skipped_bulk_bytes.saturating_add(len);
                        cursor = submsg_end;
                    }

                    // InputTensor Field 3 (`shape` encoded as Packed Repeated Varints, Tag 0x1A):
                    (MsgContext::InputTensor, 3) => {
                        while cursor < submsg_end {
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

                    // All other Wire Type 2 fields at any depth (e.g. parameters, outputs,
                    // or unknown/attacker-crafted fields) are stepped into as GenericSubmsg.
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
                            shape_product: 1,
                            has_shape: false,
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

    // Pop any frames that end exactly at `slice.len()`; if any active frame expected more bytes,
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
        finalize_frame_stats(&frames[depth - 1], &mut report);
        depth -= 1;
    }

    Ok(report)
}

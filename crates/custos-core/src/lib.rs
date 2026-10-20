#![no_std]
#![forbid(unsafe_code)]

pub mod error;
pub mod scanner;
pub mod varint;

pub use error::ParseError;
pub use scanner::{
    DEFAULT_MAX_TENSOR_ELEMENTS, MAX_DEPTH, MsgContext, ScanFailure, ScanReport, ScannerConfig,
    ScopeFrame, inspect_grpc_frame, inspect_payload, inspect_payload_with_report,
    inspect_raw_protobuf, inspect_raw_protobuf_with_report,
};
pub use varint::decode_varint;

#[cfg(test)]
mod tests {
    extern crate std;
    use std::vec;
    use std::vec::Vec;

    use super::*;

    /// Helper: encodes a `u64` value as a Base-128 varint into `buf`.
    fn push_varint(buf: &mut Vec<u8>, mut val: u64) {
        loop {
            let mut byte = (val & 0x7F) as u8;
            val >>= 7;
            if val != 0 {
                byte |= 0x80;
            }
            buf.push(byte);
            if val == 0 {
                break;
            }
        }
    }

    /// Helper: wraps a raw Protobuf message in a 5-byte uncompressed gRPC frame header.
    fn wrap_grpc_frame(protobuf: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(5 + protobuf.len());
        out.push(0x00);
        out.extend_from_slice(&(protobuf.len() as u32).to_be_bytes());
        out.extend_from_slice(protobuf);
        out
    }

    /// Helper: builds a nested Wire Type 2 submessage chain of exact `depth` levels.
    fn build_nesting_bomb(depth: usize) -> Vec<u8> {
        let mut payload = Vec::new();
        // Start from the innermost submessage (empty length-delimited field) and wrap outward.
        for _ in 0..depth {
            let mut outer = Vec::with_capacity(2 + payload.len());
            // Tag 0x22 = Field 4 (`parameters`), Wire Type 2
            outer.push(0x22);
            push_varint(&mut outer, payload.len() as u64);
            outer.extend_from_slice(&payload);
            payload = outer;
        }
        payload
    }

    /// Helper: builds a KServe v2 `InferInputTensor` (`Field 5` of `ModelInferRequest`)
    /// with either packed (`0x1A`) or unpacked (`0x18`) shape dimensions.
    fn build_input_tensor(name: &str, datatype: &str, dims: &[i64], packed: bool) -> Vec<u8> {
        let mut tensor = Vec::new();
        // Field 1: name (0x0A)
        tensor.push(0x0A);
        push_varint(&mut tensor, name.len() as u64);
        tensor.extend_from_slice(name.as_bytes());

        // Field 2: datatype (0x12)
        tensor.push(0x12);
        push_varint(&mut tensor, datatype.len() as u64);
        tensor.extend_from_slice(datatype.as_bytes());

        if packed {
            // Field 3 (packed): tag 0x1A + length + concatenated varints
            let mut packed_dims = Vec::new();
            for &d in dims {
                push_varint(&mut packed_dims, d as u64);
            }
            tensor.push(0x1A);
            push_varint(&mut tensor, packed_dims.len() as u64);
            tensor.extend_from_slice(&packed_dims);
        } else {
            // Field 3 (unpacked): repeated tag 0x18 + varint
            for &d in dims {
                tensor.push(0x18);
                push_varint(&mut tensor, d as u64);
            }
        }

        let mut root_field = Vec::new();
        // Root Field 5: inputs (0x2A)
        root_field.push(0x2A);
        push_varint(&mut root_field, tensor.len() as u64);
        root_field.extend_from_slice(&tensor);
        root_field
    }

    #[test]
    fn test_clean_kserve_v2_request_with_float_bytes() {
        let mut msg = Vec::new();
        // Field 1: model_name = "resnet50"
        msg.extend_from_slice(b"\x0a\x08resnet50");
        // Field 2: model_version = "1"
        msg.extend_from_slice(b"\x12\x011");
        // Field 3: id = "req-001"
        msg.extend_from_slice(b"\x1a\x07req-001");
        // Field 5: inputs with valid shape [1, 3, 224, 224]
        msg.extend_from_slice(&build_input_tensor(
            "input__0",
            "FP32",
            &[1, 3, 224, 224],
            true,
        ));
        // Field 7: raw_input_contents containing adversarial byte patterns (0xFF, 0x18, 0x1A, etc.)
        // that would trigger false positives if scanned as Protobuf tags.
        let mut raw_floats = vec![0xFFu8; 4096];
        for (i, b) in raw_floats.iter_mut().enumerate() {
            *b = ((i * 73 + 19) & 0xFF) as u8;
        }
        msg.push(0x3A);
        push_varint(&mut msg, raw_floats.len() as u64);
        msg.extend_from_slice(&raw_floats);

        let config = ScannerConfig::default();
        assert_eq!(inspect_raw_protobuf(&msg, &config), Ok(()));

        let grpc_framed = wrap_grpc_frame(&msg);
        assert_eq!(inspect_payload(&grpc_framed, &config), Ok(()));
        assert_eq!(inspect_grpc_frame(&grpc_framed, &config), Ok(()));
    }

    #[test]
    fn test_nesting_bombs_15_16_17_100() {
        let config = ScannerConfig::default();

        // Depth 15 -> PASS
        let bomb_15 = build_nesting_bomb(15);
        assert_eq!(inspect_payload(&wrap_grpc_frame(&bomb_15), &config), Ok(()));

        // Depth 16 -> PASS (exact ceiling MAX_DEPTH = 16)
        let bomb_16 = build_nesting_bomb(16);
        assert_eq!(inspect_payload(&wrap_grpc_frame(&bomb_16), &config), Ok(()));

        // Depth 17 -> REJECT
        let bomb_17 = build_nesting_bomb(17);
        assert_eq!(
            inspect_payload(&wrap_grpc_frame(&bomb_17), &config),
            Err(ParseError::NestingLimitExceeded(17))
        );

        // Depth 100 -> REJECT at depth 17
        let bomb_100 = build_nesting_bomb(100);
        assert_eq!(
            inspect_payload(&wrap_grpc_frame(&bomb_100), &config),
            Err(ParseError::NestingLimitExceeded(17))
        );
    }

    #[test]
    fn test_shape_overflows_packed_and_unpacked() {
        let config = ScannerConfig::default();

        for packed in [true, false] {
            // 1. Negative dimension -1 (10-byte two's complement varint)
            let neg_dim = build_input_tensor("INPUT0", "FP32", &[1, -1, 224], packed);
            assert_eq!(
                inspect_payload(&wrap_grpc_frame(&neg_dim), &config),
                Err(ParseError::NegativeDimension(-1))
            );

            // 2. 64-bit integer wraparound: [2, 4611686018427387904, 2]
            // 2 * 2^62 = 2^63 (overflows i64::MAX = 2^63 - 1)
            let wrap_dims =
                build_input_tensor("INPUT0", "FP32", &[2, 4_611_686_018_427_387_904, 2], packed);
            assert_eq!(
                inspect_payload(&wrap_grpc_frame(&wrap_dims), &config),
                Err(ParseError::ShapeOverflow)
            );

            // 3. Massive OOM: [1024, 1024, 1024] = 1,073,741,824 > 16,777,216
            // Note: Checked progressively at [1024, 1024, 1024] -> 1,073,741,824
            let oom_dims = build_input_tensor("INPUT0", "FP32", &[1024, 1024, 1024], packed);
            assert_eq!(
                inspect_payload(&wrap_grpc_frame(&oom_dims), &config),
                Err(ParseError::BufferCeilingExceeded {
                    elements: 1_073_741_824,
                    max_elements: DEFAULT_MAX_TENSOR_ELEMENTS,
                })
            );
        }
    }

    #[test]
    fn test_field_reordered_exploit_16kb_field7_before_field5() {
        let config = ScannerConfig::default();
        let mut msg = Vec::new();

        // Prepend a 16 KB Field 7 (`raw_input_contents`) BEFORE the malicious `Field 5` (`inputs`)
        let raw_tensor_bytes = vec![0x42u8; 16 * 1024];
        msg.push(0x3A); // Field 7, Wire Type 2
        push_varint(&mut msg, raw_tensor_bytes.len() as u64);
        msg.extend_from_slice(&raw_tensor_bytes);

        // Append malicious Field 5 (`inputs`) with OOM shape [1024, 1024, 1024]
        msg.extend_from_slice(&build_input_tensor(
            "evil_tensor",
            "FP32",
            &[1024, 1024, 1024],
            true,
        ));

        let framed = wrap_grpc_frame(&msg);
        assert_eq!(
            inspect_payload(&framed, &config),
            Err(ParseError::BufferCeilingExceeded {
                elements: 1_073_741_824,
                max_elements: DEFAULT_MAX_TENSOR_ELEMENTS,
            })
        );
    }

    #[test]
    fn test_malformed_varints_and_truncated_eof() {
        let config = ScannerConfig::default();

        // 1. 11 bytes of 0xFF (exceeds 10-byte limit; rejected at 10th byte `0xFF > 0x01`)
        let mut eleven_ff = vec![0x08]; // Field 1, Wire Type 0
        eleven_ff.extend_from_slice(&[0xFF; 11]);
        assert_eq!(
            inspect_raw_protobuf(&eleven_ff, &config),
            Err(ParseError::MalformedVarint)
        );

        // 2. 10th-byte overflow: 9x 0xFF followed by 0x7F (value > u64::MAX)
        let mut tenth_overflow = vec![0x08]; // Field 1, Wire Type 0
        tenth_overflow.extend_from_slice(&[0xFF; 9]);
        tenth_overflow.push(0x7F);
        assert_eq!(
            inspect_raw_protobuf(&tenth_overflow, &config),
            Err(ParseError::MalformedVarint)
        );

        // 3. Valid 10-byte varint with 10th byte == 0x01 (u64::MAX) in a non-shape field
        let mut valid_ten_byte = vec![0x08]; // Root Field 1 as Wire Type 0
        valid_ten_byte.extend_from_slice(&[0xFF; 9]);
        valid_ten_byte.push(0x01);
        assert_eq!(inspect_raw_protobuf(&valid_ten_byte, &config), Ok(()));

        // 4. Truncated EOF inside varint
        let truncated_varint = [0x08, 0x80, 0x80];
        assert_eq!(
            inspect_raw_protobuf(&truncated_varint, &config),
            Err(ParseError::UnexpectedEof)
        );

        // 5. Truncated EOF inside length-delimited field
        let truncated_len = [0x0A, 0x10, b'a', b'b', b'c'];
        assert_eq!(
            inspect_raw_protobuf(&truncated_len, &config),
            Err(ParseError::UnexpectedEof)
        );
    }

    #[test]
    fn test_grpc_header_and_submessage_overrun_errors() {
        let config = ScannerConfig::default();

        // Empty frame
        assert_eq!(inspect_payload(&[], &config), Err(ParseError::EmptyFrame));

        // Compressed gRPC frame (flag = 0x01)
        let compressed = [0x01, 0x00, 0x00, 0x00, 0x02, 0x08, 0x01];
        assert_eq!(
            inspect_payload(&compressed, &config),
            Err(ParseError::UnsupportedCompression(0x01))
        );

        // gRPC frame length mismatch
        let mismatch = [0x00, 0x00, 0x00, 0x00, 0x10, 0x08, 0x01];
        assert_eq!(
            inspect_payload(&mismatch, &config),
            Err(ParseError::FrameLengthMismatch {
                declared: 16,
                actual: 2
            })
        );

        // Submessage overrun: outer submessage declares length 2, inner field requires 4 bytes
        let overrun = [0x22, 0x02, 0x08, 0x80, 0x80, 0x01];
        assert_eq!(
            inspect_raw_protobuf(&overrun, &config),
            Err(ParseError::SubmessageOverrun)
        );

        // Invalid wire type 3 (Start group)
        let bad_wt = [0x0B]; // Field 1, Wire Type 3
        assert_eq!(
            inspect_raw_protobuf(&bad_wt, &config),
            Err(ParseError::InvalidWireType(3))
        );
    }

    #[test]
    fn test_scan_report_and_failure_diagnostics() {
        let config = ScannerConfig::default();
        let mut msg = Vec::new();
        msg.extend_from_slice(b"\x0a\x08resnet50");
        msg.extend_from_slice(&build_input_tensor(
            "input__0",
            "FP32",
            &[1, 3, 224, 224],
            true,
        ));
        let raw_floats = vec![0xAAu8; 2048];
        msg.push(0x3A);
        push_varint(&mut msg, raw_floats.len() as u64);
        msg.extend_from_slice(&raw_floats);

        let framed = wrap_grpc_frame(&msg);
        let report = inspect_payload_with_report(&framed, &config).expect("valid report");
        assert!(report.grpc_framed);
        assert_eq!(report.payload_bytes, framed.len());
        assert_eq!(report.protobuf_bytes, msg.len());
        assert_eq!(report.input_tensor_count, 1);
        assert_eq!(report.max_tensor_elements_seen, 150_528);
        assert_eq!(report.total_tensor_elements, 150_528);
        assert_eq!(report.max_depth_reached, 1);
        assert!(report.skipped_bulk_bytes >= 2048);

        // Verify failure diagnostics report exact byte offset and partial stats
        let oom = wrap_grpc_frame(&build_input_tensor(
            "bad_tensor",
            "FP32",
            &[1024, 1024, 1024],
            true,
        ));
        let failure = inspect_payload_with_report(&oom, &config).expect_err("expected failure");
        assert!(matches!(
            failure.error,
            ParseError::BufferCeilingExceeded {
                elements: 1_073_741_824,
                ..
            }
        ));
        assert!(failure.byte_offset > 5);
        assert_eq!(failure.depth_at_failure, 1);
        assert_eq!(failure.last_field_number, 3);
        assert_eq!(failure.partial_report.input_tensor_count, 1);
    }
}

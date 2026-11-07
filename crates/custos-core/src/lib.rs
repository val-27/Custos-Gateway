#![no_std]
#![forbid(unsafe_code)]

pub mod error;
pub mod scanner;
pub mod varint;

pub use error::ParseError;
pub use scanner::{
    DEFAULT_MAX_TAGS_SCANNED, DEFAULT_MAX_TENSOR_ELEMENTS, MAX_DEPTH, MsgContext, ScanFailure,
    ScanReport, ScannerConfig, ScopeFrame, inspect_grpc_frame, inspect_payload,
    inspect_payload_with_report, inspect_raw_protobuf, inspect_raw_protobuf_with_report,
    kserve_datatype_byte_width,
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
        // Exact FP32 byte length for [1, 3, 224, 224] = 150,528 * 4 = 602,112 bytes.
        let mut raw_floats = vec![0xFFu8; 150_528 * 4];
        for (i, b) in raw_floats.iter_mut().take(4096).enumerate() {
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
        let raw_floats = vec![0xAAu8; 150_528 * 4];
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
        assert_eq!(report.expected_raw_input_bytes, 602_112);
        assert_eq!(report.actual_raw_input_bytes, 602_112);
        assert_eq!(report.max_depth_reached, 1);
        assert!(report.skipped_bulk_bytes >= 602_112);

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

    #[test]
    fn test_cve_2025_23320_shape_to_payload_buffer_parity_mismatch() {
        let config = ScannerConfig::default();

        // Declare [4, 3, 224, 224] FP32 (2,408,448 bytes) but supply only 64 bytes in Field 7
        let mut underread = Vec::new();
        underread.extend_from_slice(b"\x0a\x08resnet50");
        underread.extend_from_slice(&build_input_tensor(
            "input_tensor",
            "FP32",
            &[4, 3, 224, 224],
            true,
        ));
        underread.push(0x3A); // Field 7: raw_input_contents
        push_varint(&mut underread, 64);
        underread.extend_from_slice(&[0u8; 64]);

        assert_eq!(
            inspect_payload(&wrap_grpc_frame(&underread), &config),
            Err(ParseError::PayloadSizeMismatch {
                expected_bytes: 2_408_448,
                actual_bytes: 64,
            })
        );
    }

    #[test]
    fn test_cve_2021_22569_unknown_field_ast_flood_tag_limit() {
        let config = ScannerConfig::default();
        let mut flood = Vec::new();
        flood.extend_from_slice(b"\x0a\x08resnet50");
        // Append 1,200 shallow unknown varint fields (Tag 0x78 = Field 15, Wire Type 0)
        for _ in 0..1200 {
            flood.push(0x78);
            flood.push(0x01);
        }

        assert_eq!(
            inspect_payload(&wrap_grpc_frame(&flood), &config),
            Err(ParseError::TagLimitExceeded {
                count: DEFAULT_MAX_TAGS_SCANNED + 1,
                max_tags: DEFAULT_MAX_TAGS_SCANNED,
            })
        );
    }

    #[test]
    fn test_benign_bytes_tensor_single_and_multi_input_no_false_positive() {
        let config = ScannerConfig::default();

        // 1. Single-input BYTES tensor with 2 variable-length string elements ("hello", "custos_gateway")
        let mut raw_bytes_chunk = Vec::new();
        for s in [b"hello".as_slice(), b"custos_gateway".as_slice()] {
            raw_bytes_chunk.extend_from_slice(&(s.len() as u32).to_le_bytes());
            raw_bytes_chunk.extend_from_slice(s);
        }
        let mut single_bytes_msg = Vec::new();
        single_bytes_msg.extend_from_slice(b"\x0a\x08text_cls");
        single_bytes_msg.extend_from_slice(&build_input_tensor("PROMPT", "BYTES", &[1, 2], true));
        single_bytes_msg.push(0x3A); // Field 7: raw_input_contents
        push_varint(&mut single_bytes_msg, raw_bytes_chunk.len() as u64);
        single_bytes_msg.extend_from_slice(&raw_bytes_chunk);

        let report =
            inspect_payload_with_report(&wrap_grpc_frame(&single_bytes_msg), &config).unwrap();
        assert!(report.has_bytes_tensor);
        assert_eq!(report.min_variable_input_bytes, 8);
        assert_eq!(report.actual_raw_input_bytes, raw_bytes_chunk.len());

        // 2. Multi-input request mixing fixed-width FP32 [1, 4] (16 B) + variable-length BYTES [1, 2] (27 B)
        let mut multi_msg = Vec::new();
        multi_msg.extend_from_slice(b"\x0a\x0amultimodal");
        multi_msg.extend_from_slice(&build_input_tensor("IMAGE_VEC", "FP32", &[1, 4], true));
        multi_msg.extend_from_slice(&build_input_tensor("TEXT_TOKENS", "BYTES", &[1, 2], true));
        // Chunk 0: 16 bytes for FP32 [1, 4]
        multi_msg.push(0x3A);
        push_varint(&mut multi_msg, 16);
        multi_msg.extend_from_slice(&[0u8; 16]);
        // Chunk 1: 27 bytes for BYTES [1, 2]
        multi_msg.push(0x3A);
        push_varint(&mut multi_msg, raw_bytes_chunk.len() as u64);
        multi_msg.extend_from_slice(&raw_bytes_chunk);

        assert_eq!(
            inspect_payload(&wrap_grpc_frame(&multi_msg), &config),
            Ok(())
        );

        // 3. Malicious BYTES under-read: declares [100] BYTES elements (requires >= 400 bytes of 4B length prefixes),
        // but supplies only 16 bytes in raw_input_contents.
        let mut bytes_underread = Vec::new();
        bytes_underread.extend_from_slice(b"\x0a\x08text_cls");
        bytes_underread.extend_from_slice(&build_input_tensor("PROMPT", "BYTES", &[100], true));
        bytes_underread.push(0x3A);
        push_varint(&mut bytes_underread, 16);
        bytes_underread.extend_from_slice(&[0u8; 16]);

        assert_eq!(
            inspect_payload(&wrap_grpc_frame(&bytes_underread), &config),
            Err(ParseError::PayloadSizeMismatch {
                expected_bytes: 400,
                actual_bytes: 16,
            })
        );
    }

    #[test]
    fn test_cve_2024_0088_shared_memory_parameters_blocked_on_model_infer() {
        let config = ScannerConfig::default();

        // Build an InferInputTensor that includes Field 4 (`parameters` map entry)
        // with key = "shared_memory_region" and string_param = "malicious_shm"
        let mut param_val = Vec::new();
        param_val.extend_from_slice(b"\x12\x0dmalicious_shm"); // Field 2 (`string_param`) inside InferParameter
        let mut map_entry = Vec::new();
        map_entry.extend_from_slice(b"\x0a\x14shared_memory_region"); // Field 1 (`key`)
        map_entry.push(0x12); // Field 2 (`value` -> InferParameter)
        push_varint(&mut map_entry, param_val.len() as u64);
        map_entry.extend_from_slice(&param_val);

        let mut tensor = Vec::new();
        tensor.extend_from_slice(b"\x0a\x0cinput_tensor\x12\x04FP32\x1a\x04\x01\x03\xe0\x01");
        tensor.push(0x22); // Field 4 (`parameters`) inside InferInputTensor
        push_varint(&mut tensor, map_entry.len() as u64);
        tensor.extend_from_slice(&map_entry);

        let mut msg = Vec::new();
        msg.extend_from_slice(b"\x0a\x08resnet50");
        msg.push(0x2A); // Root Field 5 (`inputs`)
        push_varint(&mut msg, tensor.len() as u64);
        msg.extend_from_slice(&tensor);

        assert_eq!(
            inspect_payload(&wrap_grpc_frame(&msg), &config),
            Err(ParseError::UnauthorizedSharedMemoryParameter)
        );
    }
}

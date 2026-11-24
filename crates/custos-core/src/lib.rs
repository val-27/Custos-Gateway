#![no_std]
#![forbid(unsafe_code)]

pub mod error;
pub mod scanner;
pub mod varint;

pub use error::ParseError;
pub use scanner::{
    DEFAULT_MAX_TAGS_SCANNED, DEFAULT_MAX_TENSOR_ELEMENTS, MAX_DEPTH, MAX_INPUT_TENSORS,
    MsgContext, ScanFailure, ScanReport, ScannerConfig, ScopeFrame, inspect_grpc_frame,
    inspect_payload, inspect_payload_with_report, inspect_raw_protobuf,
    inspect_raw_protobuf_with_report, kserve_datatype_byte_width, kserve_datatype_contents_field,
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
    fn test_kserve_shape_to_payload_buffer_parity_mismatch() {
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
        // with key = "shared_memory_region" and string_param = "malicious_shm" (Field 3 = 0x1A)
        let mut param_val = Vec::new();
        param_val.extend_from_slice(b"\x1a\x0dmalicious_shm"); // Field 3 (`string_param`) inside InferParameter
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

    #[test]
    fn test_per_chunk_swap_and_missing_field7_and_inline_contents() {
        let config = ScannerConfig::default();

        // 1. Per-chunk swap (sums match: 16 + 4 == 4 + 16, but chunk 0 != input 0)
        let mut swapped = Vec::new();
        swapped.extend_from_slice(b"\x0a\x08resnet50");
        swapped.extend_from_slice(&build_input_tensor("IN0", "FP32", &[4], true)); // expects 16 B
        swapped.extend_from_slice(&build_input_tensor("IN1", "FP32", &[1], true)); // expects 4 B
        // Chunk 0: 4 bytes (wrong for IN0)
        swapped.push(0x3A);
        push_varint(&mut swapped, 4);
        swapped.extend_from_slice(&[0u8; 4]);
        // Chunk 1: 16 bytes (wrong for IN1)
        swapped.push(0x3A);
        push_varint(&mut swapped, 16);
        swapped.extend_from_slice(&[0u8; 16]);
        assert_eq!(
            inspect_payload(&wrap_grpc_frame(&swapped), &config),
            Err(ParseError::PayloadSizeMismatch {
                expected_bytes: 16,
                actual_bytes: 4,
            })
        );

        // 2. Missing Field 7 and missing Field 5 contents on an InputTensor
        let mut no_payload = Vec::new();
        no_payload.extend_from_slice(b"\x0a\x08resnet50");
        no_payload.extend_from_slice(&build_input_tensor("IN0", "FP32", &[4], true));
        assert!(matches!(
            inspect_payload(&wrap_grpc_frame(&no_payload), &config),
            Err(ParseError::PayloadSizeMismatch { .. })
        ));

        // 3. Inline InferTensorContents (Field 5 inside InputTensor):
        //    3a. Benign packed fp32_contents (Field 6, Wire Type 2, 16 bytes = 4 floats with 0xFF bytes) -> Ok(())
        let mut contents_ok = Vec::new();
        contents_ok.push(0x32); // Field 6 (fp32_contents), Wire Type 2 (packed)
        push_varint(&mut contents_ok, 16);
        contents_ok.extend_from_slice(&[0xFFu8; 16]);
        let mut tensor_ok = Vec::new();
        tensor_ok.extend_from_slice(b"\x0a\x03IN0\x12\x04FP32\x1a\x01\x04");
        tensor_ok.push(0x2A); // Field 5 (contents)
        push_varint(&mut tensor_ok, contents_ok.len() as u64);
        tensor_ok.extend_from_slice(&contents_ok);
        let mut msg_ok = Vec::new();
        msg_ok.extend_from_slice(b"\x0a\x08resnet50");
        msg_ok.push(0x2A);
        push_varint(&mut msg_ok, tensor_ok.len() as u64);
        msg_ok.extend_from_slice(&tensor_ok);
        assert_eq!(inspect_payload(&wrap_grpc_frame(&msg_ok), &config), Ok(()));

        //    3b. Malicious InferTensorContents: shape [16777216] FP32 with only 1 unpacked fp32_contents element -> REJECT
        let mut contents_bad = Vec::new();
        contents_bad.push(0x35); // Field 6 (fp32_contents), Wire Type 5 (fixed32)
        contents_bad.extend_from_slice(&[0u8; 4]);
        let mut tensor_bad = Vec::new();
        tensor_bad.extend_from_slice(b"\x0a\x03IN0\x12\x04FP32");
        tensor_bad.push(0x1A);
        let mut shape_bytes = Vec::new();
        push_varint(&mut shape_bytes, 16_777_216);
        push_varint(&mut tensor_bad, shape_bytes.len() as u64);
        tensor_bad.extend_from_slice(&shape_bytes);
        tensor_bad.push(0x2A); // Field 5 (contents)
        push_varint(&mut tensor_bad, contents_bad.len() as u64);
        tensor_bad.extend_from_slice(&contents_bad);
        let mut msg_bad = Vec::new();
        msg_bad.extend_from_slice(b"\x0a\x08resnet50");
        msg_bad.push(0x2A);
        push_varint(&mut msg_bad, tensor_bad.len() as u64);
        msg_bad.extend_from_slice(&tensor_bad);
        assert_eq!(
            inspect_payload(&wrap_grpc_frame(&msg_bad), &config),
            Err(ParseError::PayloadSizeMismatch {
                expected_bytes: 16_777_216 * 4,
                actual_bytes: 4,
            })
        );
    }

    #[test]
    fn test_bytes_length_prefix_overrun_and_packed_shape_tag_limit() {
        let config = ScannerConfig::default();

        // 1. BYTES element with 4-byte little-endian length prefix (100) exceeding chunk size (8 bytes)
        let mut bad_bytes = Vec::new();
        bad_bytes.extend_from_slice(b"\x0a\x08text_cls");
        bad_bytes.extend_from_slice(&build_input_tensor("PROMPT", "BYTES", &[1], true));
        bad_bytes.push(0x3A);
        push_varint(&mut bad_bytes, 8);
        bad_bytes.extend_from_slice(&100u32.to_le_bytes());
        bad_bytes.extend_from_slice(b"abcd");
        assert_eq!(
            inspect_payload(&wrap_grpc_frame(&bad_bytes), &config),
            Err(ParseError::PayloadSizeMismatch {
                expected_bytes: 104,
                actual_bytes: 8,
            })
        );

        // 2. Packed shape of 1,100 dimensions of `1` (product == 1, but exceeds DEFAULT_MAX_TAGS_SCANNED)
        let ones = vec![1i64; 1100];
        let mut packed_flood = Vec::new();
        packed_flood.extend_from_slice(b"\x0a\x08resnet50");
        packed_flood.extend_from_slice(&build_input_tensor("IN0", "FP32", &ones, true));
        assert!(matches!(
            inspect_payload(&wrap_grpc_frame(&packed_flood), &config),
            Err(ParseError::TagLimitExceeded { .. })
        ));

        // 3. Zero-axis shape [1, 0, 224, 224] with empty raw_input_contents (0 bytes) -> Ok(())
        let mut zero_axis = Vec::new();
        zero_axis.extend_from_slice(b"\x0a\x08resnet50");
        zero_axis.extend_from_slice(&build_input_tensor(
            "IN0",
            "FP32",
            &[1, 0, 224, 224],
            true,
        ));
        zero_axis.extend_from_slice(b"\x3a\x00");
        assert_eq!(
            inspect_payload(&wrap_grpc_frame(&zero_axis), &config),
            Ok(())
        );

        // 4. Empty ModelInfer protobuf inside a valid 5-byte gRPC frame (00 00 00 00 00) -> Ok(())
        assert_eq!(inspect_payload(&wrap_grpc_frame(&[]), &config), Ok(()));
    }

    /// Reference Protobuf wire model for `ModelInferRequest` following standard C++ `libprotobuf` /
    /// `prost` deserialization rules:
    /// - Arbitrary field ordering at both `ModelInferRequest` and `InferInputTensor` scopes
    /// - Repeated `shape` concatenation across unpacked (`0x18`) and packed (`0x1A`) records
    /// - Last-write-wins for scalar `name` and `datatype` fields
    /// - Submessage merging when `InferInputTensor.contents` (Field 5) appears multiple times
    /// - Per-datatype `InferTensorContents` field selection (ignoring or rejecting mismatched fields)
    #[derive(Debug, Default, Clone)]
    struct ReferenceInputTensor {
        datatype: Vec<u8>,
        shape: Vec<i64>,
        has_contents_submsg: bool,
        contents_by_field: [usize; 9],
    }

    #[derive(Debug, Default, Clone)]
    struct ReferenceModelInfer {
        inputs: Vec<ReferenceInputTensor>,
        raw_chunks: Vec<Vec<u8>>,
        has_shm_param: bool,
    }

    fn reference_parse_model_infer(
        slice: &[u8],
        max_depth: usize,
        max_tags: usize,
    ) -> Result<ReferenceModelInfer, ()> {
        #[allow(clippy::too_many_arguments)]
        fn parse_submsg(
            slice: &[u8],
            ctx: MsgContext,
            depth: usize,
            max_depth: usize,
            tags: &mut usize,
            max_tags: usize,
            model: &mut ReferenceModelInfer,
            cur_input: Option<&mut ReferenceInputTensor>,
        ) -> Result<(), ()> {
            let mut cursor = 0usize;
            let mut cur_input = cur_input;
            while cursor < slice.len() {
                let tag = decode_varint(slice, &mut cursor).map_err(|_| ())?;
                let wt = (tag & 0x07) as u8;
                let field = tag >> 3;
                if field == 0 {
                    return Err(());
                }
                *tags = tags.saturating_add(1);
                if max_tags > 0 && *tags > max_tags {
                    return Err(());
                }
                match wt {
                    0 => {
                        let val = decode_varint(slice, &mut cursor).map_err(|_| ())?;
                        match (ctx, field) {
                            (MsgContext::InputTensor, 1 | 2 | 4 | 5)
                            | (MsgContext::OutputTensor, 1 | 2)
                            | (MsgContext::TensorContents, 6..=8) => return Err(()),
                            (MsgContext::InputTensor, 3) => {
                                if let Some(inp) = cur_input.as_deref_mut() {
                                    inp.shape.push(val as i64);
                                }
                            }
                            (MsgContext::TensorContents, 1..=5) => {
                                if let Some(inp) = cur_input.as_deref_mut() {
                                    inp.contents_by_field[field as usize] += 1;
                                }
                            }
                            _ => {}
                        }
                    }
                    1 => {
                        let next = cursor.checked_add(8).ok_or(())?;
                        if next > slice.len() {
                            return Err(());
                        }
                        match (ctx, field) {
                            (MsgContext::InputTensor, 1..=5)
                            | (MsgContext::OutputTensor, 1 | 2)
                            | (MsgContext::TensorContents, 1..=6 | 8) => return Err(()),
                            (MsgContext::TensorContents, 7) => {
                                if let Some(inp) = cur_input.as_deref_mut() {
                                    inp.contents_by_field[7] += 1;
                                }
                            }
                            _ => {}
                        }
                        cursor = next;
                    }
                    2 => {
                        let len_u64 = decode_varint(slice, &mut cursor).map_err(|_| ())?;
                        let len = usize::try_from(len_u64).map_err(|_| ())?;
                        let end = cursor.checked_add(len).ok_or(())?;
                        if end > slice.len() {
                            return Err(());
                        }
                        let sub = &slice[cursor..end];
                        cursor = end;
                        match (ctx, field) {
                            (MsgContext::Root, 1..=3)
                            | (MsgContext::OutputTensor, 1)
                            | (MsgContext::InputTensor, 1) => {}
                            (MsgContext::Root, 7) => {
                                if model.raw_chunks.len() >= MAX_INPUT_TENSORS {
                                    return Err(());
                                }
                                model.raw_chunks.push(sub.to_vec());
                            }
                            (MsgContext::Root, 5) => {
                                if depth + 1 > max_depth || model.inputs.len() >= MAX_INPUT_TENSORS
                                {
                                    return Err(());
                                }
                                let mut inp = ReferenceInputTensor::default();
                                parse_submsg(
                                    sub,
                                    MsgContext::InputTensor,
                                    depth + 1,
                                    max_depth,
                                    tags,
                                    max_tags,
                                    model,
                                    Some(&mut inp),
                                )?;
                                model.inputs.push(inp);
                            }
                            (MsgContext::Root, 6) => {
                                if depth + 1 > max_depth {
                                    return Err(());
                                }
                                parse_submsg(
                                    sub,
                                    MsgContext::OutputTensor,
                                    depth + 1,
                                    max_depth,
                                    tags,
                                    max_tags,
                                    model,
                                    None,
                                )?;
                            }
                            (MsgContext::Root, 4)
                            | (MsgContext::InputTensor, 4)
                            | (MsgContext::OutputTensor, 2) => {
                                if depth + 1 > max_depth {
                                    return Err(());
                                }
                                parse_submsg(
                                    sub,
                                    MsgContext::ParameterMapEntry,
                                    depth + 1,
                                    max_depth,
                                    tags,
                                    max_tags,
                                    model,
                                    None,
                                )?;
                            }
                            (MsgContext::ParameterMapEntry, 1) => {
                                if sub.starts_with(b"shared_memory") {
                                    model.has_shm_param = true;
                                    return Err(());
                                }
                            }
                            (MsgContext::InputTensor, 2) => {
                                if let Some(inp) = cur_input.as_deref_mut() {
                                    // Protobuf scalar string: last-write-wins!
                                    inp.datatype = sub.to_vec();
                                }
                            }
                            (MsgContext::InputTensor, 3) => {
                                let mut pcur = 0usize;
                                while pcur < sub.len() {
                                    *tags = tags.saturating_add(1);
                                    if max_tags > 0 && *tags > max_tags {
                                        return Err(());
                                    }
                                    let d = decode_varint(sub, &mut pcur).map_err(|_| ())?;
                                    if let Some(inp) = cur_input.as_deref_mut() {
                                        inp.shape.push(d as i64);
                                    }
                                }
                            }
                            (MsgContext::InputTensor, 5) => {
                                if depth + 1 > max_depth {
                                    return Err(());
                                }
                                if let Some(inp) = cur_input.as_deref_mut() {
                                    inp.has_contents_submsg = true;
                                    parse_submsg(
                                        sub,
                                        MsgContext::TensorContents,
                                        depth + 1,
                                        max_depth,
                                        tags,
                                        max_tags,
                                        model,
                                        Some(inp),
                                    )?;
                                }
                            }
                            (MsgContext::TensorContents, 1..=5) => {
                                let mut pcur = 0usize;
                                while pcur < sub.len() {
                                    *tags = tags.saturating_add(1);
                                    if max_tags > 0 && *tags > max_tags {
                                        return Err(());
                                    }
                                    decode_varint(sub, &mut pcur).map_err(|_| ())?;
                                    if let Some(inp) = cur_input.as_deref_mut() {
                                        inp.contents_by_field[field as usize] += 1;
                                    }
                                }
                            }
                            (MsgContext::TensorContents, 6) => {
                                if !sub.len().is_multiple_of(4) {
                                    return Err(());
                                }
                                if let Some(inp) = cur_input.as_deref_mut() {
                                    inp.contents_by_field[6] += sub.len() / 4;
                                }
                            }
                            (MsgContext::TensorContents, 7) => {
                                if !sub.len().is_multiple_of(8) {
                                    return Err(());
                                }
                                if let Some(inp) = cur_input.as_deref_mut() {
                                    inp.contents_by_field[7] += sub.len() / 8;
                                }
                            }
                            (MsgContext::TensorContents, 8) => {
                                if let Some(inp) = cur_input.as_deref_mut() {
                                    inp.contents_by_field[8] += 1;
                                }
                            }
                            _ => {
                                if depth + 1 > max_depth {
                                    return Err(());
                                }
                                parse_submsg(
                                    sub,
                                    MsgContext::GenericSubmsg,
                                    depth + 1,
                                    max_depth,
                                    tags,
                                    max_tags,
                                    model,
                                    None,
                                )?;
                            }
                        }
                    }
                    5 => {
                        let next = cursor.checked_add(4).ok_or(())?;
                        if next > slice.len() {
                            return Err(());
                        }
                        match (ctx, field) {
                            (MsgContext::InputTensor, 1..=5)
                            | (MsgContext::OutputTensor, 1 | 2)
                            | (MsgContext::TensorContents, 1..=5 | 7 | 8) => return Err(()),
                            (MsgContext::TensorContents, 6) => {
                                if let Some(inp) = cur_input.as_deref_mut() {
                                    inp.contents_by_field[6] += 1;
                                }
                            }
                            _ => {}
                        }
                        cursor = next;
                    }
                    _ => return Err(()),
                }
            }
            Ok(())
        }

        let mut model = ReferenceModelInfer::default();
        let mut tags = 0usize;
        parse_submsg(
            slice,
            MsgContext::Root,
            0,
            max_depth,
            &mut tags,
            max_tags,
            &mut model,
            None,
        )?;
        Ok(model)
    }

    fn reference_validate_model_infer(slice: &[u8], config: &ScannerConfig) -> bool {
        let Ok(model) =
            reference_parse_model_infer(slice, config.max_nesting_depth, config.max_tags_scanned)
        else {
            return false;
        };
        if model.has_shm_param {
            return false;
        }
        let mut total_elems: i64 = 0;
        let mut shape_products = Vec::with_capacity(model.inputs.len());
        for inp in &model.inputs {
            if !inp.shape.is_empty() {
                let mut prod: i64 = 1;
                for &d in &inp.shape {
                    if d < 0 {
                        return false;
                    }
                    let Some(next) = prod.checked_mul(d) else {
                        return false;
                    };
                    if next > config.max_tensor_elements {
                        return false;
                    }
                    prod = next;
                }
                total_elems = total_elems.saturating_add(prod);
                if total_elems > config.max_tensor_elements {
                    return false;
                }
                shape_products.push(prod);
            } else {
                shape_products.push(1);
            }
        }
        if config.enforce_payload_parity && (!model.inputs.is_empty() || !model.raw_chunks.is_empty())
        {
            if model.inputs.is_empty() && !model.raw_chunks.is_empty() {
                return false;
            }
            let any_inline = model.inputs.iter().any(|i| i.has_contents_submsg);
            if any_inline && !model.raw_chunks.is_empty() {
                return false;
            }
            if !any_inline && model.raw_chunks.len() != model.inputs.len() {
                return false;
            }
            for (i, inp) in model.inputs.iter().enumerate() {
                let width = kserve_datatype_byte_width(&inp.datatype);
                let is_bytes = inp.datatype.as_slice() == b"BYTES";
                if inp.shape.is_empty() || inp.datatype.is_empty() || (width.is_none() && !is_bytes)
                {
                    return false;
                }
                let expected_elems = usize::try_from(shape_products[i]).unwrap_or(usize::MAX);
                if any_inline {
                    let Some(exp_field) = kserve_datatype_contents_field(&inp.datatype) else {
                        return false;
                    };
                    let has_foreign = (1usize..=8)
                        .any(|f| f != exp_field as usize && inp.contents_by_field[f] > 0);
                    if !inp.has_contents_submsg
                        || has_foreign
                        || inp.contents_by_field[exp_field as usize] != expected_elems
                    {
                        return false;
                    }
                } else if let Some(w) = width {
                    let expected_bytes = shape_products[i]
                        .checked_mul(w)
                        .and_then(|b| usize::try_from(b).ok())
                        .unwrap_or(usize::MAX);
                    if model.raw_chunks[i].len() != expected_bytes {
                        return false;
                    }
                } else if is_bytes {
                    let chunk = &model.raw_chunks[i];
                    if chunk.len() < expected_elems.saturating_mul(4) {
                        return false;
                    }
                    let mut pos = 0usize;
                    for _ in 0..expected_elems {
                        let Some(len_end) = pos.checked_add(4) else {
                            return false;
                        };
                        if len_end > chunk.len() {
                            return false;
                        }
                        let l = u32::from_le_bytes([
                            chunk[pos],
                            chunk[pos + 1],
                            chunk[pos + 2],
                            chunk[pos + 3],
                        ]) as usize;
                        let Some(next) = len_end.checked_add(l) else {
                            return false;
                        };
                        if next > chunk.len() {
                            return false;
                        }
                        pos = next;
                    }
                    if pos != chunk.len() {
                        return false;
                    }
                }
            }
        }
        true
    }

    #[test]
    fn test_protobuf_parser_differential_edge_cases() {
        let config = ScannerConfig::default();

        // 1. Split `shape` across multiple records (unpacked 0x18 + packed 0x1A + unpacked 0x18):
        //    shape = [2] ++ [3, 2] ++ [1] = 12 FP32 elements (48 bytes).
        let mut tensor_split_shape = Vec::new();
        tensor_split_shape.extend_from_slice(b"\x0a\x03IN0\x12\x04FP32");
        tensor_split_shape.extend_from_slice(b"\x18\x02"); // unpacked dim = 2
        tensor_split_shape.extend_from_slice(b"\x1a\x02\x03\x02"); // packed dims = [3, 2]
        tensor_split_shape.extend_from_slice(b"\x18\x01"); // unpacked dim = 1
        let mut msg_split = Vec::new();
        msg_split.extend_from_slice(b"\x0a\x08resnet50");
        msg_split.push(0x2A);
        push_varint(&mut msg_split, tensor_split_shape.len() as u64);
        msg_split.extend_from_slice(&tensor_split_shape);
        msg_split.push(0x3A);
        push_varint(&mut msg_split, 48);
        msg_split.extend_from_slice(&[0u8; 48]);
        assert_eq!(inspect_raw_protobuf(&msg_split, &config), Ok(()));
        assert!(reference_validate_model_infer(&msg_split, &config));

        // 2. Duplicate scalar `datatype` field (last-write-wins):
        //    First declares "FP32" (4 B/elem), then overwrites with "FP64" (8 B/elem) on shape [4].
        //    A 16-byte raw chunk (matching the first "FP32") MUST be rejected because Triton uses "FP64" (32 B).
        let mut tensor_dup_dt = Vec::new();
        tensor_dup_dt.extend_from_slice(b"\x0a\x03IN0\x12\x04FP32\x1a\x01\x04\x12\x04FP64");
        let mut msg_dup_bad = Vec::new();
        msg_dup_bad.extend_from_slice(b"\x0a\x08resnet50\x2a");
        push_varint(&mut msg_dup_bad, tensor_dup_dt.len() as u64);
        msg_dup_bad.extend_from_slice(&tensor_dup_dt);
        msg_dup_bad.extend_from_slice(b"\x3a\x10");
        msg_dup_bad.extend_from_slice(&[0u8; 16]);
        assert_eq!(
            inspect_raw_protobuf(&msg_dup_bad, &config),
            Err(ParseError::PayloadSizeMismatch {
                expected_bytes: 32,
                actual_bytes: 16,
            })
        );
        assert!(!reference_validate_model_infer(&msg_dup_bad, &config));

        //    Conversely, a 32-byte raw chunk (matching the final "FP64") MUST be allowed by both.
        let mut msg_dup_ok = Vec::new();
        msg_dup_ok.extend_from_slice(b"\x0a\x08resnet50\x2a");
        push_varint(&mut msg_dup_ok, tensor_dup_dt.len() as u64);
        msg_dup_ok.extend_from_slice(&tensor_dup_dt);
        msg_dup_ok.extend_from_slice(b"\x3a\x20");
        msg_dup_ok.extend_from_slice(&[0u8; 32]);
        assert_eq!(inspect_raw_protobuf(&msg_dup_ok, &config), Ok(()));
        assert!(reference_validate_model_infer(&msg_dup_ok, &config));

        // 3. Cross-field `InferTensorContents` smuggling:
        //    Declares datatype "FP32" with shape [4], but supplies 2 `fp32_contents` (Field 6) +
        //    2 `int32_contents` (Field 2). Triton only reads `fp32_contents` (2 < 4) -> REJECT!
        let mut bad_contents = Vec::new();
        bad_contents.extend_from_slice(b"\x32\x08\x00\x00\x80\x3f\x00\x00\x00\x40"); // 2 packed fp32
        bad_contents.extend_from_slice(b"\x10\x01\x10\x02"); // 2 unpacked int32 (Field 2)
        let mut tensor_cross_contents = Vec::new();
        tensor_cross_contents.extend_from_slice(b"\x0a\x03IN0\x12\x04FP32\x1a\x01\x04\x2a");
        push_varint(&mut tensor_cross_contents, bad_contents.len() as u64);
        tensor_cross_contents.extend_from_slice(&bad_contents);
        let mut msg_cross = Vec::new();
        msg_cross.extend_from_slice(b"\x0a\x08resnet50\x2a");
        push_varint(&mut msg_cross, tensor_cross_contents.len() as u64);
        msg_cross.extend_from_slice(&tensor_cross_contents);
        assert_eq!(
            inspect_raw_protobuf(&msg_cross, &config),
            Err(ParseError::PayloadSizeMismatch {
                expected_bytes: 16,
                actual_bytes: 8,
            })
        );
        assert!(!reference_validate_model_infer(&msg_cross, &config));

        // 4. Split `InferTensorContents` submessages (Field 5 appearing twice on the same InputTensor):
        //    Protobuf merges multiple occurrences of a singular message field, concatenating repeated
        //    `fp32_contents` (2 floats + 2 floats == 4 floats for shape [4]). Both parsers must agree!
        let mut half_contents = Vec::new();
        half_contents.extend_from_slice(b"\x32\x08\x00\x00\x80\x3f\x00\x00\x00\x40"); // 2 packed fp32
        let mut tensor_merged_contents = Vec::new();
        tensor_merged_contents.extend_from_slice(b"\x0a\x03IN0\x12\x04FP32\x1a\x01\x04");
        for _ in 0..2 {
            tensor_merged_contents.push(0x2A);
            push_varint(&mut tensor_merged_contents, half_contents.len() as u64);
            tensor_merged_contents.extend_from_slice(&half_contents);
        }
        let mut msg_merged = Vec::new();
        msg_merged.extend_from_slice(b"\x0a\x08resnet50\x2a");
        push_varint(&mut msg_merged, tensor_merged_contents.len() as u64);
        msg_merged.extend_from_slice(&tensor_merged_contents);
        assert_eq!(inspect_raw_protobuf(&msg_merged, &config), Ok(()));
        assert!(reference_validate_model_infer(&msg_merged, &config));

        // 5. Mismatched wire type on `InferInputTensor` Field 2 (`datatype` sent as Wire Type 0 varint):
        let bad_wt_dt = b"\x0a\x08resnet50\x2a\x07\x0a\x01x\x10\x01\x18\x01\x3a\x04\x00\x00\x00\x00";
        assert_eq!(
            inspect_raw_protobuf(bad_wt_dt, &config),
            Err(ParseError::InvalidWireType(0))
        );
        assert!(!reference_validate_model_infer(bad_wt_dt, &config));
    }

    #[test]
    fn test_differential_fuzz_custos_vs_reference_protobuf_parser() {
        let config = ScannerConfig::default();
        // Deterministic SplitMix64 PRNG for reproducible fuzzing across 5,000 structured and
        // mutated KServe v2 `ModelInferRequest` wire streams.
        let mut state: u64 = 0xC057_05F0_226E_76E1;
        let mut next_u64 = || -> u64 {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };

        let datatypes: &[&[u8]] = &[
            b"BOOL", b"INT8", b"UINT8", b"INT16", b"UINT16", b"FP16", b"BF16", b"INT32",
            b"UINT32", b"FP32", b"INT64", b"UINT64", b"FP64", b"BYTES", b"FP32 ", b"UNKNOWN",
        ];

        let mut allowed_count = 0usize;
        let mut rejected_count = 0usize;

        for iter in 0..5_000 {
            let mut msg = Vec::new();
            let num_inputs = (next_u64() % 4) as usize;
            let use_inline = (next_u64() % 3) == 0;
            let raws_first = (next_u64() % 2) == 0;

            let mut input_blobs = Vec::new();
            let mut raw_blobs = Vec::new();

            for _ in 0..num_inputs {
                let dt1 = datatypes[(next_u64() as usize) % datatypes.len()];
                let has_second_dt = (next_u64() % 4) == 0;
                let dt2 = datatypes[(next_u64() as usize) % datatypes.len()];
                let final_dt = if has_second_dt { dt2 } else { dt1 };

                let dim_count = ((next_u64() % 4) + 1) as usize;
                let mut dims = Vec::with_capacity(dim_count);
                let mut prod: i64 = 1;
                for _ in 0..dim_count {
                    let d = match next_u64() % 12 {
                        0 => 0i64,
                        1 => -((next_u64() % 8) as i64 + 1),
                        _ => ((next_u64() % 4) + 1) as i64,
                    };
                    dims.push(d);
                    prod = prod.saturating_mul(d.max(0));
                }

                // Build InferInputTensor fields in randomized order, splitting shape across
                // multiple packed and unpacked tags and optionally duplicating datatype.
                let mut tensor_fields: Vec<Vec<u8>> = Vec::new();
                // Name
                tensor_fields.push(b"\x0a\x03in0".to_vec());
                // First datatype
                let mut dt1_f = vec![0x12];
                push_varint(&mut dt1_f, dt1.len() as u64);
                dt1_f.extend_from_slice(dt1);
                tensor_fields.push(dt1_f);

                // Split dims across packed and unpacked records
                let mut idx = 0;
                while idx < dims.len() {
                    if (next_u64() % 2) == 0 {
                        let mut f = vec![0x18];
                        push_varint(&mut f, dims[idx] as u64);
                        tensor_fields.push(f);
                        idx += 1;
                    } else {
                        let take = ((next_u64() % 2) as usize + 1).min(dims.len() - idx);
                        let mut packed = Vec::new();
                        for &d in &dims[idx..idx + take] {
                            push_varint(&mut packed, d as u64);
                        }
                        let mut f = vec![0x1A];
                        push_varint(&mut f, packed.len() as u64);
                        f.extend_from_slice(&packed);
                        tensor_fields.push(f);
                        idx += take;
                    }
                }

                // Deterministic Fisher-Yates shuffle of non-duplicate fields
                for i in (1..tensor_fields.len()).rev() {
                    let j = (next_u64() as usize) % (i + 1);
                    tensor_fields.swap(i, j);
                }

                // Append second datatype after shuffle if testing last-write-wins
                if has_second_dt {
                    let mut dt2_f = vec![0x12];
                    push_varint(&mut dt2_f, dt2.len() as u64);
                    dt2_f.extend_from_slice(dt2);
                    tensor_fields.push(dt2_f);
                }

                if use_inline {
                    let elem_count = if (next_u64() % 5) == 0 {
                        (prod + 1).max(0) as usize
                    } else {
                        prod.max(0) as usize
                    };
                    let c_field = if (next_u64() % 6) == 0 {
                        2u8 // intentional cross-field mismatch
                    } else {
                        kserve_datatype_contents_field(final_dt).unwrap_or(6)
                    };
                    let mut c_buf = Vec::new();
                    match c_field {
                        1..=5 => {
                            if (next_u64() % 2) == 0 {
                                let mut packed = Vec::new();
                                for _ in 0..elem_count {
                                    push_varint(&mut packed, 1);
                                }
                                c_buf.push((c_field << 3) | 2);
                                push_varint(&mut c_buf, packed.len() as u64);
                                c_buf.extend_from_slice(&packed);
                            } else {
                                for _ in 0..elem_count {
                                    c_buf.push(c_field << 3);
                                    c_buf.push(1);
                                }
                            }
                        }
                        6 => {
                            if (next_u64() % 2) == 0 {
                                c_buf.push((6 << 3) | 2);
                                push_varint(&mut c_buf, (elem_count * 4) as u64);
                                c_buf.resize(c_buf.len() + elem_count * 4, 0u8);
                            } else {
                                for _ in 0..elem_count {
                                    c_buf.push((6 << 3) | 5);
                                    c_buf.extend_from_slice(&[0u8; 4]);
                                }
                            }
                        }
                        7 => {
                            c_buf.push((7 << 3) | 2);
                            push_varint(&mut c_buf, (elem_count * 8) as u64);
                            c_buf.resize(c_buf.len() + elem_count * 8, 0u8);
                        }
                        8 => {
                            for _ in 0..elem_count {
                                c_buf.extend_from_slice(b"\x42\x02ab");
                            }
                        }
                        _ => {}
                    }
                    let mut f = vec![0x2A];
                    push_varint(&mut f, c_buf.len() as u64);
                    f.extend_from_slice(&c_buf);
                    tensor_fields.push(f);
                } else {
                    // Build matching or perturbed raw_input_contents chunk
                    let mut raw_chunk = Vec::new();
                    if final_dt == b"BYTES" && dims.iter().all(|&d| d >= 0) {
                        for _ in 0..(prod as usize) {
                            let slen = (next_u64() % 5) as u32;
                            raw_chunk.extend_from_slice(&slen.to_le_bytes());
                            raw_chunk.resize(raw_chunk.len() + slen as usize, b'x');
                        }
                    } else if let Some(w) = kserve_datatype_byte_width(final_dt) {
                        let nbytes = (prod.max(0) as usize).saturating_mul(w as usize);
                        raw_chunk.resize(nbytes, 0xAA);
                    } else {
                        raw_chunk.resize(8, 0xAA);
                    }
                    // Perturb chunk length on 25% of runs
                    if (next_u64() % 4) == 0 {
                        raw_chunk.push(0xFF);
                    }
                    let mut rf = vec![0x3A];
                    push_varint(&mut rf, raw_chunk.len() as u64);
                    rf.extend_from_slice(&raw_chunk);
                    raw_blobs.push(rf);
                }

                let tensor_body: Vec<u8> = tensor_fields.into_iter().flatten().collect();
                let mut inf = vec![0x2A];
                push_varint(&mut inf, tensor_body.len() as u64);
                inf.extend_from_slice(&tensor_body);
                input_blobs.push(inf);
            }

            msg.extend_from_slice(b"\x0a\x08resnet50");
            if raws_first {
                for r in &raw_blobs {
                    msg.extend_from_slice(r);
                }
                for inp in &input_blobs {
                    msg.extend_from_slice(inp);
                }
            } else {
                for inp in &input_blobs {
                    msg.extend_from_slice(inp);
                }
                for r in &raw_blobs {
                    msg.extend_from_slice(r);
                }
            }

            // Additionally test arbitrary byte-mutated wire streams every 10th iteration
            if iter % 10 == 9 && !msg.is_empty() {
                let idx = (next_u64() as usize) % msg.len();
                msg[idx] ^= (next_u64() & 0xFF) as u8;
            }

            let custos_ok = inspect_raw_protobuf(&msg, &config).is_ok();
            let ref_ok = reference_validate_model_infer(&msg, &config);
            assert_eq!(
                custos_ok, ref_ok,
                "differential parser mismatch at iter {iter}: custos={custos_ok}, ref={ref_ok}, bytes={msg:02x?}"
            );
            if custos_ok {
                allowed_count += 1;
            } else {
                rejected_count += 1;
            }
        }

        assert!(allowed_count > 500, "fuzzer must exercise valid payloads");
        assert!(rejected_count > 500, "fuzzer must exercise rejected payloads");
    }
}

use core::fmt;

/// Exhaustive error taxonomy for the zero-copy KServe v2 Protobuf wire scanner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParseError {
    /// Input buffer is empty or shorter than a minimal header.
    EmptyFrame,
    /// gRPC compression flag is non-zero (compressed frames cannot be zero-copy scanned).
    UnsupportedCompression(u8),
    /// The 4-byte big-endian gRPC length prefix does not match the remaining payload bytes.
    FrameLengthMismatch { declared: usize, actual: usize },
    /// Payload ended prematurely while reading a varint or length-delimited field.
    UnexpectedEof,
    /// A varint exceeded 10 bytes, overflowed 64 bits on the 10th byte, or had field number 0.
    MalformedVarint,
    /// Encountered a deprecated or undefined Protobuf wire type (e.g., 3, 4, 6, 7).
    InvalidWireType(u8),
    /// Submessage nesting depth exceeded the configured stack limit.
    NestingLimitExceeded(usize),
    /// A field inside a length-delimited submessage extended past the submessage's declared boundary.
    SubmessageOverrun,
    /// A tensor shape dimension decoded to zero or a negative 64-bit signed integer.
    NegativeDimension(i64),
    /// Checked multiplication of tensor dimensions overflowed a 64-bit signed integer.
    ShapeOverflow,
    /// Total tensor element count exceeded the configured admission ceiling.
    BufferCeilingExceeded { elements: i64, max_elements: i64 },
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyFrame => write!(f, "empty frame"),
            Self::UnsupportedCompression(flag) => {
                write!(f, "unsupported gRPC compression flag: 0x{flag:02x}")
            }
            Self::FrameLengthMismatch { declared, actual } => {
                write!(
                    f,
                    "gRPC frame length mismatch: declared {declared} bytes, actual {actual} bytes"
                )
            }
            Self::UnexpectedEof => write!(f, "unexpected end of payload"),
            Self::MalformedVarint => write!(f, "malformed 64-bit varint"),
            Self::InvalidWireType(wt) => write!(f, "invalid protobuf wire type: {wt}"),
            Self::NestingLimitExceeded(depth) => {
                write!(f, "protobuf nesting limit exceeded at depth {depth}")
            }
            Self::SubmessageOverrun => write!(f, "submessage boundary overrun"),
            Self::NegativeDimension(dim) => {
                write!(f, "invalid non-positive tensor dimension: {dim}")
            }
            Self::ShapeOverflow => write!(f, "64-bit integer overflow in tensor shape product"),
            Self::BufferCeilingExceeded {
                elements,
                max_elements,
            } => {
                write!(
                    f,
                    "tensor element count {elements} exceeds admission ceiling {max_elements}"
                )
            }
        }
    }
}

impl core::error::Error for ParseError {}

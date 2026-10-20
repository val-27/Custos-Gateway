use crate::error::ParseError;

/// Decodes a Base-128 Varint up to 64 bits (maximum 10 bytes) from `slice` starting at `*cursor`.
///
/// Advances `*cursor` past the consumed bytes on success.
/// Enforces the 10th-byte overflow guard (`byte > 0x01` at index 9 is rejected).
#[inline]
pub fn decode_varint(slice: &[u8], cursor: &mut usize) -> Result<u64, ParseError> {
    let mut result: u64 = 0;
    let mut shift: u32 = 0;

    for i in 0..10 {
        if *cursor >= slice.len() {
            return Err(ParseError::UnexpectedEof);
        }

        let byte = slice[*cursor];
        *cursor += 1;

        if i == 9 && byte > 0x01 {
            return Err(ParseError::MalformedVarint);
        }

        result |= u64::from(byte & 0x7F) << shift;

        if (byte & 0x80) == 0 {
            return Ok(result);
        }

        shift += 7;
    }

    Err(ParseError::MalformedVarint)
}

"""Exact block decode for revision-pinned DeepSeek MXFP4/Q8 source import.

MXFP4 stores an unsigned E8M0 scale then 16 bytes of E2M1 codes. The low
nibbles are logical values 0..15 and high nibbles are 16..31. Q8 stores one
little-endian binary16 multiplier followed by 32 signed bytes. No quantization
or activation rounding is performed here.
"""
from __future__ import annotations

import math
import struct

from ..formats import InterleavedBlockFormat, get_format, decode_e2m1_word


def decode_block(payload: bytes, format_name: str) -> tuple[float, ...]:
    spec = get_format(format_name)
    if not isinstance(spec, InterleavedBlockFormat) or len(payload) != spec.block_bytes:
        raise ValueError("block length/format mismatch")
    if spec.group_size==256:
        from .block256 import decode_block256
        return decode_block256(payload,format_name)
    if format_name == "mxfp4_g32_e8m0":
        if payload[0] == 255:
            raise ValueError("MXFP4 E8M0 NaN scale is not a finite weight block")
        scale = math.ldexp(1.0, payload[0] - 127)
        codes = tuple(byte & 15 for byte in payload[1:]) + tuple(byte >> 4 for byte in payload[1:])
        return tuple(decode_e2m1_word(code) * scale for code in codes)
    scale = struct.unpack_from('<e', payload)[0]
    if not math.isfinite(scale):
        raise ValueError("Q8 scale must be finite")
    return tuple(scale * code for code in struct.unpack_from('<32b', payload, 2))

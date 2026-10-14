"""Independent known-word conformance for the DeepSeek persistent formats."""
import math
import struct

import pytest

from tools.artifact.codecs.block_interleaved import decode_block
from tools.artifact.layouts import encoded_size


def test_fp16_preserves_every_word_including_nonfinite_payloads():
    import torch
    from tools.artifact.codecs.direct import encode_direct, decode_direct
    words = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16)
    payload = encode_direct(words.view(torch.float16), "fp16")
    recovered = decode_direct(payload, "fp16", (65536,)).view(torch.int16)
    assert torch.equal(words, recovered)


def test_mxfp4_all_code_words_scale_extremes_and_split_nibbles():
    expected = (0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.)
    # Low half is ascending codes; high half is descending, not adjacent pairs.
    codes = bytes(i | ((15 - i) << 4) for i in range(16))
    for exponent in (0, 1, 126, 127, 128, 254):
        values = decode_block(bytes([exponent]) + codes, "mxfp4_g32_e8m0")
        oracle = tuple(math.ldexp(v, exponent - 127) for v in expected + expected[::-1])
        assert values == oracle
        assert math.copysign(1, values[8]) == -1
        assert math.copysign(1, values[0]) == 1
    with pytest.raises(ValueError, match="NaN scale"):
        decode_block(bytes([255]) + codes, "mxfp4_g32_e8m0")


def test_q8_negative_scale_and_full_signed_byte_domain():
    for base in range(-128, 128, 32):
        codes = tuple(range(base, base + 32))
        for scale in (0., -0., 2**-24, -.5, 65504.):
            block = struct.pack("<e32b", scale, *codes)
            assert decode_block(block, "q8_g32_fp16_block") == tuple(scale * c for c in codes)
    for scale in (math.inf, -math.inf, math.nan):
        with pytest.raises(ValueError, match="finite"):
            decode_block(struct.pack("<e", scale) + bytes(32), "q8_g32_fp16_block")


def test_encoded_geometry_and_rejected_misinterpretations():
    assert encoded_size("block_interleaved_v1", "mxfp4_g32_e8m0", (384*2304, 5120)) == 2406481920
    assert encoded_size("block_interleaved_v1", "q8_g32_fp16_block", (512, 5120)) == 2785280
    assert encoded_size("contiguous_le_v1", "fp16", ()) == 2
    for shape in ((1, 31), (2, 33), (2, 32, 32), (0, 32)):
        with pytest.raises(ValueError):
            encoded_size("block_interleaved_v1", "mxfp4_g32_e8m0", shape)
    with pytest.raises(ValueError):
        encoded_size("block_interleaved_v1", "q8_g32_fp16", (1, 32))
    with pytest.raises(ValueError):
        decode_block(bytes(17), "q8_g32_fp16_block")

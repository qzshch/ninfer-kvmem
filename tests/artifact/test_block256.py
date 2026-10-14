import struct
from tools.artifact.codecs.block_interleaved import decode_block
from tools.artifact.layouts import encoded_size


def test_iq3_s_known_grid_sign_and_subscale_words():
    b=bytearray(110);struct.pack_into('<e',b,0,1)
    b[2]=1;b[74]=1;b[106]=0x21
    y=decode_block(b,'iq3_s_g256_block')
    assert y[:4]==(-9,3,3,3)
    assert y[4:32]==(3,)*28
    assert y[32:64]==(5,)*32
    assert y[64:]==(1,)*192


def test_iq4_xs_nonuniform_lut_nibble_order_and_six_bit_scales():
    b=bytearray(136);struct.pack_into('<eH',b,0,1,0xAAAA)
    b[4:8]=bytes([0x11])*4;b[8:]=bytes([0x87])*128
    y=decode_block(b,'iq4_xs_g256_block')
    assert y==((-10,)*16+(1,)*16)*8


def test_q3_k_negative_subscale_high_mask_and_lane_order():
    b=bytearray(110);struct.pack_into('<e',b,108,1)
    assert decode_block(b,'q3_k_g256_block')==(128,)*256
    b[0]=1;b[32]=3
    y=decode_block(b,'q3_k_g256_block')
    assert y[0]==-96 and y[1]==128 and y[32]==128 and y[128]==128


def test_q4_k_minimum_and_paired_halves():
    b=bytearray(144);struct.pack_into('<ee',b,0,1,.5)
    b[4:12]=bytes([1])*8;b[12:16]=bytes([0x11])*4;b[16:]=bytes([0x21])*128
    assert decode_block(b,'q4_k_g256_block')==((.5,)*32+(1.5,)*32)*4


def test_q6_k_low_high_planes_and_signed_subscales():
    b=bytearray(210);struct.pack_into('<e',b,208,1);b[192:208]=bytes([2])*16
    assert decode_block(b,'q6_k_g256_block')==(-64,)*256
    b[0]=0xA5;b[32]=0x3F;b[128]=0xE4
    y=decode_block(b,'q6_k_g256_block')
    assert (y[0],y[32],y[64],y[96])==(-54,-2,20,38)
    for name,block in [('iq3_s_g256_block',110),('iq4_xs_g256_block',136),('q3_k_g256_block',110),('q4_k_g256_block',144),('q6_k_g256_block',210)]:
        assert encoded_size('block_interleaved_v1',name,(288*2048,4096))==288*2048*4096//256*block

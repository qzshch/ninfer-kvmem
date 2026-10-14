// GGML-compatible block representations, native NInfer pointwise decoder.
// Layouts/grid derive from MIT GGML; see third_party/ggml-blocks/LICENSE.
// This is an operand codec, not an external inference backend.
#pragma once
__device__ __constant__ unsigned int iq3_s_grid[512]={
#include "ggml-blocks/iq3s_grid.inc"
};
__device__ __constant__ signed char iq4_values[16]={
    -127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113
};
__device__ inline float block_half(const unsigned char* p) {
    return decode_half(static_cast<std::uint16_t>(p[0]|(static_cast<unsigned>(p[1])<<8)));
}

template<QType Format>
__device__ float decode_block256(const unsigned char* b,int c) {
    if constexpr(Format==QType::IQ3_S_G256_BLOCK) {
        const int group=c/32,sub=c/4;
        const unsigned grid=b[2+sub]|(((b[66+group]>>(sub%8))&1u)<<8);
        const unsigned magnitude=(iq3_s_grid[grid]>>(8*(c%4)))&255u;
        const int sign=(b[74+c/8]>>(c%8))&1u?-1:1;
        const unsigned scale=1+2*((b[106+group/2]>>(4*(group%2)))&15u);
        return block_half(b)*static_cast<float>(scale)*static_cast<float>(sign*static_cast<int>(magnitude));
    }
    if constexpr(Format==QType::IQ4_XS_G256_BLOCK) {
        const int group=c/32;
        const unsigned high=b[2]|(static_cast<unsigned>(b[3])<<8);
        const int scale=static_cast<int>(((b[4+group/2]>>(4*(group%2)))&15u)|(((high>>(2*group))&3u)<<4))-32;
        const unsigned code=(b[8+group*16+c%16]>>(c%32<16?0:4))&15u;
        return block_half(b)*static_cast<float>(scale)*iq4_values[code];
    }
    if constexpr(Format==QType::Q3_K_G256_BLOCK) {
        const int group=c/16;
        const int scale=static_cast<int>(((b[96+group%8]>>(4*(group/8)))&15u)|(((b[104+group%4]>>(2*(group/4)))&3u)<<4))-32;
        const int low=(b[32+(c/128)*32+c%32]>>(2*((c%128)/32)))&3u;
        const int code=low-((b[c%32]>>(c/32))&1u?0:4);
        return block_half(b+108)*static_cast<float>(scale)*code;
    }
    if constexpr(Format==QType::Q4_K_G256_BLOCK) {
        const int group=c/32;
        const auto* s=b+4;
        const unsigned scale=group<4?s[group]&63u:(s[group+4]&15u)|((s[group-4]>>6)<<4);
        const unsigned minimum=group<4?s[group+4]&63u:(s[group+4]>>4)|((s[group]>>6)<<4);
        const unsigned code=(b[16+(c/64)*32+c%32]>>(c%64<32?0:4))&15u;
        return block_half(b)*scale*code-block_half(b+2)*minimum;
    }
    if constexpr(Format==QType::Q6_K_G256_BLOCK) {
        const int chunk=c/128,quad=(c%128)/32,lane=c%32;
        const unsigned low=(b[chunk*64+(quad%2)*32+lane]>>(quad<2?0:4))&15u;
        const unsigned high=(b[128+chunk*32+lane]>>(2*quad))&3u;
        const int code=static_cast<int>(low|(high<<4))-32;
        const int scale=static_cast<const signed char*>(static_cast<const void*>(b+192))[c/16];
        return block_half(b+208)*scale*code;
    }
    return 0.f;
}

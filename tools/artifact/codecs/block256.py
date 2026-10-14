"""Scalar reconstruction of the registered 256-value GGML-compatible blocks.

Block contracts and immutable IQ3_S grid originate in MIT GGML; attribution
and the pinned source are retained under third_party/ggml-blocks. These routines
do not call a GPU kernel or inference engine. Results are exact Python binary64
reconstructions from binary16 scales and integer codes.
"""
import math
from pathlib import Path
import re
import struct

from ..formats import BLOCK256_FORMATS

_GRID=tuple(int(x,16) for x in re.findall(r'0x[0-9a-fA-F]+',
    (Path(__file__).resolve().parents[3]/'third_party/ggml-blocks/iq3s_grid.inc').read_text()))
if len(_GRID)!=512:raise ValueError('invalid immutable IQ3_S grid')
_IQ4=(-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113)


def decode_block256(data: bytes, name: str) -> tuple[float,...]:
    spec=BLOCK256_FORMATS[name]
    if len(data)!=spec.block_bytes:raise ValueError('encoded block length mismatch')
    scale=struct.unpack_from('<e',data,spec.base_scale_offset)[0]
    if not math.isfinite(scale):raise ValueError('block multiplier must be finite')
    output=[]
    if name=='iq3_s_g256_block':
        for group in range(8):
            multiplier=scale*(1+2*((data[106+group//2]>>(4*(group%2)))&15))
            for octet in range(4):
                signs=data[74+4*group+octet]
                for half in range(2):
                    i=group*8+octet*2+half
                    index=data[2+i]+256*((data[66+group]>>(2*octet+half))&1)
                    grid=struct.pack('<I',_GRID[index])
                    output.extend(multiplier*v*(-1 if signs&(1<<(4*half+j)) else 1) for j,v in enumerate(grid))
    elif name=='iq4_xs_g256_block':
        high=struct.unpack_from('<H',data,2)[0]
        for group in range(8):
            s=((data[4+group//2]>>(4*(group%2)))&15)+16*((high>>(2*group))&3)-32
            codes=data[8+16*group:24+16*group]
            output.extend(scale*s*_IQ4[q&15] for q in codes)
            output.extend(scale*s*_IQ4[q>>4] for q in codes)
    elif name=='q3_k_g256_block':
        packed=data[96:108]
        scales=[]
        for section in range(4):
            for lane in range(4):
                low=(packed[lane+4*(section%2)]>>(4*(section//2)))&15
                high=(packed[8+lane]>>(2*section))&3
                scales.append(low+16*high-32)
        for chunk in range(2):
            for shift in range(4):
                group=chunk*8+shift*2
                for half in range(2):
                    for lane in range(16):
                        i=lane+16*half
                        code=(data[32+32*chunk+i]>>(2*shift))&3
                        if not data[i]&(1<<(4*chunk+shift)):code-=4
                        output.append(scale*scales[group+half]*code)
    elif name=='q4_k_g256_block':
        minimum=struct.unpack_from('<e',data,2)[0]
        if not math.isfinite(minimum):raise ValueError('block minimum multiplier must be finite')
        scales=data[4:16]
        for pair in range(4):
            codes=data[16+32*pair:48+32*pair]
            for half in range(2):
                group=pair*2+half
                if group<4:s,m=scales[group]&63,scales[group+4]&63
                else:s,m=(scales[group+4]&15)+16*(scales[group-4]>>6),(scales[group+4]>>4)+16*(scales[group]>>6)
                output.extend(scale*s*((q>>(4*half))&15)-minimum*m for q in codes)
    elif name=='q6_k_g256_block':
        scales=struct.unpack_from('<16b',data,192)
        for chunk in range(2):
            rows=[[] for _ in range(4)]
            for lane in range(32):
                low0,low1=data[64*chunk+lane],data[64*chunk+32+lane]
                high=data[128+32*chunk+lane]
                codes=((low0&15)+16*(high&3)-32,(low1&15)+16*((high>>2)&3)-32,
                       (low0>>4)+16*((high>>4)&3)-32,(low1>>4)+16*((high>>6)&3)-32)
                for row,code in enumerate(codes):rows[row].append(scale*scales[chunk*8+row*2+lane//16]*code)
            for row in rows:output.extend(row)
    return tuple(output)

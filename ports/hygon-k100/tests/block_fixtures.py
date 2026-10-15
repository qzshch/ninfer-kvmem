"""Bounded scalar CPU decode fixtures; never calls the device decoder."""
import json
from pathlib import Path
import random
import struct
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from tools.artifact.codecs.block256 import decode_block256
from tools.artifact.formats import BLOCK256_FORMATS

root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=True)
cases=[];rng=random.Random(73531)
problems=[(name,19,4096,1,17) for name in BLOCK256_FORMATS]
problems += [('iq3_s_g256_block',2048,4096,0,2048),('iq4_xs_g256_block',4096,2048,0,4096)]
problems += [(name,1030,1024,1,1025) for name in BLOCK256_FORMATS]
problems += [('q8_g32_fp16_block',1030,1024,1,1025)]
for index,(name,parent_rows,k,row_begin,n) in enumerate(problems):
    q8=name=='q8_g32_fp16_block'
    block_values=32 if q8 else 256
    block_bytes=34 if q8 else BLOCK256_FORMATS[name].block_bytes
    scale_offset=0 if q8 else BLOCK256_FORMATS[name].base_scale_offset
    with (root/f'{index}.encoded').open('wb') as encoded,(root/f'{index}.fp64').open('wb') as oracle:
        for _ in range(parent_rows*k//block_values):
            block=bytearray(rng.getrandbits(8) for _ in range(block_bytes))
            # Finite normal/subnormal/negative multipliers with arbitrary code,
            # sign and subscale bits; oracle evaluates every logical element.
            word=rng.choice((0x0001,0x2000,0x3007,0xB055,0x3801))
            struct.pack_into('<H',block,scale_offset,word)
            if name=='q4_k_g256_block':struct.pack_into('<H',block,2,0x2803)
            encoded.write(block)
            if q8:
                scale=struct.unpack_from('<e',block)[0]
                values=[scale*x for x in struct.unpack_from('<32b',block,2)]
            else:values=decode_block256(block,name)
            oracle.write(struct.pack(f'<{block_values}d',*values))
    cases.append(dict(format=name,parent_rows=parent_rows,k=k,row_begin=row_begin,n=n,
                      encoded=f'{index}.encoded',oracle=f'{index}.fp64'))
(root/'manifest.json').write_text(json.dumps(cases,indent=2))
(root/'stamp').write_text('complete\n')
print('Generated',len(cases),'independent CPU reconstruction fixtures')

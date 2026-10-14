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
for index,(name,parent_rows,k,row_begin,n) in enumerate(problems):
    spec=BLOCK256_FORMATS[name]
    with (root/f'{index}.encoded').open('wb') as encoded,(root/f'{index}.fp64').open('wb') as oracle:
        for _ in range(parent_rows*k//256):
            block=bytearray(rng.getrandbits(8) for _ in range(spec.block_bytes))
            # Finite normal/subnormal/negative multipliers with arbitrary code,
            # sign and subscale bits; oracle evaluates every logical element.
            word=rng.choice((0x0001,0x2000,0x3007,0xB055,0x3801))
            struct.pack_into('<H',block,spec.base_scale_offset,word)
            if name=='q4_k_g256_block':struct.pack_into('<H',block,2,0x2803)
            encoded.write(block)
            oracle.write(struct.pack('<256d',*decode_block256(block,name)))
    cases.append(dict(format=name,parent_rows=parent_rows,k=k,row_begin=row_begin,n=n,
                      encoded=f'{index}.encoded',oracle=f'{index}.fp64'))
(root/'manifest.json').write_text(json.dumps(cases,indent=2))
(root/'stamp').write_text('complete\n')
print('Generated',len(cases),'independent CPU reconstruction fixtures')

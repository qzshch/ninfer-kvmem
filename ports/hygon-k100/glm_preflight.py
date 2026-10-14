"""Offline GLM-5.3-Flash header capacity check; performs no downloads/inference."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import struct
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.convert.sources.gguf import read_header

GIB=2**30


def plan(manifest: dict, headers_dir: Path, draft_header: Path,
         draft_file_bytes: int, free_device_bytes: list[int], *, context: int=32768) -> dict:
    if len(free_device_bytes)!=3 or not 1<=context<=32768:
        raise ValueError('preflight supports three cards and at most 32768 context tokens')
    tensors={};metadata=None;ordinals=[]
    for file in manifest['files']:
        h=read_header(headers_dir/Path(file['path']).name,expected_source_bytes=file['size'])
        ordinal=h.metadata.get('split.no',0);ordinals.append(ordinal)
        if ordinal==0:metadata=h.metadata
        for t in h.tensors:
            if t.name in tensors:raise ValueError('duplicate source tensor')
            tensors[t.name]=t
    if sorted(ordinals)!=list(range(len(manifest['files']))) or metadata is None:
        raise ValueError('missing/duplicate source split')
    if metadata.get('general.architecture')!='glm5-next' or metadata.get('glm5-next.block_count')!=46:
        raise ValueError('expected 45 GLM target blocks plus one stock MTP block')
    if metadata.get('split.tensors.count')!=len(tensors):
        raise ValueError('source tensor coverage mismatch')
    layers=Counter();global_bytes=0;types=Counter()
    for name,t in tensors.items():
        types[t.ggml_type]+=1
        if name.startswith('blk.'):
            layer=int(name.split('.')[1])
            if layer not in range(46):raise ValueError('unexpected source block')
            layers[layer]+=t.bytes
        else:global_bytes+=t.bytes
    data=draft_header.read_bytes()
    if len(data)<8:raise ValueError('truncated draft header')
    size=struct.unpack_from('<Q',data)[0]
    directory=json.loads(data[8:8+size])
    ranges=[]
    for name,t in directory.items():
        if name=='__metadata__':continue
        if t['dtype']!='BF16':raise ValueError('expected unchanged BF16 DFlash2 weights')
        begin,end=t['data_offsets'];elements=1
        for extent in t['shape']:elements*=extent
        if begin<0 or end-begin!=2*elements or 8+size+end>draft_file_bytes:
            raise ValueError('invalid draft tensor extent')
        ranges.append((begin,end))
    cursor=0
    for begin,end in sorted(ranges):
        if begin!=cursor:raise ValueError('missing or duplicate draft payload coverage')
        cursor=end
    if 8+size+cursor!=draft_file_bytes:raise ValueError('draft source length mismatch')
    # All stored target tensors count, including stock MTP even though DFlash2
    # is selected. This is an upper bound, not a hot-expert/offload policy.
    candidates=[]
    for a in range(1,44):
        for b in range(a+1,45):
            weights=[global_bytes+layers[45]+sum(layers[i] for i in range(a)),
                     sum(layers[i] for i in range(a,b)),
                     sum(layers[i] for i in range(b,45))+draft_file_bytes]
            candidates.append((max(weights),a,b,weights))
    _,a,b,weights=min(candidates)
    reserve={'compressed_kv_and_index':2*GIB,'kda_state_and_k7_snapshots':GIB,
             'draft_cache_and_target_features':GIB,'workspace':2*GIB,
             'transport_graph_and_other_persistent':2*GIB}
    devices=[]
    for i,w in enumerate(weights):
        weight_bound=(w+255)//256*256+len(tensors)*256
        required=weight_bound+sum(reserve.values())
        devices.append(dict(device=i,encoded_weight_bytes=w,aligned_weight_bound_bytes=weight_bound,
                            runtime_reserve_bytes=sum(reserve.values()),required_bytes=required,
                            free_device_bytes=free_device_bytes[i],remaining_bytes=free_device_bytes[i]-required))
    return dict(status='budget_fits' if all(d['remaining_bytes']>=0 for d in devices) else 'budget_exceeds_device',
                model=manifest['repo'],revision=manifest['revision'],quantization=manifest['quantization'],
                source_tensor_count=len(tensors),source_wire_types=dict(types),draft_tensor_count=len(ranges),
                stock_mtp_included_in_weight_bound=True,stage_layer_ranges=[[0,a],[a,b],[b,45]],
                global_weights_device=0,draft_device=2,max_context=context,max_concurrency=1,dflash2_k=7,
                runtime_reserve_per_card=reserve,devices=devices,weight_or_kv_offload_allowed=False,
                payload_hashes_verified=False,actual_model_peak_measured=False,full_model_inference=False,
                implementation_condition='keep encoded weights and compressed MLA caches; Program allocations must remain inside these per-card ceilings')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--headers-dir',type=Path,required=True)
    p.add_argument('--draft-header',type=Path,required=True)
    p.add_argument('--draft-file-bytes',type=int,required=True)
    p.add_argument('--free-device-bytes',type=int,nargs=3,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    report=plan(json.loads(args.manifest.read_text()),args.headers_dir,args.draft_header,
                args.draft_file_bytes,args.free_device_bytes)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(report['status'],[(d['device'],round(d['required_bytes']/GIB,3),round(d['remaining_bytes']/GIB,3)) for d in report['devices']])
    return 0 if report['status']=='budget_fits' else 1


if __name__=='__main__':raise SystemExit(main())

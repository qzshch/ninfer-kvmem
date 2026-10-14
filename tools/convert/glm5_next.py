"""Offline GLM-5.3-Flash GGUF + original BF16 DFlash2 -> native v3 artifact.

All encoded weight bytes are preserved. Quantized expert/head groups are
reshaped to rank-two parents without transposing or requantizing. Model roles
are explicit; the runtime never depends on GGUF tensor names or a GGUF engine.
"""
from __future__ import annotations

import argparse
import json
from math import prod
from pathlib import Path
import re
import struct

from tools.artifact.layouts import encoded_size
from tools.artifact.schema import TensorSpec, ResourceSpec
from tools.convert.sources.gguf import GgufSource

FORMATS={0:('fp32','contiguous_le_v1'),1:('fp16','contiguous_le_v1'),
         8:('q8_g32_fp16_block','block_interleaved_v1'),30:('bf16','contiguous_le_v1'),
         11:('q3_k_g256_block','block_interleaved_v1'),12:('q4_k_g256_block','block_interleaved_v1'),
         14:('q6_k_g256_block','block_interleaved_v1'),21:('iq3_s_g256_block','block_interleaved_v1'),
         23:('iq4_xs_g256_block','block_interleaved_v1')}
ROLES={
    'attn_norm.weight':'attention_norm','ffn_norm.weight':'ffn_norm',
    'attn_q.weight':'kda/query','attn_k.weight':'kda/key','attn_v.weight':'kda/value',
    'ssm_conv1d_q.weight':'kda/query_conv','ssm_conv1d_k.weight':'kda/key_conv','ssm_conv1d_v.weight':'kda/value_conv',
    'ssm_a':'kda/negative_decay_rate','ssm_dt.bias':'kda/decay_bias','ssm_beta.weight':'kda/input_gate',
    'ssm_f_a.weight':'kda/forget_in','ssm_f_b.weight':'kda/forget_out',
    'ssm_g_a.weight':'kda/output_gate_in','ssm_g_b.weight':'kda/output_gate_out','ssm_norm.weight':'kda/output_norm',
    'attn_q_a.weight':'mla/query_lowrank','attn_q_a_norm.weight':'mla/query_norm','attn_q_b.weight':'mla/query_heads',
    'attn_kv_a_mqa.weight':'mla/kv_latent','attn_kv_a_norm.weight':'mla/kv_norm',
    'attn_k_b.weight':'mla/key_absorb','attn_v_b.weight':'mla/value_expand',
    'indexer.attn_k.weight':'mla/index/key','indexer.attn_q_b.weight':'mla/index/query',
    'indexer.k_norm.weight':'mla/index/key_norm','indexer.k_norm.bias':'mla/index/key_norm_bias',
    'indexer.proj.weight':'mla/index/head_weights','indexer_compressor_ape.weight':'mla/index/position_bias',
    'indexer_compressor_gate.weight':'mla/index/pool_gate',
    'ffn_gate.weight':'ffn/dense/gate','ffn_up.weight':'ffn/dense/up','ffn_down.weight':'ffn/dense/down',
    'ffn_gate_inp.weight':'ffn/router','exp_probs_b.bias':'ffn/router_bias',
    **{f'ffn_{kind}_exps.weight':f'ffn/experts/{kind}' for kind in ('gate','up','down')},
    **{f'ffn_{kind}_shexp.weight':f'ffn/shared/{kind}' for kind in ('gate','up','down')},
    **{f'hc_{part}_{suffix}.weight':f'{role}_hc/{field}'
       for part,role in [('attn','attention'),('ffn','ffn')]
       for suffix,field in [('base','base'),('fn','projection'),('scale','scale')]},
    'nextn.eh_proj.weight':'fusion','nextn.enorm.weight':'embedding_norm',
    'nextn.hnorm.weight':'hidden_norm','nextn.shared_head_norm.weight':'output_norm',
}
GLOBALS={'token_embd.weight':'text/token_embedding','output.weight':'text/output_head','output_norm.weight':'text/output_norm'}
DRAFT_GLOBALS={'fc.weight':'feature_projection','hidden_norm.weight':'context_norm','norm.weight':'final_norm',
               'candidate_selector.hidden_projection.weight':'selector/hidden_projection',
               'candidate_selector.predecessor_codebook':'selector/predecessor_codebook',
               'candidate_selector.successor_codebook':'selector/successor_codebook'}
DRAFT_ROLES={'input_layernorm.weight':'attention_norm','post_attention_layernorm.weight':'ffn_norm',
    **{f'self_attn.{src}.weight':f'attention/{dst}' for src,dst in
       [('q_proj','query'),('k_proj','key'),('v_proj','value'),('o_proj','output'),('q_norm','query_norm'),('k_norm','key_norm')]},
    **{f'mlp.{kind}_proj.weight':f'ffn/{kind}' for kind in ('gate','up','down')},
    **{f'{part}.{field}'+('.weight' if field=='kernel_projection' else ''):f'{part}/{field}'
       for part in ('attention_conv','mlp_conv') for field in ('base_kernel','kernel_projection')},
}


def target_role(name: str, layer_types: list[str]) -> str:
    if name in GLOBALS:return GLOBALS[name]
    match=re.fullmatch(r'blk\.(\d+)\.(.+)',name)
    if not match:raise ValueError(f'unknown GLM source parameter {name}')
    layer=int(match[1]);role=match[2]
    if layer>len(layer_types):raise ValueError('source block exceeds target and stock MTP')
    prefix=f'text/layers/{layer}/' if layer<len(layer_types) else 'mtp/'
    if role=='attn_output.weight':
        return prefix+('kda/output' if layer<len(layer_types) and layer_types[layer]=='linear_attention' else 'mla/output')
    try:return prefix+ROLES[role]
    except KeyError:raise ValueError(f'unknown GLM source role {role}') from None


def draft_role(name: str) -> str:
    if name in DRAFT_GLOBALS:return 'dflash2/'+DRAFT_GLOBALS[name]
    match=re.fullmatch(r'layers\.(\d+)\.(.+)',name)
    if not match or not 0<=int(match[1])<5 or match[2] not in DRAFT_ROLES:
        raise ValueError(f'unknown GLM DFlash2 parameter {name}')
    return f'dflash2/layers/{int(match[1])}/'+DRAFT_ROLES[match[2]]


def parameter_shapes(config: dict) -> dict[str,tuple[int,...]]:
    """Finite GLM mathematics, including the preserved optional stock MTP."""
    h=config['hidden_size'];v=config['vocab_size'];e=config['n_routed_experts']
    i=config['moe_intermediate_size'];dense=config['intermediate_size']
    q=config['q_lora_rank'];kv=config['kv_lora_rank'];heads=config['num_attention_heads']
    d=config['qk_nope_head_dim'];vd=config['v_head_dim'];ix=config['index_head_dim']
    kd=config['linear_attn_config']['head_dim'];kh=config['linear_attn_config']['num_heads'];c=kd*kh
    common={'attention_norm':(h,),'ffn_norm':(h,)}
    hc={f'{part}_hc/{kind}':shape for part in ('attention','ffn')
        for kind,shape in [('base',(24,)),('scale',(3,)),('projection',(24,4*h))]}
    kda={**{f'kda/{part}':(c,h) for part in ('query','key','value')},
         **{f'kda/{part}_conv':(c,1,4) for part in ('query','key','value')},
         'kda/negative_decay_rate':(kh,),'kda/decay_bias':(c,),'kda/input_gate':(kh,h),
         **{f'kda/{part}_in':(kd,h) for part in ('forget','output_gate')},
         **{f'kda/{part}_out':(c,kd) for part in ('forget','output_gate')},
         'kda/output_norm':(kd,),'kda/output':(h,c)}
    mla={'mla/query_lowrank':(q,h),'mla/query_norm':(q,),'mla/query_heads':(heads*d,q),
         'mla/kv_latent':(kv,h),'mla/kv_norm':(kv,),'mla/key_absorb':(heads*kv,d),
         'mla/value_expand':(heads*vd,kv),'mla/output':(h,heads*vd),
         'mla/index/query':(config['index_n_heads']*ix,q),'mla/index/key':(ix,h),
         'mla/index/key_norm':(ix,),'mla/index/key_norm_bias':(ix,),
         'mla/index/head_weights':(config['index_n_heads'],h),
         'mla/index/position_bias':(config['index_kpool'],ix),'mla/index/pool_gate':(ix,h)}
    moe={'ffn/router':(e,h),'ffn/router_bias':(e,),
         **{f'ffn/experts/{p}':(e*i,h) for p in ('gate','up')},'ffn/experts/down':(e*h,i),
         **{f'ffn/shared/{p}':(i,h) for p in ('gate','up')},'ffn/shared/down':(h,i)}
    dense_ffn={**{f'ffn/dense/{p}':(dense,h) for p in ('gate','up')},'ffn/dense/down':(h,dense)}
    out={'text/token_embedding':(v,h),'text/output_head':(v,h),'text/output_norm':(h,)}
    for layer,kind in enumerate(config['layer_types']):
        params={**common,**hc,**(kda if kind=='linear_attention' else mla),
                **(dense_ffn if layer<config['first_k_dense_replace'] else moe)}
        out.update({f'text/layers/{layer}/'+k:s for k,s in params.items()})
    out.update({'mtp/'+k:s for k,s in {**common,**mla,**moe,'fusion':(h,2*h),
               'embedding_norm':(h,),'hidden_norm':(h,),'output_norm':(h,)}.items()})
    return out


def draft_parameter_shapes(config: dict) -> dict[str,tuple[int,...]]:
    h=config['hidden_size'];q=config['num_attention_heads']*config['head_dim']
    kv=config['num_key_value_heads']*config['head_dim'];d=config['head_dim'];i=config['intermediate_size']
    dc=config['dflash_config'];groups=h//dc['conv_group_size'];kernel=dc['conv_kernel_size'];rank=dc['selector_rank']
    layer={'attention_norm':(h,),'ffn_norm':(h,),
           'attention/query':(q,h),'attention/key':(kv,h),'attention/value':(kv,h),'attention/output':(h,q),
           'attention/query_norm':(d,),'attention/key_norm':(d,),
           'ffn/gate':(i,h),'ffn/up':(i,h),'ffn/down':(h,i),
           **{f'{part}/base_kernel':(2,kernel,h) for part in ('attention_conv','mlp_conv')},
           **{f'{part}/kernel_projection':(2*kernel*groups,h) for part in ('attention_conv','mlp_conv')}}
    out={'dflash2/'+k:s for k,s in {'feature_projection':(h,h*len(dc['target_layer_ids'])),
         'context_norm':(h,),'final_norm':(h,),'selector/hidden_projection':(rank,h),
         'selector/predecessor_codebook':(config['vocab_size'],rank),
         'selector/successor_codebook':(config['vocab_size'],rank)}.items()}
    for index in range(config['num_hidden_layers']):out.update({f'dflash2/layers/{index}/'+k:s for k,s in layer.items()})
    return out


def draft_directory(path: Path) -> tuple[dict,int]:
    with path.open('rb') as stream:
        length=struct.unpack('<Q',stream.read(8))[0]
        if length>2**20:raise ValueError('unsupported draft header extent')
        header=json.loads(stream.read(length))
    end=0
    for item in sorted((v for k,v in header.items() if k!='__metadata__'),key=lambda x:x['data_offsets'][0]):
        begin,stop=item['data_offsets']
        if item['dtype']!='BF16' or begin!=end or stop-begin!=2*prod(item['shape']):
            raise ValueError('draft must preserve complete original BF16 tensors')
        end=stop
    if 8+length+end!=path.stat().st_size:raise ValueError('truncated/extended draft source')
    return header,8+length


def chunks(path: Path,begin: int,count: int):
    with path.open('rb') as stream:
        stream.seek(begin)
        while count:
            block=stream.read(min(count,8*2**20))
            if not block:raise ValueError('truncated source payload during conversion')
            yield block;count-=len(block)


def config_contract(target: dict,draft: dict,metadata: dict) -> dict:
    text=target['text_config']
    expected={'hidden_size':4096,'num_hidden_layers':45,'vocab_size':154880,'n_routed_experts':288,'hc_mult':4}
    if target.get('architectures')!=['Glm5NextForConditionalGeneration'] or any(text.get(k)!=v for k,v in expected.items()):
        raise ValueError('target config is not the selected GLM-5.3-Flash architecture')
    if metadata.get('general.architecture')!='glm5-next' or metadata.get('glm5-next.block_count')!=46:
        raise ValueError('source GGUF architecture/layer count mismatch')
    for key,value in [('embedding_length',4096),('vocab_size',154880),('expert_count',288)]:
        if metadata.get('glm5-next.'+key)!=value:raise ValueError('source config mismatch: '+key)
    dc=draft.get('dflash_config',{})
    if (draft.get('architectures')!=['DFlash2DraftModel'] or dc.get('block_size')!=8 or
        dc.get('target_layer_ids')!=[5,14,24,33,42] or draft.get('hidden_size')!=4096 or
        draft.get('vocab_size')!=154880 or draft.get('num_hidden_layers')!=5):
        raise ValueError('draft is not the GLM-specific DFlash2 K7 companion')
    return {**text,'architectures':['Glm5NextForConditionalGeneration']}


def convert(source: GgufSource,draft: Path,target_config: dict,draft_config: dict,output: Path,
            provenance: dict,tokenizer: dict) -> None:
    from tools.artifact.writer import ArtifactWriter
    config=config_contract(target_config,draft_config,source.metadata)
    header,draft_start=draft_directory(draft)
    specs=[];bindings={};payloads=[]
    def add(role,shape,format,layout,file,offset,bytes_):
        if role in bindings:raise ValueError('duplicate logical GLM parameter '+role)
        if encoded_size(layout,format,shape)!=bytes_:raise ValueError('source/native geometry mismatch '+role)
        specs.append(TensorSpec(role,tuple(shape),format,layout));bindings[role]={'object':role}
        payloads.append((role,file,offset,bytes_))
    for name,t in source.tensors.items():
        if t.ggml_type not in FORMATS:raise ValueError(f'no lossless native source import for GGML type {t.ggml_type}')
        format,layout=FORMATS[t.ggml_type];shape=t.shape
        if layout=='block_interleaved_v1' and len(shape)>2:shape=(prod(shape[:-1]),shape[-1])
        add(target_role(name,config['layer_types']),shape,format,layout,t.file,t.offset,t.bytes)
    for name,t in header.items():
        if name=='__metadata__':continue
        begin,end=t['data_offsets']
        add(draft_role(name),t['shape'],'bf16','contiguous_le_v1',draft,draft_start+begin,end-begin)
    expected={**parameter_shapes(config),**draft_parameter_shapes(draft_config)}
    actual={s.id:s.shape for s in specs}
    if actual!=expected:
        wrong=sorted(k for k in actual.keys()&expected.keys() if actual[k]!=expected[k])
        raise ValueError(f'GLM parameter coverage/shape mismatch: missing={sorted(expected.keys()-actual.keys())}, '
                         f'unexpected={sorted(actual.keys()-expected.keys())}, wrong_shapes={wrong}')
    metadata={k:v for k,v in source.metadata.items() if k.startswith('tokenizer.')}
    if tokenizer.get('normalizer') is not None or tokenizer['model']['type']!='BPE':
        raise ValueError('GLM source requires its unnormalized byte BPE tokenizer')
    for word,index in tokenizer['model']['vocab'].items():
        if metadata['tokenizer.ggml.tokens'][index]!=word:raise ValueError('GGUF/official tokenizer vocabulary mismatch')
    for added in tokenizer['added_tokens']:
        if metadata['tokenizer.ggml.tokens'][added['id']]!=added['content'] or any(added[k] for k in ('single_word','lstrip','rstrip','normalized')):
            raise ValueError('unsupported or mismatched GLM added-token semantics')
    # GGUF CONTROL classification alone loses HF special=False for code/box/media
    # delimiters. Preserve the authoritative flags instead of hardcoding names.
    metadata['tokenizer.native.added_tokens']=tokenizer['added_tokens']
    tokenizer_bytes=json.dumps(metadata,ensure_ascii=False).encode()
    specs.append(ResourceSpec('text/tokenizer',len(tokenizer_bytes)))
    components={'text':{'config':config,'resources':{'gguf_tokenizer':'text/tokenizer'}},
                'dflash2':{'config':draft_config,'target':'text'},
                'mtp':{'config':{'num_hidden_layers':1,'source_layer':45},'target':'text'}}
    with ArtifactWriter(output,specs,components=components,bindings=bindings,
                        metadata={'name':'GLM-5.3-Flash-UD-IQ4_XS-DFlash2'},
                        provenance={**provenance,'method':'preserve_encoded_source_bytes','weight_requantization':False}) as writer:
        writer.write_object('text/tokenizer',tokenizer_bytes)
        for index,(role,file,offset,bytes_) in enumerate(payloads):
            writer.write_object(role,chunks(file,offset,bytes_))
            print(json.dumps({'object':index+1,'objects':len(payloads),'parameter':role,'bytes':bytes_}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-manifest',type=Path,required=True)
    p.add_argument('--input-dir',type=Path,required=True)
    p.add_argument('--draft',type=Path,required=True)
    p.add_argument('--target-config',type=Path,required=True)
    p.add_argument('--draft-config',type=Path,required=True)
    p.add_argument('--tokenizer-json',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();manifest=json.loads(args.source_manifest.read_text())
    files=[args.input_dir/Path(f['path']).name for f in manifest['files']]
    for file,item in zip(files,manifest['files']):
        if file.stat().st_size!=item['size']:raise ValueError('source file length differs from pinned manifest')
    source=GgufSource(files)
    convert(source,args.draft,json.loads(args.target_config.read_text()),json.loads(args.draft_config.read_text()),
            args.output,{'source_repository':manifest['repo'],'source_revision':manifest['revision'],
                         'source_files':manifest['files'],'draft_repository':'incoai/GLM-5.3-Flash-DFlash2',
                         'draft_revision':'bf582e4eacc1810f76656d1811693ff6c6737d2a'},
            json.loads(args.tokenizer_json.read_text()))


if __name__=='__main__':main()

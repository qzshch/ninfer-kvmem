"""GGUF framing and bounded source behavior, independent hand-authored fixtures."""
import struct

import pytest

from tools.convert.sources.gguf import GgufSource, read_header


def string(value):
    data=value.encode();return struct.pack('<Q',len(data))+data


def fixture(path, *, declared_offset=0, payload=bytes(range(34)), shape=(32,), metadata=()):
    raw=b'GGUF'+struct.pack('<IQQ',3,1,len(metadata))
    for key,kind,value in metadata:
        raw+=string(key)+struct.pack('<I',kind)+value
    raw+=string('matrix')+struct.pack('<I',len(shape))+struct.pack('<'+'Q'*len(shape),*shape)
    raw+=struct.pack('<IQ',8,declared_offset)
    raw+=bytes((-len(raw))%32)
    path.write_bytes(raw+payload)
    return len(raw)


def test_exact_matrix_dimensions_bounded_reads_and_array_metadata(tmp_path):
    p=tmp_path/'test.gguf'
    values=struct.pack('<IQiii',5,3,5,14,24)
    offset=fixture(p,shape=(32,1),metadata=[('target_layers',9,values)])
    source=GgufSource([p])
    t=source.tensors['matrix']
    assert (t.shape,t.bytes,t.offset)==((1,32),34,offset)
    assert source.metadata['target_layers']==[5,14,24]
    assert list(source.chunks('matrix',chunk_bytes=13))==[bytes(range(13)),bytes(range(13,26)),bytes(range(26,34))]


def test_header_inspection_is_explicit_and_complete_source_rejects_truncation(tmp_path):
    p=tmp_path/'header.gguf'
    offset=fixture(p,payload=b'')
    with pytest.raises(ValueError,match='exceeds file'):
        read_header(p)
    h=read_header(p,expected_source_bytes=offset+34)
    assert h.header_only and h.tensors[0].bytes==34
    with pytest.raises(ValueError):
        GgufSource([p])


def test_invalid_block_stride_offsets_duplicate_keys_and_split_coverage(tmp_path):
    p=tmp_path/'invalid.gguf'
    fixture(p,shape=(31,))
    with pytest.raises(ValueError,match='alignment'):
        read_header(p)
    fixture(p,declared_offset=1)
    with pytest.raises(ValueError,match='alignment'):
        read_header(p)
    fixture(p,metadata=[('a',4,struct.pack('<I',1)),('a',4,struct.pack('<I',2))])
    with pytest.raises(ValueError,match='duplicate'):
        read_header(p)
    fixture(p,metadata=[('split.count',2,struct.pack('<H',2))])
    with pytest.raises(ValueError,match='incomplete'):
        GgufSource([p])

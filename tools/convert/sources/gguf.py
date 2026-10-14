"""Bounded GGUF-v3 source reads for offline conversion, never model execution.

Dimensions are exposed in logical C order, unlike GGUF's fastest-axis-first
directory. Encoded bytes are preserved; numerical decoding is a separate codec.
Header-only inspection requires explicit expected source lengths and disables
all payload reads. It must not be mistaken for a validated local checkpoint.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import prod
from pathlib import Path
import struct
from typing import BinaryIO, Iterator, Sequence

# Wire identities and exact encoded geometry, not NInfer execution registrations.
GGML_BLOCKS = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2),
    8: ("Q8_0", 32, 34), 10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210),
    18: ("IQ3_XXS", 256, 98), 21: ("IQ3_S", 256, 110),
    23: ("IQ4_XS", 256, 136), 30: ("BF16", 1, 2),
    39: ("MXFP4", 32, 17),
}
_SCALARS = {0:"B",1:"b",2:"H",3:"h",4:"I",5:"i",6:"f",7:"?",10:"Q",11:"q",12:"d"}


def _read(stream: BinaryIO, count: int) -> bytes:
    data = stream.read(count)
    if len(data) != count:
        raise ValueError("truncated GGUF header or payload")
    return data


def _scalar(stream: BinaryIO, fmt: str):
    return struct.unpack("<" + fmt, _read(stream, struct.calcsize(fmt)))[0]


def _string(stream: BinaryIO) -> str:
    size = _scalar(stream, "Q")
    if size > 64 * 2**20:
        raise ValueError("GGUF string exceeds the supported header extent")
    return _read(stream, size).decode("utf-8")


def _value(stream: BinaryIO, kind: int):
    if kind in _SCALARS:
        return _scalar(stream, _SCALARS[kind])
    if kind == 8:
        return _string(stream)
    if kind == 9:
        element, count = _scalar(stream,"I"), _scalar(stream,"Q")
        if element == 9 or count > 16 * 2**20:
            raise ValueError("unsupported GGUF array geometry")
        return [_value(stream, element) for _ in range(count)]
    raise ValueError(f"unsupported GGUF metadata type {kind}")


@dataclass(frozen=True, slots=True)
class GgufTensor:
    file: Path
    name: str
    shape: tuple[int, ...]
    ggml_type: int
    offset: int
    bytes: int


@dataclass(frozen=True, slots=True)
class GgufHeader:
    file: Path
    metadata: dict
    tensors: tuple[GgufTensor, ...]
    data_offset: int
    source_bytes: int
    header_only: bool


def read_header(path: str | Path, *, expected_source_bytes: int | None = None) -> GgufHeader:
    path = Path(path)
    local_bytes = path.stat().st_size
    source_bytes = local_bytes if expected_source_bytes is None else expected_source_bytes
    if type(source_bytes) is not int or source_bytes < local_bytes or source_bytes > 2**63-1:
        raise ValueError("invalid expected GGUF source length")
    with path.open("rb") as stream:
        if _read(stream,4) != b"GGUF" or _scalar(stream,"I") != 3:
            raise ValueError("only little-endian GGUF v3 is accepted")
        count, keys = _scalar(stream,"Q"), _scalar(stream,"Q")
        if count > 1_000_000 or keys > 1_000_000:
            raise ValueError("GGUF directory exceeds the supported extent")
        metadata = {}
        for _ in range(keys):
            key = _string(stream)
            if key in metadata:
                raise ValueError(f"duplicate GGUF metadata key {key}")
            metadata[key] = _value(stream,_scalar(stream,"I"))
        alignment = metadata.get("general.alignment",32)
        if type(alignment) is not int or alignment <= 0 or alignment & (alignment-1):
            raise ValueError("GGUF alignment must be a positive power of two")
        entries = []
        names = set()
        for _ in range(count):
            name = _string(stream)
            rank = _scalar(stream,"I")
            if name in names or not 1 <= rank <= 4:
                raise ValueError("duplicate tensor or invalid GGUF rank")
            names.add(name)
            wire_shape = tuple(_scalar(stream,"Q") for _ in range(rank))
            kind, relative = _scalar(stream,"I"), _scalar(stream,"Q")
            if kind not in GGML_BLOCKS or not all(wire_shape):
                raise ValueError(f"unsupported GGUF tensor type/shape: {name}, {kind}")
            _, group, block_bytes = GGML_BLOCKS[kind]
            if wire_shape[0] % group or relative % alignment:
                raise ValueError(f"GGUF tensor row/block alignment mismatch: {name}")
            size = prod(wire_shape) // group * block_bytes
            entries.append((name, tuple(reversed(wire_shape)), kind, relative, size))
        data_offset = (stream.tell()+alignment-1)//alignment*alignment
        if data_offset > local_bytes:
            raise ValueError("GGUF directory extends beyond the local header")
    tensors = tuple(GgufTensor(path,n,s,t,data_offset+o,b) for n,s,t,o,b in entries)
    previous_end = data_offset
    for tensor in sorted(tensors,key=lambda t:t.offset):
        if tensor.offset < previous_end or tensor.offset+tensor.bytes > source_bytes:
            raise ValueError(f"GGUF tensor payload overlaps or exceeds file: {tensor.name}")
        previous_end = tensor.offset+tensor.bytes
    return GgufHeader(path,metadata,tensors,data_offset,source_bytes,expected_source_bytes is not None)


class GgufSource:
    """Explicit complete split set with bounded payload reads and exact coverage."""

    def __init__(self, paths: Sequence[str | Path]) -> None:
        self.headers = tuple(read_header(path) for path in paths)
        if not self.headers:
            raise ValueError("GGUF source needs an explicit split set")
        split_count = self.headers[0].metadata.get("split.count",1)
        if split_count != len(self.headers):
            raise ValueError("incomplete GGUF split set")
        split_ids = [h.metadata.get("split.no",0) for h in self.headers]
        if sorted(split_ids) != list(range(split_count)):
            raise ValueError("duplicate or missing GGUF split ordinal")
        self.metadata = next(h.metadata for h in self.headers if h.metadata.get("split.no",0)==0)
        self.tensors = {}
        for header in self.headers:
            if header.metadata.get("split.count",1) != split_count:
                raise ValueError("inconsistent GGUF split counts")
            for tensor in header.tensors:
                if tensor.name in self.tensors:
                    raise ValueError(f"duplicate tensor across GGUF parts: {tensor.name}")
                self.tensors[tensor.name] = tensor
        if self.metadata.get("split.tensors.count",len(self.tensors)) != len(self.tensors):
            raise ValueError("GGUF split tensor coverage is incomplete")

    def chunks(self, name: str, *, chunk_bytes: int = 8 * 2**20) -> Iterator[bytes]:
        if type(chunk_bytes) is not int or chunk_bytes <= 0:
            raise ValueError("positive source chunk size required")
        tensor = self.tensors[name]
        with tensor.file.open("rb") as stream:
            stream.seek(tensor.offset)
            remaining = tensor.bytes
            while remaining:
                data = _read(stream,min(remaining,chunk_bytes))
                yield data
                remaining -= len(data)

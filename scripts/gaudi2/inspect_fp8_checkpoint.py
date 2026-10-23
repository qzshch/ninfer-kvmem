"""Inspect official block-FP8 checkpoints without reading tensor payloads.

Produces absolute file offsets for ninfer-gaudi2-fp8-tile. This is a source
inspection tool, not a .ninfer converter or an HPU inference implementation.
"""
import argparse
import json
import math
from pathlib import Path
import struct
import sys


HEADER_LIMIT = 16 * 1024 * 1024
DTYPE_BYTES = {
    'BOOL': 1, 'U8': 1, 'I8': 1, 'F8_E4M3': 1, 'F8_E5M2': 1,
    'I16': 2, 'U16': 2, 'F16': 2, 'BF16': 2,
    'I32': 4, 'U32': 4, 'F32': 4, 'I64': 8, 'U64': 8, 'F64': 8,
}


def unique_object(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=unique_object)


def integers(values):
    return isinstance(values, list) and all(type(x) is int and x >= 0 for x in values)


def read_header(path):
    """Read the bounded header only; validate every tensor's physical extent."""
    size = path.stat().st_size
    with path.open('rb') as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f'{path.name}: truncated safetensors prefix')
        length, = struct.unpack('<Q', prefix)
        if not 2 <= length <= HEADER_LIMIT or length > size - 8:
            raise ValueError(f'{path.name}: invalid or oversized safetensors header')
        raw = stream.read(length)
        if not raw.startswith(b'{'):
            raise ValueError(f'{path.name}: safetensors header must start with an object')
        header = json.loads(raw, object_pairs_hook=unique_object)
    if not isinstance(header, dict):
        raise ValueError(f'{path.name}: tensor directory must be an object')
    tensors = {}
    extents = []
    data_base = 8 + length
    for name, info in header.items():
        if name == '__metadata__':
            continue
        if not isinstance(info, dict):
            raise ValueError(f'{name}: invalid tensor directory entry')
        shape, offsets, dtype = info.get('shape'), info.get('data_offsets'), info.get('dtype')
        if not integers(shape) or not integers(offsets) or len(offsets) != 2:
            raise ValueError(f'{name}: invalid shape/data_offsets')
        begin, end = offsets
        if dtype not in DTYPE_BYTES:
            raise ValueError(f'{name}: unsupported stored dtype {dtype}')
        if end < begin or end - begin != math.prod(shape) * DTYPE_BYTES[dtype]:
            raise ValueError(f'{name}: stored byte length does not match dtype/shape')
        if end > size - data_base:
            raise ValueError(f'{name}: tensor extends past end of file')
        extents.append((begin, end))
        tensors[name] = dict(name=name, file=str(path.resolve()), dtype=dtype,
                             shape=shape, offset=data_base + begin, bytes=end - begin)
    cursor = 0
    for begin, end in sorted(extents):
        if begin != cursor:
            raise ValueError(f'{path.name}: tensor data has a hole or overlap')
        cursor = end
    if cursor != size - data_base:
        raise ValueError(f'{path.name}: unindexed trailing tensor data')
    return tensors, data_base


def inspect_checkpoint(directory):
    directory = Path(directory).resolve()
    config = read_json(directory / 'config.json')
    if not isinstance(config, dict):
        raise ValueError('checkpoint config must be an object')
    quant = config.get('quantization_config', {})
    if not isinstance(quant, dict):
        raise ValueError('quantization_config must be an object')
    if quant.get('quant_method') != 'fp8' or quant.get('weight_block_size') != [128, 128]:
        raise ValueError('expected quant_method=fp8 and weight_block_size=[128,128]')
    index_path = directory / 'model.safetensors.index.json'
    if index_path.exists():
        index = read_json(index_path)
        if not isinstance(index, dict):
            raise ValueError('checkpoint index must be an object')
        weight_map = index.get('weight_map')
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError('checkpoint index has no weight_map')
        if not all(isinstance(name, str) and isinstance(file, str)
                   for name, file in weight_map.items()):
            raise ValueError('invalid checkpoint weight_map')
        shards = sorted(set(weight_map.values()))
    else:
        weight_map = None
        shards = ['model.safetensors']
    tensors = {}
    header_bytes = 0
    for shard in shards:
        entries, size = read_header(directory / shard)
        header_bytes += size
        for name, info in entries.items():
            if name in tensors:
                raise ValueError(f'duplicate tensor across shards: {name}')
            if weight_map is not None and weight_map.get(name) != shard:
                raise ValueError(f'{name}: index disagrees with shard tensor directory')
            tensors[name] = info
    if weight_map is not None and set(weight_map) != set(tensors):
        raise ValueError('checkpoint index contains missing tensors')
    pairs = []
    used_scales = set()
    for name, weight in sorted(tensors.items()):
        if not weight['dtype'].startswith('F8_'):
            continue
        if weight['dtype'] != 'F8_E4M3' or not name.endswith('.weight'):
            raise ValueError(f'{name}: expected E4M3FN FP8 weight tensor')
        if len(weight['shape']) != 2 or any(x == 0 or x % 128 for x in weight['shape']):
            raise ValueError(f'{name}: checkpoint inspector currently requires nonzero 128-aligned [N,K]')
        scale_name = name[:-len('weight')] + 'weight_scale_inv'
        scale = tensors.get(scale_name)
        n, k = weight['shape']
        if scale is None or scale['dtype'] != 'F32' or scale['shape'] != [n // 128, k // 128]:
            raise ValueError(f'{name}: missing or incompatible F32 block weight_scale_inv')
        pairs.append(dict(weight=weight, scale=scale, rows=n, cols=k,
                          profile='e4m3fn-block128-w8a16-bf16-rne'))
        used_scales.add(scale_name)
    if not pairs:
        raise ValueError('checkpoint contains no supported block-FP8 matrices')
    orphan_scales = [name for name in tensors if name.endswith('.weight_scale_inv')
                     and name not in used_scales]
    if orphan_scales:
        raise ValueError(f'orphan block scales: {orphan_scales[0]}')
    return dict(schema_version=1, checkpoint=str(directory),
                scope='source inspection; not .ninfer conversion or HPU inference',
                source_files=[str(directory / 'config.json')]
                + ([str(index_path)] if weight_map is not None else [])
                + [str((directory / shard).resolve()) for shard in shards],
                tensor_count=len(tensors), shard_count=len(shards),
                header_bytes_read=header_bytes,
                tensor_payload_bytes=sum(x['bytes'] for x in tensors.values()),
                fp8_weight_bytes=sum(x['weight']['bytes'] for x in pairs),
                fp32_scale_bytes=sum(x['scale']['bytes'] for x in pairs), matrices=pairs)


def protect_sources(output, sources):
    for source in map(Path, sources):
        if output.resolve() == source.resolve() or (output.exists() and output.samefile(source)):
            raise ValueError(f'output would overwrite checkpoint source: {source.name}')


def tile_command(matrix, executable, tile, output):
    row, col, rows, cols = tile
    if min(row, col) < 0 or min(rows, cols) <= 0:
        raise ValueError('tile origin must be nonnegative and tile size positive')
    if row + rows > matrix['rows'] or col + cols > matrix['cols']:
        raise ValueError('tile exceeds matrix bounds')
    return [str(executable), matrix['weight']['file'], matrix['scale']['file'],
            str(matrix['rows']), str(matrix['cols']), str(row), str(col), str(rows), str(cols),
            str(output), '--codes-offset', str(matrix['weight']['offset']),
            '--scales-offset', str(matrix['scale']['offset'])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--output', type=Path, help='write inventory JSON; otherwise stdout')
    parser.add_argument('--tensor', help='emit a tile command for this weight tensor')
    parser.add_argument('--tile', nargs=4, type=int, metavar=('ROW', 'COL', 'ROWS', 'COLS'))
    parser.add_argument('--tile-output', type=Path)
    parser.add_argument('--tile-executable', default='ninfer-gaudi2-fp8-tile')
    args = parser.parse_args()
    try:
        result = inspect_checkpoint(args.checkpoint)
        for output in (args.output, args.tile_output):
            if output is not None:
                protect_sources(output, result['source_files'])
        if any((args.tensor, args.tile, args.tile_output)):
            if not all((args.tensor, args.tile, args.tile_output)):
                raise ValueError('--tensor, --tile and --tile-output must be used together')
            matrix = next((m for m in result['matrices'] if m['weight']['name'] == args.tensor), None)
            if matrix is None:
                raise ValueError(f'unsupported/missing FP8 tensor: {args.tensor}')
            result['tile_command'] = tile_command(matrix, args.tile_executable, args.tile,
                                                   args.tile_output)
        payload = json.dumps(result, indent=2) + '\n'
        if args.output:
            args.output.write_text(payload, encoding='utf-8')
        else:
            print(payload, end='')
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(f'Checkpoint inspection failed: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())

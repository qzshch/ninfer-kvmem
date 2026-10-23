"""Source format/offset tests; no tensor library, accelerator or model download."""
import importlib.util
from datetime import datetime, timezone
import contextlib
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('checkpoint_inspector',
    ROOT / 'scripts/gaudi2/inspect_fp8_checkpoint.py')
inspector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inspector)
WEIGHT = 'model.language_model.layers.0.self_attn.q_proj.weight'
SCALE = WEIGHT[:-len('weight')] + 'weight_scale_inv'


def shard(path, entries):
    """Make an independent safetensors fixture, including real payload bytes."""
    header = {}
    data = bytearray()
    for name, dtype, shape, payload in entries:
        start = len(data)
        data.extend(payload)
        header[name] = dict(dtype=dtype, shape=shape, data_offsets=[start, len(data)])
    encoded = json.dumps(header).encode('utf-8')
    encoded += b' ' * (-len(encoded) % 8)
    path.write_bytes(struct.pack('<Q', len(encoded)) + encoded + data)
    return 8 + len(encoded)


class CheckpointTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'config.json').write_text(json.dumps({'quantization_config': {
            'quant_method': 'fp8', 'weight_block_size': [128, 128]}}), encoding='utf-8')

    def single(self, scale_shape=None, scale_dtype='F32', weight_dtype='F8_E4M3'):
        scales = struct.pack('<HH', 0x3f00, 0x4000) if scale_dtype == 'BF16' else struct.pack('<ff', 0.5, 2.0)
        return shard(self.root / 'model.safetensors', [
            (WEIGHT, weight_dtype, [256, 128], bytes([0x7e]) * 32768),
            (SCALE, scale_dtype, scale_shape or [2, 1], scales),
            ('norm.weight', 'BF16', [2], b'\x80\x3f\x00\x40')])

    def test_same_file_planes_and_exact_absolute_offsets(self):
        base = self.single()
        result = inspector.inspect_checkpoint(self.root)
        self.assertEqual(result['tensor_count'], 3)
        self.assertEqual(result['header_bytes_read'], base)
        self.assertEqual(result['tensor_payload_bytes'], 32780)
        matrix, = result['matrices']
        self.assertEqual(matrix['weight']['offset'], base)
        self.assertEqual(matrix['scale']['offset'], base + 32768)
        command = inspector.tile_command(matrix, 'tile.exe', [127, 1, 2, 7], 'tile.bf16')
        self.assertEqual(command[-4:], ['--codes-offset', str(base), '--scales-offset', str(base + 32768)])
        self.assertEqual(command[3:9], ['256', '128', '127', '1', '2', '7'])

    def test_pairs_can_span_indexed_shards(self):
        a = shard(self.root / 'a.safetensors', [(WEIGHT, 'F8_E4M3', [256, 128], bytes(32768))])
        b = shard(self.root / 'b.safetensors', [(SCALE, 'F32', [2, 1], struct.pack('<ff', 1, 1))])
        (self.root / 'model.safetensors.index.json').write_text(json.dumps({
            'weight_map': {WEIGHT: 'a.safetensors', SCALE: 'b.safetensors'}}), encoding='utf-8')
        result = inspector.inspect_checkpoint(self.root)
        matrix, = result['matrices']
        self.assertEqual((matrix['weight']['offset'], matrix['scale']['offset']), (a, b))
        self.assertEqual(result['shard_count'], 2)
        self.assertEqual(result['header_bytes_read'], a + b)
        index = {'weight_map': {WEIGHT: 'b.safetensors', SCALE: 'b.safetensors'}}
        (self.root / 'model.safetensors.index.json').write_text(json.dumps(index), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'missing tensors'):
            inspector.inspect_checkpoint(self.root)

    def test_inspection_reads_only_prefix_and_header(self):
        self.single()
        original_open = Path.open
        reads = []

        class HeaderOnly:
            def __init__(self, stream):
                self.stream = stream
            def __enter__(self):
                return self
            def __exit__(self, *args):
                self.stream.close()
            def read(self, count=-1):
                reads.append(count)
                if len(reads) > 2 or count < 0:
                    raise AssertionError('tensor payload read')
                return self.stream.read(count)

        def open_path(path, *args, **kwargs):
            stream = original_open(path, *args, **kwargs)
            return HeaderOnly(stream) if path.suffix == '.safetensors' else stream

        with patch.object(Path, 'open', open_path):
            result = inspector.inspect_checkpoint(self.root)
        self.assertEqual(reads[0], 8)
        self.assertEqual(sum(reads), result['header_bytes_read'])

    def test_reject_row_scale_wrong_fp8_and_bad_scale_type(self):
        for kwargs, message in (({'scale_shape': [2]}, 'incompatible F32 block'),
                                ({'scale_dtype': 'BF16'}, 'incompatible F32 block'),
                                ({'weight_dtype': 'F8_E5M2'}, 'expected E4M3FN')):
            with self.subTest(kwargs=kwargs):
                self.single(**kwargs)
                with self.assertRaisesRegex(ValueError, message):
                    inspector.inspect_checkpoint(self.root)
        self.single()
        (self.root / 'config.json').write_text('{"quantization_config":{"quant_method":"fp8"}}')
        with self.assertRaisesRegex(ValueError, 'weight_block_size'):
            inspector.inspect_checkpoint(self.root)

    def test_truncation_overlap_and_header_limit(self):
        path = self.root / 'model.safetensors'
        self.single()
        path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, 'end of file'):
            inspector.read_header(path)
        header = {'a': {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]},
                  'b': {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}}
        raw = json.dumps(header).encode()
        path.write_bytes(struct.pack('<Q', len(raw)) + raw + bytes(8))
        with self.assertRaisesRegex(ValueError, 'overlap'):
            inspector.read_header(path)
        path.write_bytes(struct.pack('<Q', inspector.HEADER_LIMIT + 1))
        with self.assertRaisesRegex(ValueError, 'oversized'):
            inspector.read_header(path)

    def test_tile_bounds_and_duplicate_json_keys(self):
        self.single()
        matrix, = inspector.inspect_checkpoint(self.root)['matrices']
        for tile in ([-1, 0, 1, 1], [0, 0, 0, 1], [255, 0, 2, 1], [0, 127, 1, 2]):
            with self.subTest(tile=tile), self.assertRaises(ValueError):
                inspector.tile_command(matrix, 'tile', tile, 'output')
        (self.root / 'config.json').write_text('{"a":1,"a":2}')
        with self.assertRaisesRegex(ValueError, 'duplicate JSON key'):
            inspector.inspect_checkpoint(self.root)

    def test_cli_rejects_overwriting_source_and_hardlink(self):
        self.single()
        alias = self.root / 'alias.json'
        alias.hardlink_to(self.root / 'model.safetensors')
        for name in ('config.json', 'model.safetensors', 'alias.json'):
            output = self.root / name
            before = output.read_bytes()
            with patch('sys.argv', ['inspect', str(self.root), '--output', str(output)]), \
                    contextlib.redirect_stderr(io.StringIO()) as errors:
                self.assertEqual(inspector.main(), 1)
            self.assertIn('overwrite checkpoint source', errors.getvalue())
            self.assertEqual(output.read_bytes(), before)

    def test_invalid_configuration_types_are_reported(self):
        self.single()
        for config in ([], {'quantization_config': None}, {'quantization_config': []}):
            (self.root / 'config.json').write_text(json.dumps(config))
            with patch('sys.argv', ['inspect', str(self.root)]), \
                    contextlib.redirect_stderr(io.StringIO()) as errors:
                self.assertEqual(inspector.main(), 1)
            self.assertIn('must be an object', errors.getvalue())

    @unittest.skipUnless(os.environ.get('NINFER_GAUDI2_TILE_EXE'), 'native tile executable not supplied')
    def test_native_reader_on_checkpoint_plane_offsets(self):
        self.single()
        matrix, = inspector.inspect_checkpoint(self.root)['matrices']
        output = self.root / 'tile.bf16'
        command = inspector.tile_command(matrix, os.environ['NINFER_GAUDI2_TILE_EXE'],
                                         [127, 1, 2, 7], output)
        started = datetime.now(timezone.utc).isoformat()
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(json.dumps({'purpose': 'native checkpoint offset integration test',
                          'pid': process.pid, 'started_utc': started,
                          'command': command[0]}), flush=True)
        try:
            stdout, stderr = process.communicate(timeout=10)
            self.assertEqual(process.returncode, 0, stderr.decode(errors='replace'))
            json.loads(stdout)
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
        # E4M3FN 0x7e = 448; row 127 uses scale .5, row 128 scale 2.
        self.assertEqual(output.read_bytes(), struct.pack('<14H', *([0x4360] * 7 + [0x4460] * 7)))


if __name__ == '__main__':
    unittest.main()

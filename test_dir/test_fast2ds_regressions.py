import contextlib
import configparser
import datetime
import io
import os
import tempfile
import unittest

import numpy

import nrc_spifpy.input.fast_2ds_file as fast_2ds_module
from nrc_spifpy.input.fast_2ds_file import Fast2DSFile
from nrc_spifpy.spif import SPIFFile


class FakeSPIFFile:
    def __init__(self):
        self.writes = {}

    def set_start_date(self, value):
        self.start_date = value

    def create_inst_group(self, name):
        pass

    def set_filenames_attr(self, name, filename):
        pass

    def write_buffer_info(self, start_date, datetimes):
        pass

    def write_images(self, name, images):
        self.writes.setdefault(name, []).append({
            'image_size': len(images.image),
            'length': list(images.length),
        })


class TestFast2DSRegressions(unittest.TestCase):
    def setUp(self):
        self.original_has_cython = fast_2ds_module.HAS_CYTHON

    def tearDown(self):
        fast_2ds_module.HAS_CYTHON = self.original_has_cython

    @staticmethod
    def make_file(n_frames):
        fast_2ds = Fast2DSFile('dummy.F2DS', 'F2DS', 10)
        fast_2ds.data = numpy.zeros(n_frames, dtype=fast_2ds.file_dtype)
        fast_2ds.start_date = datetime.datetime(2024, 1, 1)
        fast_2ds.datetimes = numpy.array([
            numpy.datetime64('2024-01-01T00:00:00') + numpy.timedelta64(i, 's')
            for i in range(n_frames)
        ])
        fast_2ds.tas = numpy.full(n_frames, 100.0)
        fast_2ds.user_temp = numpy.full(n_frames, 21.5)
        fast_2ds.ps_temp = numpy.full(n_frames, 31.5)
        return fast_2ds

    def implementations(self):
        implementations = [('python', False)]
        if hasattr(fast_2ds_module, 'decode_frame'):
            implementations.append(('cython', True))
        return implementations

    def test_split_header_and_oversized_packet_are_safe(self):
        split = self.make_file(2)
        split.data[0]['data'][2046:2048] = [split.SYNC_2S, 4]
        split.data[1]['data'][:7] = [
            0, 7, 1, split.RLE_FULL_SHADED, 100, 0, 0,
        ]

        for name, use_cython in self.implementations():
            with self.subTest(implementation=name, case='split-header'):
                fast_2ds_module.HAS_CYTHON = use_cython
                result = split.process_frame_with_state(
                    0, {'pending_h': {}, 'pending_v': {}}
                )
                self.assertEqual(len(result['h'][0]['data']), 128)
                self.assertEqual(result['next_idx'], 7)

        oversized = self.make_file(2)
        oversized.data[0]['data'][2046:2048] = [oversized.SYNC_2S, 0x0FFF]
        oversized.data[1]['data'][:3] = [0, 8, 1]

        for name, use_cython in self.implementations():
            with self.subTest(implementation=name, case='oversized'):
                fast_2ds_module.HAS_CYTHON = use_cython
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    result = oversized.process_frame_with_state(
                        0, {'pending_h': {}, 'pending_v': {}}
                    )
                self.assertEqual(result['h'], [])
                self.assertEqual(result['v'], [])
                self.assertEqual(result['next_idx'], 0)
                self.assertIn('spans more than two frames', output.getvalue())

    def test_process_file_lookahead_skips_consumed_prefix(self):
        fast_2ds = self.make_file(501)
        fast_2ds.data[499]['data'][2046:2048] = [fast_2ds.SYNC_2S, 4]
        fast_2ds.data[500]['data'][:7] = [
            0, 42, 1, fast_2ds.RLE_FULL_SHADED, 100, 0, 0,
        ]

        output = FakeSPIFFile()
        fast_2ds_module.HAS_CYTHON = hasattr(fast_2ds_module, 'decode_frame')
        fast_2ds.process_file(output, processors=4)

        writes = output.writes['F2DS-H']
        self.assertEqual(len(writes), 1)
        self.assertEqual(writes[0]['length'], [1])
        self.assertEqual(writes[0]['image_size'], 128)

    def test_multiprocessing_uses_fixed_chunks_with_lookahead(self):
        fast_2ds = self.make_file(1001)
        fast_2ds.data[499]['data'][2046:2048] = [fast_2ds.SYNC_2S, 4]
        fast_2ds.data[500]['data'][:7] = [
            0, 42, 1, fast_2ds.RLE_FULL_SHADED, 100, 0, 0,
        ]
        fast_2ds.data[700]['data'][:9] = [
            fast_2ds.SYNC_2S,
            4,
            0,
            43,
            1,
            fast_2ds.RLE_FULL_SHADED,
            200,
            0,
            0,
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            filename = os.path.join(temp_dir, 'synthetic.F2DS')
            fast_2ds.data.tofile(filename)
            fast_2ds.filename = filename

            output = FakeSPIFFile()
            fast_2ds.process_file(output, processors=2)

        writes = output.writes['F2DS-H']
        self.assertEqual(
            [length for write in writes for length in write['length']],
            [1, 1],
        )
        self.assertEqual(sum(write['image_size'] for write in writes), 256)

    def test_extract_images_preserves_counts_and_uses_buffer_time(self):
        fast_2ds = self.make_file(1)
        pixels = numpy.ones(128, dtype=numpy.uint8)
        h_images, v_images = fast_2ds.extract_images({
            'h': [{
                'time': 200,
                'buffer_index': 0,
                'word_index': 20,
                'data': pixels,
            }],
            'v': [{
                'time': 100,
                'buffer_index': 0,
                'word_index': 10,
                'data': pixels,
            }],
        })

        self.assertEqual(h_images.clock_counts, [200])
        self.assertEqual(v_images.clock_counts, [100])
        self.assertEqual(h_images.ns, [0])
        self.assertEqual(v_images.ns, [0])
        self.assertEqual(h_images.timing_quality, [3])
        self.assertEqual(v_images.timing_quality, [3])

    def test_temperature_channels_are_preserved(self):
        fast_2ds = self.make_file(1)
        fast_2ds.tas[:] = numpy.nan
        pixels = numpy.ones(128, dtype=numpy.uint8)
        h_images, _ = fast_2ds.extract_images({
            'h': [
                {
                    'time': 100,
                    'buffer_index': 0,
                    'word_index': 10,
                    'data': pixels,
                },
                {
                    'time': 200,
                    'buffer_index': 0,
                    'word_index': 20,
                    'data': pixels,
                },
            ],
            'v': [],
        })

        self.assertEqual(h_images.ns, [0, 0])
        self.assertTrue(numpy.isnan(h_images.tas).all())
        self.assertEqual(h_images.user_temp, [21.5, 21.5])
        self.assertEqual(h_images.ps_temp, [31.5, 31.5])

    def test_netcdf_preserves_48_bit_counts_and_finalizes_times(self):
        fast_2ds = self.make_file(2)
        particle_count = 22_500_000
        fast_2ds.data[0]['data'][:9] = [
            fast_2ds.SYNC_2S,
            4,
            0,
            1,
            1,
            fast_2ds.RLE_FULL_SHADED,
            particle_count & 0xFFFF,
            (particle_count >> 16) & 0xFFFF,
            (particle_count >> 32) & 0xFFFF,
        ]
        fast_2ds.hk_counts = numpy.array(
            [20_000_000, 30_000_000], dtype=numpy.uint64
        )
        fast_2ds.hk_datetimes = fast_2ds.datetimes.copy()
        fast_2ds.hk_tas = numpy.full(2, 100.0)

        config = configparser.ConfigParser()
        config.read(os.path.join(
            os.path.dirname(__file__),
            '..',
            'nrc_spifpy',
            'config',
            'Fast2DS.ini',
        ))

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, 'fast2ds.nc')
            output = SPIFFile(output_path, config)
            output.create_file()
            try:
                fast_2ds.process_file(output, processors=1)
                core = output.instgrps['F2DS-H']['core']
                self.assertEqual(core['clock_counts'].dtype, numpy.dtype('uint64'))
                self.assertEqual(int(core['clock_counts'][0]), particle_count)
                self.assertEqual(int(core['image_sec'][0]), 0)
                self.assertAlmostEqual(
                    int(core['image_ns'][0]), 250_000_000, delta=1
                )
                self.assertEqual(int(core['timing_quality'][0]), 1)
                self.assertEqual(
                    core['timing_quality'].flag_meanings,
                    'hk_anchored_probe_counter '
                    'buffer_anchored_probe_counter '
                    'buffer_timestamp invalid_buffer_index',
                )
            finally:
                output.close()


if __name__ == '__main__':
    unittest.main()

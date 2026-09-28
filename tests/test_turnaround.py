"""Offline regression checks for the reproducible reference pack.
Run: python3 -m unittest discover -s tests -v
"""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
from PIL import Image, ImageChops, ImageOps

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('build_turnaround', ROOT / 'tools' / 'build_turnaround.py')
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)
HAS_PSD = importlib.util.find_spec('psd_tools') is not None


class TurnaroundTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.temp.name)
        cls.manifest = builder.build(cls.out, psd=HAS_PSD)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_full_rotation_order_and_disclosed_drafts(self):
        frames = self.manifest['frames']
        self.assertEqual([f['angle'] for f in frames], list(range(0, 360, 45)))
        self.assertEqual([f['angle'] for f in frames if f['mirrored']], [45, 135, 270])
        self.assertEqual(len({f['id'] for f in frames}), 8)
        self.assertEqual(self.manifest['kind'], 'reference_pack_not_live2d_rig')
        for frame in frames:
            self.assertEqual(frame['status'] == 'mirror_draft', frame['mirrored'])

    def test_common_canvas_transparency_and_alignment(self):
        for frame in self.manifest['frames']:
            with self.subTest(frame=frame['id']), Image.open(self.out / frame['file']) as im:
                self.assertEqual(im.size, (768, 1024))
                self.assertEqual(im.mode, 'RGBA')
                alpha = np.asarray(im.getchannel('A'))
                self.assertFalse(alpha[:85].any(), 'No baked labels above character')
                self.assertFalse(alpha[980:].any(), 'No cropping at feet')
                self.assertFalse(alpha[:, :20].any())
                self.assertFalse(alpha[:, -20:].any())
                ys, xs = np.where(alpha > 127)
                self.assertGreater(len(xs), 80000)
                self.assertLessEqual(abs(ys.min() - 94), 2)
                self.assertLessEqual(abs(ys.max() - 973), 2)
                self.assertLessEqual(abs((xs.min() + xs.max()) / 2 - 384), 2)
                self.assertEqual(frame['pivot'], [384, 974])

    def test_mirrors_are_exact_and_explicit(self):
        frames = self.manifest['frames']
        for a, b in [(1, 7), (3, 5), (6, 2)]:
            with Image.open(self.out / frames[a]['file']) as left, Image.open(self.out / frames[b]['file']) as right:
                # Normalize by content bounds: integer centering can differ by 1px.
                left = left.crop(left.getbbox())
                right = ImageOps.mirror(right.crop(right.getbbox()))
                self.assertEqual(left.size, right.size)
                self.assertIsNone(ImageChops.difference(left, right).getbbox())

    def test_head_key_grid_not_claimed_calibrated(self):
        heads = self.manifest['head_frames']
        self.assertEqual(len(heads), 9)
        keys = {(f['target_parameters']['ParamAngleX'], f['target_parameters']['ParamAngleY']) for f in heads}
        self.assertEqual(keys, {(x, y) for x in (-30, 0, 30) for y in (-30, 0, 30)})
        for frame in heads:
            self.assertFalse(frame['calibrated'])
            with Image.open(self.out / frame['file']) as im:
                self.assertEqual(im.size, (512, 512))
                self.assertEqual(im.mode, 'RGB')

    def test_sheet_cells_match_individual_exports(self):
        with Image.open(self.out / 'body_sprite_sheet.png') as sheet:
            self.assertEqual(sheet.size, (3072, 2048))
            for i, frame in enumerate(self.manifest['frames']):
                x, y = i % 4 * 768, i // 4 * 1024
                with Image.open(self.out / frame['file']) as im:
                    self.assertIsNone(ImageChops.difference(sheet.crop((x, y, x + 768, y + 1024)), im).getbbox())

    def test_animation_and_csv_timing(self):
        with Image.open(self.out / 'turnaround.gif') as im:
            self.assertEqual(im.n_frames, 8)
            self.assertEqual(im.info['loop'], 0)
            for i in range(8):
                im.seek(i)
                self.assertEqual(im.info['duration'], 500)
        with (self.out / 'storyboard.csv').open(encoding='utf-8-sig', newline='') as file:
            rows = list(csv.DictReader(file))
        self.assertEqual([int(r['time_ms']) for r in rows], list(range(0, 4000, 500)))
        self.assertEqual(rows[1]['status'], 'mirror_draft')
        self.assertEqual(rows[0]['label'], 'Спереди')

    def test_manifest_provenance_and_archive(self):
        saved = json.loads((self.out / 'manifest.json').read_text())
        self.assertEqual(saved, self.manifest)
        for file, digest in saved['source_sha256'].items():
            self.assertEqual(hashlib.sha256((builder.SOURCES / file).read_bytes()).hexdigest(), digest)
        with zipfile.ZipFile(self.out / saved['downloads']['zip']) as archive:
            self.assertIsNone(archive.testzip())
            self.assertEqual(len(archive.namelist()), len(set(archive.namelist())))
            for file in archive.namelist():
                path = Path(file)
                self.assertNotIn('..', path.parts)
                self.assertEqual(path.parts[0], 'ChibiVT_turnaround')
                self.assertNotEqual(path.suffix, '.zip')
                relative = Path(*path.parts[1:])
                self.assertEqual(archive.read(file), (self.out / relative).read_bytes())

    @unittest.skipUnless(HAS_PSD, 'psd-tools is optional')
    def test_psd_has_reference_groups_and_only_front_visible(self):
        from psd_tools import PSDImage
        psd = PSDImage.open(self.out / 'ChibiVT_turnaround_references.psd')
        self.assertEqual(psd.size, (768, 1024))
        self.assertEqual(len(psd[1]), 8)
        self.assertEqual(len(psd[2]), 9)
        self.assertFalse(psd[0].visible)
        self.assertFalse(psd[2].visible)
        self.assertEqual([layer.visible for layer in psd[1]], [True] + [False] * 7)
        self.assertEqual(sum('MIRROR_DRAFT' in layer.name for layer in psd[1]), 3)
        self.assertIsNotNone(psd.composite())

    def test_all_static_downloads_resolve(self):
        from html.parser import HTMLParser
        class Links(HTMLParser):
            def __init__(self):
                super().__init__()
                self.paths = []
            def handle_starttag(self, tag, attrs):
                for key, value in attrs:
                    if key in ('src', 'href') and value and not value.startswith(('#', 'http')):
                        self.paths.append(value)
        links = Links()
        links.feed((ROOT / 'turnaround' / 'index.html').read_text())
        for path in links.paths:
            self.assertTrue((ROOT / 'turnaround' / path).exists(), path)


if __name__ == '__main__':
    unittest.main()

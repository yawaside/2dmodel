"""Офлайн-проверки пакета «Раскадровка глаз и рта» и порта блендинга.

Run: python3 -m unittest discover -s tests -v
Тесты не пересобирают весь пакет (он коммитится), но проверяют:
  - структуру и консистентность storyboard/pack (манифест ↔ файлы ↔ CSV);
  - свойства плавности mouth_blend / eye_blend (сумма весов, ≤2 кадра, соседи);
  - ТОЧНОЕ совпадение JS-порта (storyboard/blend.js) с Python-функциями модели;
  - интерактивные ассеты (тайлы, геометрия) на месте и согласованы с атласом.
"""
import csv
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / 'storyboard' / 'pack'
SPEC = importlib.util.spec_from_file_location(
    'build_face_storyboard', ROOT / 'tools' / 'build_face_storyboard.py')
sb = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sb)

from build_model import mouth_blend, eye_blend, stack_opa  # noqa: E402

HAS_NODE = shutil.which('node') is not None


def js_blend_sweep() -> dict:
    """Прогон blend.js через node по той же сетке, что в python-тесте."""
    script = (
        "const b = require(%r);"
        "const out = {mouth: [], eyes: []};"
        "for (let o = 0; o <= 100; o++) for (let f = -10; f <= 10; f++)"
        "  out.mouth.push(b.mouthBlend(o / 100, f / 10));"
        "for (let o = 0; o <= 100; o++) for (let s = 0; s <= 10; s++)"
        "  out.eyes.push(b.eyeBlend(o / 100, s / 10));"
        "console.log('===JSON===' + JSON.stringify(out));"
    ) % str(ROOT / 'storyboard' / 'blend.js')
    res = subprocess.run(['node', '-e', script], capture_output=True, text=True,
                         timeout=60)
    if res.returncode != 0:
        raise RuntimeError(res.stderr[-500:])
    return json.loads(res.stdout.split('===JSON===')[-1].strip().splitlines()[0])


class BlendProperties(unittest.TestCase):
    """Свойства плавности — то, ради чего делалась раскадровка."""

    def test_mouth_weights_sum_to_one(self):
        for o in [i / 100 for i in range(101)]:
            for f in [-1, -0.5, 0, 0.5, 1]:
                w = mouth_blend(o, f)
                self.assertAlmostEqual(sum(w), 1.0, places=9, msg=(o, f))

    def test_mouth_max_two_adjacent_frames(self):
        for o in [i / 100 for i in range(101)]:
            w = mouth_blend(o, 0.0)
            active = [i for i, x in enumerate(w) if x > 1e-6]
            self.assertLessEqual(len(active), 2, msg=o)
            if len(active) == 2:
                self.assertEqual(abs(active[0] - active[1]), 1,
                                 'смешиваются только соседние кадры: %s %s' % (o, w))

    def test_mouth_chain_is_monotone(self):
        """Цепочка closed→slight→half→A: с ростом open вес никогда не «прыгает» назад."""
        prev = None
        for o in [i / 200 for i in range(201)]:
            w = mouth_blend(o, 0.0)
            top = max(range(4), key=lambda i: w[i])  # доминирующий кадр цепочки
            if prev is not None:
                self.assertGreaterEqual(top, prev, msg=o)
            prev = top

    def test_eye_weights_sum_to_one(self):
        for o in [i / 100 for i in range(101)]:
            for s in [i / 10 for i in range(11)]:
                w = eye_blend(o, s)
                self.assertAlmostEqual(sum(w), 1.0, places=9, msg=(o, s))

    def test_eye_max_two_adjacent_frames(self):
        """Без улыбки — не более 2 соседних кадров; улыбка добавляет только
        хвост happy/squint (так нарисовано: прищур подмешивается к arke)."""
        for o in [i / 100 for i in range(101)]:
            for s in [0.0, 0.4, 1.0]:
                w = eye_blend(o, s)
                active = [i for i, x in enumerate(w) if x > 1e-6]
                chain = [i for i in active if i < 3]
                tail = [i for i in active if i >= 3]
                self.assertLessEqual(len(chain), 2, msg=(o, s))
                if len(chain) == 2:
                    self.assertEqual(abs(chain[0] - chain[1]), 1, msg=(o, s))
                if s == 0.0:
                    self.assertEqual(tail, [], msg=(o, s))
                else:
                    self.assertTrue(set(tail) <= {3, 4}, msg=(o, s))
                    if w[4] > 1e-6:
                        self.assertGreater(w[3], 0.0, 'прищур идёт поверх happy')

    def test_stack_opa_bottom_layer_is_opaque(self):
        """Нижний активный слой стека всегда полностью непрозрачен —
        базовая прорисовка не просвечивает (нет «призраков»)."""
        for o in [0.0, 0.15, 0.3, 0.55, 0.9]:
            for f in (0.0, 0.3, -0.8):
                w = mouth_blend(o, f)
                bottom = next(i for i, x in enumerate(w) if x > 1e-6)
                self.assertAlmostEqual(stack_opa(w, bottom), 1.0, places=9,
                                       msg=(o, f))
                for i in range(6):
                    self.assertLessEqual(stack_opa(w, i), 1.0 + 1e-9)


class JsPythonParity(unittest.TestCase):
    """JS-порт (blend.js в браузере) должен совпадать с Python-моделью точно."""

    @unittest.skipUnless(HAS_NODE, 'node недоступен')
    def test_blend_port_matches_model(self):
        js = js_blend_sweep()
        py_mouth, py_eyes = [], []
        for o in range(101):
            for f in range(-10, 11):
                py_mouth.append(mouth_blend(o / 100, f / 10))
        for o in range(101):
            for s in range(11):
                py_eyes.append(eye_blend(o / 100, s / 10))
        self.assertEqual(len(js['mouth']), len(py_mouth))
        self.assertEqual(len(js['eyes']), len(py_eyes))
        for i, (a, b) in enumerate(zip(js['mouth'], py_mouth)):
            for j, (x, y) in enumerate(zip(a, b)):
                self.assertAlmostEqual(x, y, places=9,
                                       msg='mouth #%d shape %d' % (i, j))
        for i, (a, b) in enumerate(zip(js['eyes'], py_eyes)):
            for j, (x, y) in enumerate(zip(a, b)):
                self.assertAlmostEqual(x, y, places=9,
                                       msg='eye #%d shape %d' % (i, j))


class PackStructure(unittest.TestCase):
    """Собранный пакет консистентен: манифест ↔ файлы ↔ CSV ↔ тайлы."""

    @classmethod
    def setUpClass(cls):
        if not PACK.exists():
            raise unittest.SkipTest('storyboard/pack не собран')
        cls.manifest = json.loads((PACK / 'manifest.json').read_text(encoding='utf-8'))

    def test_declared_files_exist(self):
        for group in ('mouth_frames', 'eye_frames', 'expressions'):
            for item in self.manifest[group]:
                self.assertTrue((PACK / item['file']).exists(), item['file'])
        for anim in self.manifest['animations']:
            self.assertTrue((PACK / anim['file']).exists(), anim['file'])
        for board in ('boards/mouth_board.png', 'boards/eye_board.png'):
            self.assertTrue((PACK / board).exists(), board)

    def test_counts(self):
        self.assertEqual(len(self.manifest['mouth_frames']), 6)
        self.assertEqual(len(self.manifest['eye_frames']), 5)
        self.assertEqual(len(self.manifest['expressions']), 4)
        self.assertEqual(len(self.manifest['mouth_timeline']), 7)
        self.assertEqual(len(self.manifest['eye_timeline']), 3)

    def test_timeline_covers_full_range(self):
        for key in ('mouth_timeline', 'eye_timeline'):
            tl = self.manifest[key]
            self.assertAlmostEqual(tl[0]['from'], 0.0)
            self.assertAlmostEqual(tl[-1]['to'], 1.0)
            for a, b in zip(tl, tl[1:]):
                self.assertAlmostEqual(a['to'], b['from'])

    def test_csv_matches_manifest(self):
        with (PACK / 'storyboard.csv').open(encoding='utf-8-sig') as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), 15)
        files = {r['file'] for r in rows}
        declared = {f['file'] for g in ('mouth_frames', 'eye_frames', 'expressions')
                    for f in self.manifest[g]}
        self.assertEqual(files, declared)

    def test_interactive_assets_consistent(self):
        inter = self.manifest['interactive']
        self.assertEqual(len(inter['mouth_tiles']), 6)
        self.assertEqual(len(inter['eye_tiles']['L']), 5)
        self.assertEqual(len(inter['eye_tiles']['R']), 5)
        for key in (inter['head'], *inter['mouth_tiles'],
                    *inter['eye_tiles']['L'], *inter['eye_tiles']['R']):
            self.assertTrue((PACK / key).exists(), key)
        hr = inter['boxes']['head_rect']
        mbox = inter['boxes']['mouth_box']
        # рот внутри головы по вертикали и тайл совпадает по размеру с боксом
        self.assertGreaterEqual(mbox[1], hr[1])
        self.assertLessEqual(mbox[3], hr[3])
        from PIL import Image
        with Image.open(PACK / inter['mouth_tiles'][0]) as im:
            self.assertEqual(im.size, (mbox[2] - mbox[0], mbox[3] - mbox[1]))
        with Image.open(PACK / inter['head']) as im:
            self.assertEqual(im.size, (hr[2] - hr[0], hr[3] - hr[1]))

    @unittest.skipUnless((ROOT / 'build' / 'atlas.json').exists(),
                         'build/ не собран (python3 tools/rig.py)')
    def test_frames_reference_same_tiles_as_model_atlas(self):
        """Кадры рта из пакета = тайлы атласа, из которого собран .moc3."""
        import numpy as np
        from PIL import Image
        meta = json.loads((ROOT / 'build' / 'atlas.json').read_text(encoding='utf-8'))
        atlas = Image.open(ROOT / 'build' / 'texture_atlas.png').convert('RGBA')
        for i in range(6):
            rx0, ry0, rx1, ry1 = meta['placed']['mouth_%d' % i]['rect']
            ref = atlas.crop((rx0, ry0, rx1, ry1))
            with Image.open(PACK / ('tiles/mouth_%d.png' % i)) as im:
                self.assertEqual(im.size, ref.size)
                a = np.asarray(im.convert('RGBA'), dtype=np.int16)
                b = np.asarray(ref, dtype=np.int16)
                diff = np.abs(a - b)
                # PNG-8 палитра может дать +/-1 на градиентах, не больше
                self.assertLessEqual(float(diff.max()), 2,
                                     'тайл mouth_%d отличается от атласа' % i)


if __name__ == '__main__':
    unittest.main()

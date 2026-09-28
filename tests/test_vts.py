"""Проверки интерактивного предпросмотра «как в VTube Studio» (папка vts/).

Run: python3 -m unittest discover -s tests -v

Что проверяется:
  - структура vts/pack (манифест ↔ файлы ↔ размеры тайлов ↔ боксы);
  - ТОЧНОЕ совпадение JS-порта деформации (vts/model.js) с Python-моделью:
    head_warp / body_warp и сетки деформеров 6x6 и 5x5;
  - свойства порта: нейтраль = тождество, шов голова/корпус непрерывен,
    ниже cut_y голова не деформируется, сетка и UV в допустимых пределах;
  - параметры предпросмотра совпадают с параметрами собранной модели (cdi3).
"""
import importlib.util
import json
import math
import os
import random
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))

PACK = ROOT / 'vts' / 'pack'
HAS_NODE = shutil.which('node') is not None

from build_model import CFG, body_warp, head_warp  # noqa: E402

MODEL_NAME = CFG['name']          # имя модели: dist/<name>/<name>.moc3


def js_warp_sweep(states) -> dict:
    """Прогон model.js через node по тем же состояниям, что в python-тесте."""
    script = (
        "const Chibi = require(%r);"
        "const fs = require('fs');"
        "const m = JSON.parse(fs.readFileSync(%r, 'utf8'));"
        "Chibi.loadRig(m);"
        "const states = JSON.parse(process.argv[1]);"
        "const out = [];"
        "for (const st of states) {"
        "  const hg = Chibi.headGrid(st), bg = Chibi.bodyGrid(st);"
        "  const pts = [];"
        "  for (const p of [[150,60],[515,420],[875,700],[150,700],[875,1024],[512,900]]) {"
        "    const w = Chibi.warpPoint(p[0], p[1], st, hg, bg);"
        "    pts.push([w[0], w[1]]);"
        "  }"
        "  out.push({head: Array.from(hg), body: Array.from(bg), pts: pts});"
        "}"
        "console.log('===JSON===' + JSON.stringify(out));"
    ) % (str(ROOT / 'vts' / 'model.js'), str(PACK / 'manifest.json'))
    res = subprocess.run(['node', '-e', script, json.dumps(states)],
                         capture_output=True, text=True, timeout=120)
    if res.returncode != 0:
        raise RuntimeError(res.stderr[-800:])
    return json.loads(res.stdout.split('===JSON===')[-1].strip().splitlines()[0])


def py_warp_sweep(states):
    """Тот же расчёт на Python: смещения узлов + итоговая деформация точек."""
    HEAD_RECT = CFG['head_rect']
    BODY_RECT = CFG['body_rect']
    MIR = CFG['canvas_px']
    out = []
    for st in states:
        ax = st.get('ParamAngleX', 0.0)
        head_disp, body_disp = [], []
        for j in range(7):
            for i in range(7):
                px = HEAD_RECT[0] + (HEAD_RECT[2] - HEAD_RECT[0]) * i / 6.0
                py = HEAD_RECT[1] + (HEAD_RECT[3] - HEAD_RECT[1]) * j / 6.0
                wx, wy = head_warp(px, py - st.get('ParamBreath', 0.0) * 3.0, ax,
                                   st.get('ParamAngleY', 0.0), st.get('ParamAngleZ', 0.0))
                head_disp += [wx - px, wy - py]
        for j in range(6):
            for i in range(6):
                px = BODY_RECT[0] + (BODY_RECT[2] - BODY_RECT[0]) * i / 5.0
                py = BODY_RECT[1] + (BODY_RECT[3] - BODY_RECT[1]) * j / 5.0
                ym = MIR - py
                wx, wym = body_warp(px, ym, st.get('ParamBodyAngleX', 0.0),
                                    st.get('ParamBodyAngleY', 0.0),
                                    st.get('ParamBodyAngleZ', 0.0),
                                    st.get('ParamBreath', 0.0))
                wx += 0.30 * ax
                body_disp += [wx - px, py + (ym - wym) - py]
        # Итоговая деформация точки — билинейная интерполяция смещений по
        # сеткам деформеров (так считает и Cubism Core, и vts/model.js),
        # а не подстановка в аналитические head_warp/body_warp.
        pts = []
        for px, py in ((150, 60), (515, 420), (875, 700), (150, 700), (875, 1024), (512, 900)):
            hdx, hdy = sample_grid(head_disp, HEAD_RECT, 6, 6, px, py)
            qx, qy = px + hdx, py + hdy
            bdx, bdy = sample_grid(body_disp, BODY_RECT, 5, 5, qx, qy)
            pts.append([qx + bdx, qy + bdy])
        out.append({'head': head_disp, 'body': body_disp, 'pts': pts})
    return out


def sample_grid(disp, rect, rows, cols, px, py):
    """Билинейная выборка смещения сетки деформера в точке (px, py)."""
    x0, y0, x1, y1 = rect
    u = max(0.0, min(cols, (px - x0) / (x1 - x0) * cols))
    v = max(0.0, min(rows, (py - y0) / (y1 - y0) * rows))
    i = min(int(u), cols - 1)
    j = min(int(v), rows - 1)
    fu, fv = u - i, v - j
    stride = cols + 1
    a = (j * stride + i) * 2
    b = (j * stride + i + 1) * 2
    c = ((j + 1) * stride + i) * 2
    d = ((j + 1) * stride + i + 1) * 2
    dx = (disp[a] * (1 - fu) + disp[b] * fu) * (1 - fv) + \
         (disp[c] * (1 - fu) + disp[d] * fu) * fv
    dy = (disp[a + 1] * (1 - fu) + disp[b + 1] * fu) * (1 - fv) + \
         (disp[c + 1] * (1 - fu) + disp[d + 1] * fu) * fv
    return dx, dy


def py_warp_sweep_full(states):
    """Деформация произвольных точек (сетка деформеров + билинейная выборка)."""
    out = []
    for st in states:
        ref = py_warp_sweep([st])[0]
        pts = []
        for px, py in ((515, 420), (400, 300), (650, 550), (300, 650)):
            hdx, hdy = sample_grid(ref['head'], CFG['head_rect'], 6, 6, px, py)
            qx, qy = px + hdx, py + hdy
            bdx, bdy = sample_grid(ref['body'], CFG['body_rect'], 5, 5, qx, qy)
            pts.append((qx + bdx, qy + bdy))
        out.append(pts)
    return out


def sample_states(n=24, seed=7):
    rnd = random.Random(seed)
    states = [{}]
    for _ in range(n):
        states.append({
            'ParamAngleX': round(rnd.uniform(-30, 30), 3),
            'ParamAngleY': round(rnd.uniform(-30, 30), 3),
            'ParamAngleZ': round(rnd.uniform(-30, 30), 3),
            'ParamBodyAngleX': round(rnd.uniform(-10, 10), 3),
            'ParamBodyAngleY': round(rnd.uniform(-10, 10), 3),
            'ParamBodyAngleZ': round(rnd.uniform(-10, 10), 3),
            'ParamBreath': round(rnd.uniform(0, 1), 3),
        })
    return states


class WarpPortParity(unittest.TestCase):
    """vts/model.js должен деформировать ровно так же, как tools/build_model.py."""

    @unittest.skipUnless(HAS_NODE, 'node недоступен')
    def test_head_and_body_grids_match_python(self):
        states = sample_states()
        js = js_warp_sweep(states)
        py = py_warp_sweep(states)
        self.assertEqual(len(js), len(py))
        for k, (a, b) in enumerate(zip(js, py)):
            self.assertEqual(len(a['head']), len(b['head']))
            for i, (x, y) in enumerate(zip(a['head'], b['head'])):
                self.assertAlmostEqual(x, y, places=6,
                                       msg='head disp #%d[%d]' % (k, i))
            for i, (x, y) in enumerate(zip(a['body'], b['body'])):
                self.assertAlmostEqual(x, y, places=6,
                                       msg='body disp #%d[%d]' % (k, i))
            for i, (p, q) in enumerate(zip(a['pts'], b['pts'])):
                self.assertAlmostEqual(p[0], q[0], places=6, msg='point #%d[%d].x' % (k, i))
                self.assertAlmostEqual(p[1], q[1], places=6, msg='point #%d[%d].y' % (k, i))

    def test_neutral_keyform_is_identity(self):
        """Регрессия: нейтральный ключ деформера головы обязан быть
        тождественным на всём прямоугольнике, иначе модель «защипывает»
        макушку уже в покое (было: cos(a) ограничивался после asin)."""
        x0, y0, x1, y1 = CFG['head_rect']
        for j in range(7):
            for i in range(7):
                px = x0 + (x1 - x0) * i / 6.0
                py = y0 + (y1 - y0) * j / 6.0
                wx, wy = head_warp(px, py, 0.0, 0.0, 0.0)
                self.assertAlmostEqual(wx, px, places=9, msg=(px, py))
                self.assertAlmostEqual(wy, py, places=9, msg=(px, py))


    def test_grid_interpolation_stays_near_analytic_warp(self):
        """Сетка деформера — это интерполяция, а не сама формула: на
        контрольных точках разница должна быть небольшой (десятки пикселей при
        ячейке сетки ~107 px) — если порт «уехал» от модели, она вырастет
        многократно."""
        for st in sample_states(8, seed=11)[1:]:
            py = py_warp_sweep([st])[0]
            pts = py_warp_sweep_full([st])[0]
            for (px, py0), (gx, gy) in zip(
                    ((515, 420), (400, 300), (650, 550), (300, 650)), pts):
                wx, wy = head_warp(px, py0 - st.get('ParamBreath', 0.0) * 3.0,
                                   st.get('ParamAngleX', 0.0),
                                   st.get('ParamAngleY', 0.0),
                                   st.get('ParamAngleZ', 0.0))
                ym = CFG['canvas_px'] - wy
                bx, bym = body_warp(wx, ym, st.get('ParamBodyAngleX', 0.0),
                                    st.get('ParamBodyAngleY', 0.0),
                                    st.get('ParamBodyAngleZ', 0.0),
                                    st.get('ParamBreath', 0.0))
                bx += 0.30 * st.get('ParamAngleX', 0.0)
                self.assertLess(abs(gx - bx), 30.0, (st, px, py0))
                self.assertLess(abs(gy - (wy + (ym - bym))), 30.0, (st, px, py0))

    @unittest.skipUnless(HAS_NODE, 'node недоступен')
    def test_neutral_pose_is_identity(self):
        js = js_warp_sweep([{}])[0]
        for v in js['head'] + js['body']:
            self.assertAlmostEqual(v, 0.0, places=9)
        for p, expect in zip(js['pts'], ((150, 60), (515, 420), (875, 700),
                                         (150, 700), (875, 1024), (512, 900))):
            self.assertAlmostEqual(p[0], expect[0], places=6)
            self.assertAlmostEqual(p[1], expect[1], places=6)


class PackStructure(unittest.TestCase):
    """Собранный пакет предпросмотра консистентен."""

    @classmethod
    def setUpClass(cls):
        if not PACK.exists():
            raise unittest.SkipTest('vts/pack не собран (python3 tools/build_vts_demo.py)')
        cls.manifest = json.loads((PACK / 'manifest.json').read_text(encoding='utf-8'))

    def test_files_exist_and_match_boxes(self):
        from PIL import Image
        files = self.manifest['files']
        for key in (files['head'], files['body'], *files['mouth_tiles'],
                    *files['eye_tiles']['L'], *files['eye_tiles']['R']):
            self.assertTrue((PACK / key).exists(), key)
        head_rect = self.manifest['boxes']['head_rect']
        with Image.open(PACK / files['head']) as im:
            self.assertEqual(im.size, (head_rect[2] - head_rect[0], head_rect[3] - head_rect[1]))
        body_rect = self.manifest['canvas']['body_rect']
        with Image.open(PACK / files['body']) as im:
            self.assertEqual(im.size, (body_rect[2] - body_rect[0], body_rect[3] - body_rect[1]))
        mouth_box = self.manifest['boxes']['mouth_box']
        with Image.open(PACK / files['mouth_tiles'][0]) as im:
            self.assertEqual(im.size, (mouth_box[2] - mouth_box[0], mouth_box[3] - mouth_box[1]))
        for side in ('L', 'R'):
            box = self.manifest['boxes']['eye_boxes'][0 if side == 'L' else 1]
            with Image.open(PACK / files['eye_tiles'][side][0]) as im:
                self.assertEqual(im.size, (box[2] - box[0], box[3] - box[1]))

    def test_face_boxes_inside_head(self):
        hr = self.manifest['boxes']['head_rect']
        for box in [self.manifest['boxes']['mouth_box'], *self.manifest['boxes']['eye_boxes']]:
            self.assertGreaterEqual(box[0], hr[0])
            self.assertGreaterEqual(box[1], hr[1])
            self.assertLessEqual(box[2], hr[2])
            self.assertLessEqual(box[3], hr[3])

    def test_deformers_match_model_cfg(self):
        self.assertEqual(self.manifest['deformers']['head']['rect'], list(CFG['head_rect']))
        self.assertEqual(self.manifest['deformers']['body']['rect'], list(CFG['body_rect']))
        self.assertEqual((self.manifest['deformers']['head']['rows'],
                          self.manifest['deformers']['head']['cols']), (6, 6))
        self.assertEqual((self.manifest['deformers']['body']['rows'],
                          self.manifest['deformers']['body']['cols']), (5, 5))
        rig = self.manifest['rig']
        for key, cfg_key in (('head_fade', 'head_fade'), ('r_yaw', 'r_yaw'),
                             ('r_pitch', 'r_pitch'), ('yaw_scale', 'yaw_scale'),
                             ('pitch_scale', 'pitch_scale'), ('roll_scale', 'roll_scale'),
                             ('body_shift_x', 'body_shift_x'), ('body_shift_y', 'body_shift_y'),
                             ('body_roll_deg', 'body_roll_deg')):
            self.assertAlmostEqual(rig[key], CFG[cfg_key], places=9, msg=key)
        self.assertEqual(rig['head_center'], list(CFG['head_center']))
        self.assertEqual(rig['neck_point'], list(CFG['neck_point']))
        self.assertAlmostEqual(self.manifest['canvas']['cut_y'], CFG['cut_y'], places=9)

    def test_parameters_match_cdi3(self):
        """Предпросмотр крутит ровно те параметры, что есть в модели."""
        name = MODEL_NAME
        cdi = ROOT / 'dist' / name / ('%s.cdi3.json' % name)
        if not cdi.exists():
            self.skipTest('dist/%s не собран' % name)
        model_params = [p['Id'] for p in json.loads(cdi.read_text(encoding='utf-8'))['Parameters']]
        preview = [p['id'] for p in self.manifest['parameters']]
        self.assertEqual(sorted(preview), sorted(model_params))
        for p in self.manifest['parameters']:
            self.assertLess(p['min'], p['max'])
            self.assertGreaterEqual(p['default'], p['min'])
            self.assertLessEqual(p['default'], p['max'])
            self.assertTrue(p['label'] and p['group'])

    def test_vts_groups_match_model3(self):
        name = MODEL_NAME
        model3 = ROOT / 'dist' / name / ('%s.model3.json' % name)
        if not model3.exists():
            self.skipTest('dist/%s не собран' % name)
        groups = json.loads(model3.read_text(encoding='utf-8'))['Groups']
        self.assertEqual(
            sorted((g['name'], tuple(g['ids'])) for g in self.manifest['vts_groups']),
            sorted((g['Name'], tuple(g['Ids'])) for g in groups))

    @unittest.skipUnless((ROOT / 'build' / 'atlas.json').exists(),
                         'build/ не собран (python3 tools/rig.py)')
    def test_tiles_are_the_models_atlas_tiles(self):
        """Тайлы предпросмотра = тайлы атласа, из которого собран .moc3."""
        import numpy as np
        from PIL import Image
        meta = json.loads((ROOT / 'build' / 'atlas.json').read_text(encoding='utf-8'))
        atlas = Image.open(ROOT / 'build' / 'texture_atlas.png').convert('RGBA')
        files = self.manifest['files']
        checks = [('mouth_%d' % i, f) for i, f in enumerate(files['mouth_tiles'])]
        for side in (0, 1):
            for name, f in zip(('neutral', 'half', 'blink', 'happy', 'squint'),
                               files['eye_tiles']['L' if side == 0 else 'R']):
                checks.append(('eye_%d_%s' % (side, name), f))
        for key, rel in checks:
            rx0, ry0, rx1, ry1 = meta['placed'][key]['rect']
            ref = np.asarray(atlas.crop((rx0, ry0, rx1, ry1)), dtype=np.int16)
            with Image.open(PACK / rel) as im:
                self.assertEqual(im.size, (rx1 - rx0, ry1 - ry0), key)
                got = np.asarray(im.convert('RGBA'), dtype=np.int16)
            self.assertLessEqual(int(np.abs(got - ref).max()), 2,
                                 'тайл %s отличается от атласа' % key)


class MeshProperties(unittest.TestCase):
    """Свойства сетки для warp-рендера (vts/model.js)."""

    @unittest.skipUnless(HAS_NODE and PACK.exists(), 'node или vts/pack недоступны')
    def test_grid_and_mesh_are_sane(self):
        script = (
            "const Chibi = require(%r);"
            "const fs = require('fs');"
            "const m = JSON.parse(fs.readFileSync(%r, 'utf8'));"
            "Chibi.loadRig(m);"
            "const step = m.browser_mesh.step;"
            "const hg = Chibi.buildGrid(m.canvas.head_rect, step);"
            "const bg = Chibi.buildGrid(m.canvas.body_rect, step);"
            "const cov = Chibi.dilateCover(new Uint8Array(hg.count).fill(1), hg, 2);"
            "const st = {ParamAngleX: 20, ParamAngleY: -12, ParamAngleZ: 6,"
            "            ParamBodyAngleX: 4, ParamBodyAngleY: -3, ParamBreath: 0.5};"
            "const w = Chibi.warpGrid(hg, st, Chibi.headGrid(st), Chibi.bodyGrid(st));"
            "const tri = Chibi.buildTriangles(hg, i => cov[i]);"
            "const wire = Chibi.buildWire(hg, i => cov[i]);"
            "let uvOK = true, maxIdx = 0, finite = true;"
            "for (let i = 0; i < hg.count; i++) {"
            "  for (const v of [hg.uv[2*i], hg.uv[2*i+1]]) if (!(v >= 0 && v <= 1)) uvOK = false;"
            "  if (!isFinite(w[2*i]) || !isFinite(w[2*i+1])) finite = false;"
            "}"
            "for (const i of tri) if (i > maxIdx) maxIdx = i;"
            "console.log('===JSON===' + JSON.stringify({"
            "  rows: hg.rows, cols: hg.cols, count: hg.count, bodyRows: bg.rows,"
            "  uvOK: uvOK, finite: finite, tri: tri.length, wire: wire.length,"
            "  maxIdx: maxIdx, verts: hg.count}));"
        ) % (str(ROOT / 'vts' / 'model.js'), str(PACK / 'manifest.json'))
        res = subprocess.run(['node', '-e', script], capture_output=True, text=True, timeout=120)
        if res.returncode != 0:
            raise RuntimeError(res.stderr[-800:])
        out = json.loads(res.stdout.split('===JSON===')[-1].strip().splitlines()[0])
        self.assertTrue(out['uvOK'], 'UV сетки должны лежать в [0,1]')
        self.assertTrue(out['finite'], 'деформация не должна давать NaN/Inf')
        self.assertEqual(out['tri'] % 3, 0)
        self.assertEqual(out['wire'] % 2, 0)
        self.assertGreater(out['tri'], 1000)
        self.assertLess(out['maxIdx'], out['verts'])
        self.assertLessEqual(out['verts'], 65535, 'индексы должны влезать в Uint16')

    @unittest.skipUnless(HAS_NODE and PACK.exists(), 'node или vts/pack недоступны')
    def test_head_and_body_meshes_stitch_at_cut(self):
        """На шве (y = cut_y) голова и корпус деформируются одинаково —
        иначе по шее модели пошёл бы разрыв."""
        script = (
            "const Chibi = require(%r);"
            "const fs = require('fs');"
            "const m = JSON.parse(fs.readFileSync(%r, 'utf8'));"
            "Chibi.loadRig(m);"
            "const st = {ParamAngleX: 18, ParamAngleY: 10, ParamAngleZ: -7,"
            "            ParamBodyAngleX: 5, ParamBodyAngleY: -4, ParamBodyAngleZ: 6,"
            "            ParamBreath: 0.7};"
            "const hg = Chibi.headGrid(st), bg = Chibi.bodyGrid(st);"
            "const xs = [150, 300, 512, 700, 875];"
            "const out = xs.map(x => {"
            "  const a = Chibi.warpPoint(x, m.canvas.cut_y - 0.0001, st, hg, bg);"
            "  const b = Chibi.warpPoint(x, m.canvas.cut_y + 0.0001, st, hg, bg);"
            "  return [Math.abs(a[0]-b[0]), Math.abs(a[1]-b[1])];"
            "});"
            "console.log('===JSON===' + JSON.stringify(out));"
        ) % (str(ROOT / 'vts' / 'model.js'), str(PACK / 'manifest.json'))
        res = subprocess.run(['node', '-e', script], capture_output=True, text=True, timeout=120)
        if res.returncode != 0:
            raise RuntimeError(res.stderr[-800:])
        gaps = json.loads(res.stdout.split('===JSON===')[-1].strip().splitlines()[0])
        for dx, dy in gaps:
            self.assertLess(dx, 0.05, 'разрыв по X на шве головы и корпуса')
            self.assertLess(dy, 0.05, 'разрыв по Y на шве головы и корпуса')

    def test_head_warp_vanishes_below_cut(self):
        """Ниже cut_y деформер головы не работает: этим держится шея."""
        for ax, ay, az in ((0, 0, 0), (30, 30, 30), (-25, 12, -18)):
            for x in (150, 512, 875):
                wx, wy = head_warp(x, CFG['cut_y'], ax, ay, az)
                self.assertAlmostEqual(wx, x, places=9, msg=(ax, ay, az, x))
                self.assertAlmostEqual(wy, CFG['cut_y'], places=9, msg=(ax, ay, az, x))

    def test_body_warp_is_bounded(self):
        """Корпус смещается в пределах разумного — модель не «уезжает»."""
        limit = 60.0
        for bx in (-10, 0, 10):
            for by in (-10, 0, 10):
                for bz in (-10, 0, 10):
                    for py in (700, 850, 1024):
                        ym = CFG['canvas_px'] - py
                        wx, wym = body_warp(512.0, ym, bx, by, bz, 1.0)
                        self.assertLess(abs(wx - 512.0), limit, (bx, by, bz, py))
                        self.assertLess(abs(wym - ym), limit, (bx, by, bz, py))


if __name__ == '__main__':
    unittest.main()

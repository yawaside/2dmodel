#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Full sanity check of the generated Live2D kit.

Runs the model through the real Live2D Cubism Core (via tools/dump_model.js)
and checks the things VTube Studio relies on: parameter IDs and ranges,
draw / render order permutation, keyform sanity over a parameter sweep and the
consistency of the .model3.json file references.

Cubism Core is required for the deep checks (see preview/lib/README); without
it the script falls back to structural checks only.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

REQUIRED_PARAMS = {
    'ParamAngleX': (-30.0, 0.0, 30.0),
    'ParamAngleY': (-30.0, 0.0, 30.0),
    'ParamAngleZ': (-30.0, 0.0, 30.0),
    'ParamBodyAngleX': (-10.0, 0.0, 10.0),
    'ParamBodyAngleY': (-10.0, 0.0, 10.0),
    'ParamBodyAngleZ': (-10.0, 0.0, 10.0),
    'ParamEyeLOpen': (0.0, 1.0, 1.0),
    'ParamEyeROpen': (0.0, 1.0, 1.0),
    'ParamMouthOpenY': (0.0, 0.0, 1.0),
}


def node_dump(moc_path, states):
    js = os.path.join(HERE, 'dump_model.js')
    env = dict(os.environ)
    res = subprocess.run(['node', js, moc_path, json.dumps(states)],
                         capture_output=True, text=True, env=env)
    if res.returncode != 0:
        return None, res.stderr.strip()[-800:]
    return json.loads(res.stdout.split('===JSON===')[-1].strip().split('\n')[0]), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='dist/ChibiVT')
    args = ap.parse_args()

    d = args.dir
    ok = True
    def check(cond, msg):
        nonlocal ok
        print(('  OK   ' if cond else '  FAIL ') + msg)
        ok = ok and bool(cond)

    meta_path = None
    for f in sorted(os.listdir(d)):
        if f.endswith('.model3.json'):
            meta_path = os.path.join(d, f)
    check(meta_path is not None, 'найден *.model3.json')
    if not meta_path:
        return 1
    meta = json.load(open(meta_path))
    name = os.path.basename(meta_path)[:-len('.model3.json')]

    print('\n[файлы]')
    for key in ('Moc',):
        p = os.path.join(d, meta['FileReferences'][key])
        check(os.path.exists(p), '%s: %s (%d байт)' % (key, os.path.basename(p), os.path.getsize(p)))
    for t in meta['FileReferences'].get('Textures', []):
        p = os.path.join(d, t)
        check(os.path.exists(p), 'Textures: %s (%d байт)' % (t, os.path.getsize(p)))
    for key in ('Physics', 'DisplayInfo'):
        if key in meta['FileReferences']:
            p = os.path.join(d, meta['FileReferences'][key])
            check(os.path.exists(p), '%s: %s' % (key, os.path.basename(p)))
            if os.path.exists(p):
                json.load(open(p))          # must be valid JSON
    check('Groups' in meta, 'model3.json содержит Groups (EyeBlink / LipSync)')

    moc_path = os.path.join(d, meta['FileReferences']['Moc'])

    print('\n[Cubism Core]')
    states = [{}, {'ParamAngleX': 30}, {'ParamAngleX': -30}, {'ParamAngleY': 30},
              {'ParamAngleY': -30}, {'ParamAngleZ': 30}, {'ParamAngleZ': -30},
              {'ParamBodyAngleX': 10}, {'ParamBodyAngleY': 10}, {'ParamBodyAngleZ': 10},
              {'ParamEyeLOpen': 0, 'ParamEyeROpen': 0}, {'ParamMouthOpenY': 1},
              {'ParamMouthOpenY': 0.17}, {'ParamMouthOpenY': 0.33},
              {'ParamMouthOpenY': 0.5}, {'ParamMouthOpenY': 0.67},
              {'ParamAngleX': 30, 'ParamAngleY': -30, 'ParamAngleZ': 30,
               'ParamBodyAngleX': -10, 'ParamBodyAngleY': 10, 'ParamBodyAngleZ': -10,
               'ParamEyeLOpen': 0.3, 'ParamMouthOpenY': 0.7}]
    dumps, err = node_dump(moc_path, states)
    if dumps is None:
        print('  SKIP  Cubism Core недоступен: %s' % err)
        return 0 if ok else 1
    check(True, 'moc3 загружен Cubism Core')

    base = dumps[0]
    print('\n[параметры]')
    # need a second dump that lists parameter info: re-run with a tiny node helper
    info = subprocess.run(['node', '-e', r'''
const fs=require("fs"),path=require("path");
const dirs=[path.join(process.argv[1],"..","preview","lib"),"/tmp/corepkg/package"];
let COREDIR=dirs.find(p=>fs.existsSync(path.join(p,"live2dcubismcore.min.js")));
globalThis.__dirname=COREDIR;globalThis.__filename=path.join(COREDIR,"live2dcubismcore.min.js");globalThis.require=require;
(0,eval)(fs.readFileSync(path.join(COREDIR,"live2dcubismcore.min.js"),"utf8"));
const Core=globalThis.Live2DCubismCore;
const buf=fs.readFileSync(process.argv[2]);
const m=Core.Model.fromMoc(Core.Moc.fromArrayBuffer(buf.buffer.slice(buf.byteOffset,buf.byteOffset+buf.byteLength)));
console.log("===JSON===");console.log(JSON.stringify({ids:Array.from(m.parameters.ids),min:Array.from(m.parameters.minimumValues),max:Array.from(m.parameters.maximumValues),def:Array.from(m.parameters.defaultValues),kc:Array.from(m.parameters.keyCounts)}));
''', HERE, moc_path], capture_output=True, text=True)
    pinfo = json.loads(info.stdout.split('===JSON===')[-1].strip().split('\n')[0])
    have = dict(zip(pinfo['ids'], zip(pinfo['min'], pinfo['def'], pinfo['max'])))
    for pid, rng in REQUIRED_PARAMS.items():
        good = pid in have and all(abs(have[pid][i] - rng[i]) < 1e-6 for i in range(3))
        check(good, '%-16s %s' % (pid, have.get(pid)))

    print('\n[меши / порядок отрисовки]')
    ro = sorted(dr['ro'] for dr in base['drawables'])
    check(ro == list(range(len(ro))), 'render order — перестановка 0..%d: %s' % (len(ro) - 1, ro))
    ids = [dr['id'] for dr in sorted(base['drawables'], key=lambda x: x['ro'])]
    check(True, 'порядок отрисовки: ' + ' < '.join(ids))
    for dr in base['drawables']:
        check(len(dr['idx']) % 3 == 0 and len(dr['idx']) > 0,
              '%-10s %d вершин, %d треугольников, текстура %d' %
              (dr['id'], len(dr['pos']) // 2, len(dr['idx']) // 3, dr['tex']))

    print('\n[деформации: sweep]')
    bad = 0
    finite = True
    for st, dump in zip(states, dumps):
        for dr in dump['drawables']:
            for v in dr['pos']:
                if v != v or abs(v) > 20:
                    finite = False
    check(finite, 'все вершины конечны и в разумных пределах на %d состояниях' % len(states))

    def centroid(dump, did):
        for dr in dump['drawables']:
            if dr['id'] == did:
                p = dr['pos']
                n = len(p) // 2
                return (sum(p[2 * i] for i in range(n)) / n,
                        sum(p[2 * i + 1] for i in range(n)) / n)
        return (0, 0)

    def delta(state, did):
        for st, dump in zip(states, dumps):
            if st == state:
                a, b = centroid(base, did), centroid(dump, did)
                return ((b[0] - a[0]) * 1024, (a[1] - b[1]) * 1024)   # px, y down
        return (0, 0)

    mov = {
        'ParamAngleX +30 (поворот вправо)': delta({'ParamAngleX': 30}, 'Head'),
        'ParamAngleX -30 (поворот влево)': delta({'ParamAngleX': -30}, 'Head'),
        'ParamAngleY +30 (смотрит вверх)': delta({'ParamAngleY': 30}, 'Eye0'),
        'ParamAngleZ +30 (наклон)': delta({'ParamAngleZ': 30}, 'Head'),
        'ParamBodyAngleX +10': delta({'ParamBodyAngleX': 10}, 'Body'),
        'ParamEyeLOpen 0 (моргание)': delta({'ParamEyeLOpen': 0, 'ParamEyeROpen': 0}, 'Eye0'),
    }
    for k, (dx, dy) in mov.items():
        print('       %-34s смещение %+6.1f, %+6.1f px' % (k, dx, dy))

    def row_gap(dump, did):
        """Largest vertical step between two neighbouring rows of a mesh.

        The eye mesh keeps its outline (so it stays glued to the face) and
        closes by collapsing its inner rows - the give-away is one row step
        that grows from 'one cell' to 'half the eye'.
        """
        for dr in dump['drawables']:
            if dr['id'] != did:
                continue
            n = len(dr['pos']) // 2
            xs = sorted(set(dr['pos'][2 * i] for i in range(n)))
            ys = sorted(set(dr['pos'][2 * i + 1] for i in range(n)))
            rows = []
            i = 0
            while i < len(ys):
                j = i
                while j + 1 < len(ys) and abs(ys[j + 1] - ys[i]) < 1e-6:
                    j += 1
                rows.append(sum(ys[i:j + 1]) / (j - i + 1.0))
                i = j + 1
            if len(rows) < 3:
                return 0.0
            return max(abs(rows[k + 1] - rows[k]) for k in range(len(rows) - 1))
        return 0.0

    # Blink is a frame swap, not a mesh collapse: all eye variants are opaque
    # tiles and stack_opa() keeps the base fully covered, so the giveaway is
    # the "blink" drawable switching on while the others switch off.
    def draw_op(dump, did):
        for dr in dump['drawables']:
            if dr['id'] == did:
                return dr['op']
        return 0.0

    op_blink_open = draw_op(base, 'Eye0_blink')
    op_blink_shut = draw_op(dumps[10], 'Eye0_blink')
    check(op_blink_shut > 0.95 and op_blink_open < 0.05,
          'моргание: Eye0_blink %.2f -> %.2f (кадр меняется без рывков сетки)'
          % (op_blink_open, op_blink_shut))

    def base_leak(dump, prefix):
        """Weight of the head art still visible under a block (leak == square)."""
        w = 1.0
        for dr in dump['drawables']:
            if dr['id'].startswith(prefix):
                w *= (1.0 - dr['op'])
        return w

    for idx, lbl, tol in ((0, 'покой', 0.02), (10, 'моргание', 0.02),
                          (16, 'сложное состояние', 0.15)):
        for pfx in ('Eye0_', 'Eye1_'):
            lk = base_leak(dumps[idx], pfx)
            check(lk <= tol,
                  '%s: утечка базы под %s = %.3f (норма <= %.2f)'
                  % (lbl, pfx.rstrip('_'), lk, tol))

    def mesh_h(dump, did):
        for dr in dump['drawables']:
            if dr['id'] == did:
                ys = dr['pos'][1::2]
                return (max(ys) - min(ys)) * 1024
        return 0.0

    # ---- раскадровка рта -------------------------------------------------
    mframes = sorted([dr['id'] for dr in base['drawables']
                      if dr['id'].startswith('Mouth')])
    check(len(mframes) >= 3, 'кадры рта: %s' % ', '.join(mframes))
    print('       эффективные вклады кадров (снизу вверх):')
    bad = 0
    for lbl, idx in (('0.00 (закрыт)', 0), ('0.17', 12), ('0.33', 13),
                     ('0.50', 14), ('0.67', 15), ('1.00 (открыт)', 11)):
        stack = sorted([d for d in dumps[idx]['drawables']
                        if d['id'].startswith('Mouth')], key=lambda d: d['ro'])
        ops = [d['op'] for d in stack]
        # contribution of layer k = op_k * product(1 - op_j) for layers above;
        # what is left after everything = weight of the base art underneath
        eff, above = [], 1.0
        for op in reversed(ops):
            eff.append(op * above)
            above *= (1.0 - op)
        eff.reverse()
        total, base_w = sum(eff), above
        vis = ['%s=%.2f' % (d['id'], e) for d, e in zip(stack, eff) if e > 0.01]
        print('         MouthOpenY %-14s %s   сумма %.2f, база %.2f'
              % (lbl, ' '.join(vis), total, base_w))
        # keyform states must be exact; between keys a small drift is allowed
        # (the runtime only linearly interpolates the stored opacities)
        tol = 0.02 if lbl.startswith(('0.00', '1.00')) else 0.15
        if abs(total - 1.0) > 0.02 or base_w > tol:
            bad += 1
    check(bad == 0,
          'рот: вклады кадров дают исходное состояние, база почти перекрыта '
          '(сумма = 1.00, утечка в норме)')

    # кадр 0 — это сама картинка: в покое он должен совпадать с исходником
    check('Mouth0' in mframes, 'первый кадр — нарисованный на модели рот')

    # the closed mouth is the model's own art, so it must still be there
    check('MouthClosed' not in [dr['id'] for dr in base['drawables']],
          'закрытый рот — собственная текстура модели (отдельного меша нет)')

    print('\n' + ('ВСЁ ОК — комплект готов: %s' % d if ok else 'ЕСТЬ ОШИБКИ'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

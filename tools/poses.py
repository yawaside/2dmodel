#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render a contact sheet of poses so the rig can be eyeballed without a GPU.

    python tools/poses.py --out build/poses_sheet.png
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from render import rasterise          # noqa: E402

FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'

POSES = [
    ('нейтраль', {}),
    ('поворот влево\nAngleX -30', {'ParamAngleX': -30}),
    ('поворот вправо\nAngleX +30', {'ParamAngleX': 30}),
    ('смотрит вверх\nAngleY +30', {'ParamAngleY': 30}),
    ('смотрит вниз\nAngleY -30', {'ParamAngleY': -30}),
    ('наклон влево\nAngleZ -30', {'ParamAngleZ': -30}),
    ('наклон вправо\nAngleZ +30', {'ParamAngleZ': 30}),
    ('глаза полузакрыты\nEyeLOpen .5', {'ParamEyeLOpen': 0.5, 'ParamEyeROpen': 0.5}),
    ('глаза закрыты\nEyeLOpen 0', {'ParamEyeLOpen': 0, 'ParamEyeROpen': 0}),
    ('рот: кадр 1\nMouthOpenY .33', {'ParamMouthOpenY': 0.33}),
    ('рот: кадр 2\nMouthOpenY .67', {'ParamMouthOpenY': 0.67}),
    ('рот: кадр 3\nMouthOpenY 1', {'ParamMouthOpenY': 1}),
    ('тело влево\nBodyAngleX -10', {'ParamBodyAngleX': -10}),
    ('тело вправо\nBodyAngleX +10', {'ParamBodyAngleX': 10}),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='dist/ChibiVT/ChibiVT.moc3')
    ap.add_argument('--textures', nargs='+',
                    default=['dist/ChibiVT/textures/texture_00.png'])
    ap.add_argument('--tile', type=int, default=256)
    ap.add_argument('--cols', type=int, default=4)
    ap.add_argument('--out', default='build/poses_sheet.png')
    args = ap.parse_args()

    states = json.dumps([p[1] for p in POSES])
    js = os.path.join(HERE, 'dump_model.js')
    res = subprocess.run(['node', js, args.model, states],
                         capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stderr[-2000:])
        sys.exit(res.returncode)
    dumps = json.loads(res.stdout.split('===JSON===')[-1].strip().split('\n')[0])

    cols = args.cols
    rows = (len(POSES) + cols - 1) // cols
    T = args.tile
    head = 46
    sheet = Image.new('RGB', (cols * T, rows * (T + head)), (24, 24, 28))
    dr = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(FONT, 15)
    for i, (dump, (title, _)) in enumerate(zip(dumps, POSES)):
        img = rasterise(dump, args.textures, T, True)
        rgba = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
        r, c = divmod(i, cols)
        x, y = c * T, r * (T + head)
        sheet.paste(rgba, (x, y + head), rgba)
        dr.text((x + 6, y + 6), title.replace('\n', '  '), fill=(235, 235, 240),
                font=font)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    sheet.save(args.out)
    print('wrote %s  (%dx%d, %d poses)' % (args.out, sheet.size[0], sheet.size[1], len(POSES)))


if __name__ == '__main__':
    main()

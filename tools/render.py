#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Headless rasteriser for the generated model (used to eyeball the rig
without a GPU).  Reads the JSON dump produced by tools/dump_model.js."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
RAMP = ' .:-=+*#%@'


def rasterise(dump, tex_paths, size=512, flip_v=True):
    tex = [np.array(Image.open(p).convert('RGBA')).astype(np.float32) / 255.0 for p in tex_paths]
    H = W = size
    canvas = np.zeros((H, W, 4), np.float32)
    cw, ch = dump['canvas']['w'], dump['canvas']['h']
    ox, oy = dump['canvas']['ox'], dump['canvas']['oy']
    ppu = dump['canvas']['ppu']
    scale = size / max(cw, ch) / ppu * ppu      # model units -> px (square fit)
    scale = size / max(cw, ch)

    def to_screen(x, y):
        # model space is y-up, canvas origin (ox, oy) in pixels from top-left
        px = (x * ppu + ox) * (size / cw)
        py = (oy - y * ppu) * (size / ch)
        return px, py

    for dr in sorted(dump['drawables'], key=lambda d: d['ro']):
        if not dr['visible'] or dr['op'] <= 0.002:
            continue
        pos = np.array(dr['pos'], np.float32).reshape(-1, 2)
        uv = np.array(dr['uv'], np.float32).reshape(-1, 2)
        idx = np.array(dr['idx'], np.int32).reshape(-1, 3)
        img = tex[dr['tex'] if dr['tex'] < len(tex) else 0]
        th, tw = img.shape[:2]
        sx, sy = to_screen(pos[:, 0], pos[:, 1])
        sx = sx * (size / cw) / (size / cw)          # keep in output pixels
        sp = np.stack([sx, sy], 1)
        tu = uv[:, 0] * tw
        tv = (1.0 - uv[:, 1] if flip_v else uv[:, 1]) * th
        tp = np.stack([tu, tv], 1)
        for tri in idx:
            a, b, c = sp[tri[0]], sp[tri[1]], sp[tri[2]]
            ta, tb, tc = tp[tri[0]], tp[tri[1]], tp[tri[2]]
            # affine map screen -> texture
            M = np.array([[b[0] - a[0], c[0] - a[0]], [b[1] - a[1], c[1] - a[1]]], np.float32)
            if abs(np.linalg.det(M)) < 1e-9:
                continue
            Minv = np.linalg.inv(M)
            x0, x1 = int(max(0, np.floor(min(a[0], b[0], c[0])))), int(min(W, np.ceil(max(a[0], b[0], c[0]))) + 1)
            y0, y1 = int(max(0, np.floor(min(a[1], b[1], c[1])))), int(min(H, np.ceil(max(a[1], b[1], c[1]))) + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            gy, gx = np.mgrid[y0:y1, x0:x1]
            dx = gx - a[0]
            dy = gy - a[1]
            u = Minv[0, 0] * dx + Minv[0, 1] * dy
            v = Minv[1, 0] * dx + Minv[1, 1] * dy
            inside = (u >= -1e-4) & (v >= -1e-4) & (u + v <= 1.0 + 1e-4)
            if not inside.any():
                continue
            su = (ta[0] + u * (tb[0] - ta[0]) + v * (tc[0] - ta[0])).astype(np.int32)
            sv = (ta[1] + u * (tb[1] - ta[1]) + v * (tc[1] - ta[1])).astype(np.int32)
            np.clip(su, 0, tw - 1, out=su)
            np.clip(sv, 0, th - 1, out=sv)
            src = img[sv.ravel(), su.ravel()].reshape(su.shape[0], su.shape[1], 4)
            alpha = src[..., 3:4] * dr['op']
            dst = canvas[y0:y1, x0:x1]
            m = inside[..., None]
            if dst.shape[:2] != src.shape[:2]:
                h = min(dst.shape[0], src.shape[0]); w = min(dst.shape[1], src.shape[1])
                src, alpha, dst, m = src[:h, :w], alpha[:h, :w], dst[:h, :w], m[:h, :w]
            # correct straight-alpha "over": the canvas keeps colour and alpha
            # separate instead of darkening everything that is not opaque.
            oa = alpha + dst[..., 3:4] * (1.0 - alpha)
            oc = (src[..., :3] * alpha + dst[..., :3] * dst[..., 3:4] * (1.0 - alpha)
                  ) / np.maximum(oa, 1e-6)
            out = np.where(m, np.concatenate([oc, oa], 2), dst)
            canvas[y0:y1, x0:x1][:dst.shape[0], :dst.shape[1]] = out
    return canvas


def ascii_view(rgba, cols=64, rows=34):
    a = rgba[..., 3]
    lum = 0.299 * rgba[..., 0] + 0.587 * rgba[..., 1] + 0.114 * rgba[..., 2]
    H, W = a.shape
    out = []
    for r in range(rows):
        line = ''
        for c in range(cols):
            y0, y1 = int(H * r / rows), max(int(H * r / rows) + 1, int(H * (r + 1) / rows))
            x0, x1 = int(W * c / cols), max(int(W * c / cols) + 1, int(W * (c + 1) / cols))
            av = a[y0:y1, x0:x1].mean()
            if av < 0.25:
                line += ' '
            else:
                lv = lum[y0:y1, x0:x1].mean()
                line += RAMP[min(len(RAMP) - 1, int((1.0 - lv) * (len(RAMP) - 1) + 0.5))]
        out.append(line)
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='dist/ChibiVT/ChibiVT.moc3')
    ap.add_argument('--textures', nargs='+', default=['dist/ChibiVT/textures/texture_00.png'])
    ap.add_argument('--states', default='[{}]')
    ap.add_argument('--size', type=int, default=512)
    ap.add_argument('--no-flip-v', dest='flip_v', action='store_false',
                    help='use uv v as-is (Core already returns GL-style v)')
    ap.add_argument('--out', default='build/preview_%d.png')
    ap.add_argument('--ascii', action='store_true')
    args = ap.parse_args()

    js = os.path.join(HERE, 'dump_model.js')
    res = subprocess.run(['node', js, args.model, args.states], capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stderr[-2000:])
        sys.exit(res.returncode)
    marker = '===JSON==='
    dumps = json.loads(res.stdout.split(marker)[-1].strip().split('\n')[0])
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    for i, d in enumerate(dumps):
        img = rasterise(d, args.textures, args.size, args.flip_v)
        path = args.out % i if '%' in args.out else args.out
        Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(path)
        tag = json.dumps(d['state'], ensure_ascii=False)
        print('wrote %s   %s' % (path, tag))
        if args.ascii:
            print(ascii_view(img))
            print()


if __name__ == '__main__':
    main()

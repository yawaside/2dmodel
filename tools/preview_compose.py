#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quick and dishonest-free visual sanity check that does NOT require the
proprietary Live2D Cubism Core.

It reuses the exact same mouth_blend()/eye_blend() weighting functions from
tools/build_model.py (so the crossfade timing you see here is the same
timing baked into the real .moc3 keyforms) and composites the drawn PSD
frames straight from the already-built atlas (build/texture_atlas.png).

It does NOT show the 3D head-turn mesh warp or the physics (spring/damping,
breathing) - those only exist inside the moc3 and need either VTube Studio
or the real Cubism Core to evaluate. This script is only here to let you
eyeball the lip-sync / blink blending quality quickly in the sandbox.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from build_model import mouth_blend, eye_blend  # noqa: E402

ATLAS_PNG = os.path.join(ROOT, "build", "texture_atlas.png")
ATLAS_JSON = os.path.join(ROOT, "build", "atlas.json")


def crop(atlas, rect):
    x0, y0, x1, y1 = rect
    return atlas[y0:y1, x0:x1].astype(np.float32) / 255.0


def over(base, patch, box):
    """Straight-alpha 'over' composite of `patch` onto `base` at `box`=(x0,y0,x1,y1)."""
    x0, y0, x1, y1 = box
    h, w = y1 - y0, x1 - x0
    ph, pw = patch.shape[:2]
    h, w = min(h, ph), min(w, pw)
    dst = base[y0:y0 + h, x0:x0 + w]
    src = patch[:h, :w]
    a = src[..., 3:4]
    dst[..., :3] = src[..., :3] * a + dst[..., :3] * (1 - a)
    dst[..., 3:4] = a + dst[..., 3:4] * (1 - a)


def blend_frames(frames, weights):
    """weighted average of RGBA frames, matching the moc3 keyform math."""
    out = np.zeros_like(frames[0])
    for f, w in zip(frames, weights):
        if w > 0:
            out += f * w
    return out


def make_frame(atlas, meta, mouth_open, mouth_form, eye_open, eye_smile):
    base_rect = meta["base_rect"]
    base = crop(atlas, base_rect).copy()

    mouth_frames = [crop(atlas, meta["placed"]["mouth_%d" % i]["rect"]) for i in range(meta["mouth_frame_count"])]
    mw = mouth_blend(mouth_open, mouth_form)
    mouth_patch = blend_frames(mouth_frames, mw)
    over(base, mouth_patch, meta["mouth_box"])

    variants = meta["eyes"][0]["variants"]
    for eye_i, eye in enumerate(meta["eyes"]):
        e_frames = [crop(atlas, meta["placed"]["eye_%d_%s" % (eye_i, v)]["rect"]) for v in variants]
        ew = eye_blend(eye_open, eye_smile)
        patch = blend_frames(e_frames, ew)
        box = eye["rig_box"]
        over(base, patch, box)

    return (np.clip(base, 0, 1) * 255).astype(np.uint8)


BG = np.array([30, 32, 38, 255], np.float32)  # neutral dark backdrop, like the real preview page


def flatten(rgba):
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    out = rgba[..., :3].astype(np.float32) * a + BG[:3] * (1 - a)
    return np.concatenate([out, np.full_like(a, 255.0)], axis=-1).astype(np.uint8)


def main():
    atlas = np.array(Image.open(ATLAS_PNG).convert("RGBA"))
    meta = json.load(open(ATLAS_JSON))

    os.makedirs(os.path.join(ROOT, "build", "preview"), exist_ok=True)

    # 1) lip-sync sweep (mouth opens through the same monotonic chain the
    #    real model uses), with a blink happening mid-way.
    frames = []
    N = 48
    for i in range(N):
        t = i / (N - 1)
        # simple talking envelope: two "syllables"
        o = max(0.0, np.sin(t * np.pi * 3.1) ** 2)
        blink = 1.0 - np.clip(1.0 - abs((t - 0.5) * 18), 0.0, 1.0)  # quick blink near t=0.5
        eye_open = 1.0 - blink
        frames.append(make_frame(atlas, meta, o, 0.0, eye_open, 0.0))

    frames = [flatten(f) for f in frames]
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(os.path.join(ROOT, "build", "preview", "lipsync_demo.gif"),
                 save_all=True, append_images=imgs[1:], duration=42, loop=0, disposal=2)

    # 2) static grid of key expressions for a quick side-by-side look
    states = [
        ("closed_neutral", 0.0, 0.0, 1.0, 0.0),
        ("half_open", 0.5, 0.0, 1.0, 0.0),
        ("A_open", 1.0, 0.0, 1.0, 0.0),
        ("smile", 0.3, 1.0, 1.0, 1.0),
        ("smirk", 0.2, -1.0, 1.0, 0.0),
        ("blink", 0.0, 0.0, 0.0, 0.0),
    ]
    tiles = [flatten(make_frame(atlas, meta, o, f, eo, es)) for _, o, f, eo, es in states]
    tile_h, tile_w = tiles[0].shape[:2]
    cols = 3
    rows = (len(tiles) + cols - 1) // cols
    grid = np.zeros((rows * tile_h, cols * tile_w, 4), np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        grid[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w] = tile
    Image.fromarray(grid).save(os.path.join(ROOT, "build", "preview", "expressions_grid.png"))

    print("wrote build/preview/lipsync_demo.gif")
    print("wrote build/preview/expressions_grid.png")


if __name__ == "__main__":
    main()

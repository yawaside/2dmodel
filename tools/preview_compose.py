#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quick and dishonest-free visual sanity check that does NOT require the
proprietary Live2D Cubism Core.

It reuses the exact same eye_blend() weighting function and mouth_deform()
mesh-deformation math from tools/build_model.py (so what you see here is the
same thing baked into the real .moc3 keyforms) and renders straight from the
already-built atlas (build/texture_atlas.png). The mouth is a single texture
whose mesh is warped per-pixel with a small numpy resampler that mirrors the
real (mesh-based) deformation closely enough for a sanity check; the eyes
still cross-fade drawn frames exactly like the shipped model does.

It does NOT show the 3D head-turn mesh warp or the physics (spring/damping,
breathing) - those only exist inside the moc3 and need either VTube Studio
or the real Cubism Core to evaluate. This script is only here to let you
eyeball the lip-sync / blink quality quickly in the sandbox.
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
from build_model import eye_blend, mouth_deform  # noqa: E402

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


_MOUTH_TILE_CACHE = {}


def render_mouth_patch(atlas, meta, mouth_open, mouth_form):
    """Resample the single mouth tile through mouth_deform()'s per-vertex
    math with a tiny numpy inverse-mapping (per-column LUT for Y, then one
    more for X) - a cheap stand-in for the real triangle-mesh renderer that
    is enough to eyeball whether the deformation itself looks right.

    The real mesh renderer simply does not draw where the (retracted) mesh
    no longer reaches - the face art underneath shows through, and it is
    pixel-identical there (see mouth_deform()'s docstring). This resampler
    mirrors that: destination rows/cols outside the deformed mesh's actual
    footprint are left fully transparent instead of being clamped/repeated,
    so over() lets the correct base face art show through instead of a
    smeared copy of the mesh's edge pixel.
    """
    box = tuple(meta["mouth_box"])
    art_bbox = tuple(meta["mouth_art_bbox"])
    pivot = tuple(meta["mouth_center"])
    mx0, my0, mx1, my1 = box
    key = box
    if key not in _MOUTH_TILE_CACHE:
        _MOUTH_TILE_CACHE[key] = crop(atlas, meta["placed"]["mouth_0"]["rect"])
    tile = _MOUTH_TILE_CACHE[key]
    H, W = tile.shape[:2]

    ys = np.arange(my0, my1) + 0.5
    xs = np.arange(mx0, mx1) + 0.5
    PX, PY = np.meshgrid(xs, ys)
    final_x = np.zeros_like(PX)
    final_y = np.zeros_like(PY)
    # mouth_deform() is a scalar function (matches the real per-vertex
    # pos_fn) - vectorize it by hand over the small mouth tile.
    for r in range(H):
        for c in range(W):
            final_x[r, c], final_y[r, c] = mouth_deform(
                PX[r, c], PY[r, c], mouth_open, mouth_form, box, art_bbox, pivot)

    out = np.zeros_like(tile)
    py_rows = ys
    for c in range(W):
        g = final_y[:, c]
        order = np.argsort(g, kind="stable")
        g_sorted, src_sorted = g[order], np.arange(H)[order]
        src_py_idx = np.interp(py_rows, g_sorted, src_sorted, left=-1, right=-1)
        covered = (py_rows >= g_sorted[0]) & (py_rows <= g_sorted[-1])
        idx = np.clip(np.round(src_py_idx).astype(int), 0, H - 1)
        out[:, c] = tile[idx, c]
        out[~covered, c, 3] = 0.0

    gx = final_x[0]
    order = np.argsort(gx, kind="stable")
    gx_sorted, srcx_sorted = gx[order], np.arange(W)[order]
    src_px_idx = np.interp(xs, gx_sorted, srcx_sorted, left=-1, right=-1)
    covered_x = (xs >= gx_sorted[0]) & (xs <= gx_sorted[-1])
    idxx = np.clip(np.round(src_px_idx).astype(int), 0, W - 1)
    out = out[:, idxx]
    out[:, ~covered_x, 3] = 0.0
    return out


def make_frame(atlas, meta, mouth_open, mouth_form, eye_open, eye_smile):
    base_rect = meta["base_rect"]
    base = crop(atlas, base_rect).copy()

    mouth_patch = render_mouth_patch(atlas, meta, mouth_open, mouth_form)
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

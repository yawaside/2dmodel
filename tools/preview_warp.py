#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline rasteriser of the HEAD/BODY mesh warp (pseudo-3D turn) that does
NOT need the moc3 file or Cubism Core - it calls the exact same head_warp()/
body_warp() math tools/build_model.py bakes into the model, and rasterises
the resulting mesh directly from the flat source art.

Used only to eyeball / tune the "volume" of the head turn quickly.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from build_model import CFG, head_warp, body_warp, grid_mesh, build_shade_texture, clamp  # noqa: E402

CUTOUT = os.path.join(ROOT, "build", "texture_atlas.png")  # has the same base head/body art at (0,0)-(1024,1024)


def rasterise_tris(canvas, src, verts, tris, get_xy, opacity=1.0):
    """get_xy(px,py) -> warped (x,y) in the SAME pixel space as canvas/src."""
    if opacity <= 0.002:
        return
    H, W = canvas.shape[:2]
    sh, sw = src.shape[:2]
    pos = np.array([get_xy(x, y) for (x, y) in verts], np.float32)
    src_pos = np.array(verts, np.float32)
    for tri in tris:
        a, b, c = tri
        pa, pb, pc = pos[a], pos[b], pos[c]
        ta, tb, tc = src_pos[a], src_pos[b], src_pos[c]
        x0 = int(max(0, np.floor(min(pa[0], pb[0], pc[0]))))
        x1 = int(min(W, np.ceil(max(pa[0], pb[0], pc[0])) + 1))
        y0 = int(max(0, np.floor(min(pa[1], pb[1], pc[1]))))
        y1 = int(min(H, np.ceil(max(pa[1], pb[1], pc[1])) + 1))
        if x1 <= x0 or y1 <= y0:
            continue
        M = np.array([[pb[0] - pa[0], pc[0] - pa[0]], [pb[1] - pa[1], pc[1] - pa[1]]], np.float32)
        det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
        if abs(det) < 1e-9:
            continue
        Minv = np.linalg.inv(M)
        gy, gx = np.mgrid[y0:y1, x0:x1]
        dx = gx + 0.5 - pa[0]
        dy = gy + 0.5 - pa[1]
        u = Minv[0, 0] * dx + Minv[0, 1] * dy
        v = Minv[1, 0] * dx + Minv[1, 1] * dy
        inside = (u >= -1e-4) & (v >= -1e-4) & (u + v <= 1.0 + 1e-4)
        if not inside.any():
            continue
        su = np.clip((ta[0] + u * (tb[0] - ta[0]) + v * (tc[0] - ta[0])).astype(np.int32), 0, sw - 1)
        sv = np.clip((ta[1] + u * (tb[1] - ta[1]) + v * (tc[1] - ta[1])).astype(np.int32), 0, sh - 1)
        srcpix = src[sv.ravel(), su.ravel()].reshape(su.shape[0], su.shape[1], 4).astype(np.float32) / 255.0
        alpha = srcpix[..., 3:4] * opacity
        dst = canvas[y0:y1, x0:x1]
        m = inside[..., None]
        oa = alpha + dst[..., 3:4] * (1 - alpha)
        oc = (srcpix[..., :3] * alpha + dst[..., :3] * dst[..., 3:4] * (1 - alpha)) / np.maximum(oa, 1e-6)
        out = np.where(m, np.concatenate([oc, oa], 2), dst)
        canvas[y0:y1, x0:x1] = out


def build_shade_overlays(alpha_mask, canvas_size, head_rect, cfg=CFG):
    """Build 4 full-canvas RGBA images (same pixel frame as the source art)
    carrying the yaw/pitch shade textures pasted at head_rect, normal and
    mirrored - mirrors exactly what build_model.py bakes into the moc3."""
    hx0, hy0, hx1, hy1 = [int(round(v)) for v in head_rect]
    yaw_tex = build_shade_texture(alpha_mask, head_rect, "x", cfg)
    pitch_tex = build_shade_texture(alpha_mask, head_rect, "y", cfg)

    def full(tex, flip):
        img = np.zeros((canvas_size, canvas_size, 4), np.uint8)
        t = tex[:, ::-1] if flip == "x" else (tex[::-1, :] if flip == "y" else tex)
        img[hy0:hy1, hx0:hx1] = t
        return img

    return {
        "yaw_pos": full(yaw_tex, None),
        "yaw_neg": full(yaw_tex, "x"),
        "pitch_pos": full(pitch_tex, "y"),
        "pitch_neg": full(pitch_tex, None),
    }


def render_pose(alpha_mask, src, head_verts, head_tris, body_verts, body_tris, ax, ay, az, bx, by, bz, breath=0.0,
                 shade=None):
    H, W = src.shape[:2]
    canvas = np.zeros((H, W, 4), np.float32)

    def body_xy(px, py):
        ym = W - py
        wx, wym = body_warp(px, ym, bx, by, bz, breath=breath)
        return (px + (wx - px), py - (wym - ym))

    def head_xy(px, py):
        py_bob = py - breath * 3.0
        wx, wy = head_warp(px, py_bob, ax, ay, az)
        bx2, by2 = body_xy(wx, wy)
        return (bx2, by2)

    rasterise_tris(canvas, src, body_verts, body_tris, body_xy)
    rasterise_tris(canvas, src, head_verts, head_tris, head_xy)

    if shade is not None:
        yaw_max = CFG["shade_yaw_max_deg"]
        pitch_max = CFG["shade_pitch_max_deg"]
        rasterise_tris(canvas, shade["yaw_pos"], head_verts, head_tris, head_xy,
                       opacity=clamp(ax, 0.0, yaw_max) / yaw_max)
        rasterise_tris(canvas, shade["yaw_neg"], head_verts, head_tris, head_xy,
                       opacity=clamp(-ax, 0.0, yaw_max) / yaw_max)
        rasterise_tris(canvas, shade["pitch_pos"], head_verts, head_tris, head_xy,
                       opacity=clamp(ay, 0.0, pitch_max) / pitch_max)
        rasterise_tris(canvas, shade["pitch_neg"], head_verts, head_tris, head_xy,
                       opacity=clamp(-ay, 0.0, pitch_max) / pitch_max)
    return canvas


def main():
    atlas = np.array(Image.open(CUTOUT).convert("RGBA"))
    base_w = CFG["canvas_px"]
    src = atlas[: int(base_w), : int(base_w)]
    alpha = src[..., 3] > 128

    cut = CFG["cut_y"]
    Hh, Ww = alpha.shape
    head_verts, head_tris = grid_mesh(alpha, (0, 0, Ww, cut), CFG["mesh_step"], CFG["mesh_dilate"])
    body_verts, body_tris = grid_mesh(alpha, (0, cut, Ww, Hh - 1), CFG["mesh_step"], CFG["mesh_dilate"])
    shade = build_shade_overlays(alpha, Ww, CFG["head_rect"])

    BG = np.array([30, 32, 38, 255], np.float32)

    def flat(canvas):
        a = canvas[..., 3:4]
        rgb = canvas[..., :3] * 255.0 * a + BG[:3] * (1 - a)
        return np.clip(np.concatenate([rgb, np.full_like(a, 255.0)], -1), 0, 255).astype(np.uint8)

    os.makedirs(os.path.join(ROOT, "build", "preview"), exist_ok=True)

    # sweep: yaw left -> right -> pitch up -> down -> roll, with body catch-up lag simulated simply
    N = 60
    frames = []
    for i in range(N):
        t = i / (N - 1)
        ax = 30.0 * np.sin(t * 2 * np.pi)
        ay = 18.0 * np.sin(t * 2 * np.pi * 2 + 1.0)
        az = 10.0 * np.sin(t * 2 * np.pi + 0.5)
        bx = ax * 0.33
        by = ay * 0.25
        bz = az * 0.4
        breath = 0.5 + 0.5 * np.sin(t * 2 * np.pi * 3)
        canvas = render_pose(alpha, src, head_verts, head_tris, body_verts, body_tris,
                              ax, ay, az, bx, by, bz, breath, shade=shade)
        frames.append(flat(canvas))

    imgs = [Image.fromarray(f) for f in frames]
    out_gif = os.path.join(ROOT, "build", "preview", "head_turn_demo.gif")
    imgs[0].save(out_gif, save_all=True, append_images=imgs[1:], duration=45, loop=0, disposal=2)
    print("wrote", out_gif)

    # static grid of extreme poses
    poses = [
        ("center", 0, 0, 0), ("yaw_L", -30, 0, 0), ("yaw_R", 30, 0, 0),
        ("pitch_up", 0, 20, 0), ("pitch_down", 0, -20, 0), ("roll", 0, 0, 20),
    ]
    tiles = []
    for _, ax, ay, az in poses:
        canvas = render_pose(alpha, src, head_verts, head_tris, body_verts, body_tris,
                              ax, ay, az, ax * 0.33, ay * 0.25, az * 0.4, 0.0, shade=shade)
        tiles.append(flat(canvas))
    th, tw = tiles[0].shape[:2]
    cols = 3
    rows = (len(tiles) + cols - 1) // cols
    grid = np.zeros((rows * th, cols * tw, 4), np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        grid[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = tile
    out_png = os.path.join(ROOT, "build", "preview", "head_turn_grid.png")
    Image.fromarray(grid).save(out_png)
    print("wrote", out_png)


if __name__ == "__main__":
    main()

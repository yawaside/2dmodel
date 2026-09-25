#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Texture preparation for the Live2D model: eye inpaint, atlas layout.

Everything is authored in the pixel space of the source art (1024x1024).
The produced atlas is 2048x1024: left half = character, right half = extra art.
"""

from __future__ import annotations

import json
import math
import os
from collections import deque

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #

def dilate(mask: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return mask
    img = Image.fromarray((mask * 255).astype(np.uint8))
    img = img.filter(ImageFilter.MaxFilter(2 * k + 1))
    return np.array(img) > 127


def erode(mask: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return mask
    img = Image.fromarray((mask * 255).astype(np.uint8))
    img = img.filter(ImageFilter.MinFilter(2 * k + 1))
    return np.array(img) > 127


def connected(mask: np.ndarray, seed) -> np.ndarray:
    """4-connected flood fill of `mask` starting at seed (y, x)."""
    H, W = mask.shape
    sy, sx = int(seed[0]), int(seed[1])
    seen = np.zeros_like(mask)
    if not (0 <= sy < H and 0 <= sx < W) or not mask[sy, sx]:
        return seen
    q = deque([(sy, sx)])
    seen[sy, sx] = True
    while q:
        y, x = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True
                q.append((ny, nx))
    return seen


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill holes (background not touching the image border) of a mask."""
    H, W = mask.shape
    bg = ~mask
    outside = np.zeros_like(bg)
    q = deque()
    for x in range(W):
        for y in (0, H - 1):
            if bg[y, x] and not outside[y, x]:
                outside[y, x] = True
                q.append((y, x))
    for y in range(H):
        for x in (0, W - 1):
            if bg[y, x] and not outside[y, x]:
                outside[y, x] = True
                q.append((y, x))
    while q:
        y, x = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and bg[ny, nx] and not outside[ny, nx]:
                outside[ny, nx] = True
                q.append((ny, nx))
    return mask | (~outside)


def soft_alpha(mask: np.ndarray, feather: int) -> np.ndarray:
    """0..1 alpha: 1 inside the mask, fading to 0 over `feather` px outside."""
    core = erode(mask, feather) if feather else mask
    a = core.astype(np.float32)
    grown = dilate(mask, feather) if feather else mask
    dist_out = np.zeros_like(a)
    cur = core
    for i in range(1, feather + 1):
        nxt = dilate(cur, 1)
        ring = nxt & ~cur
        dist_out[ring] = 1.0 - i / (feather + 1.0)
        cur = nxt
    a[~core & grown] = dist_out[~core & grown]
    a[~grown] = 0.0
    return a


# --------------------------------------------------------------------------- #
# eye detection / inpainting
# --------------------------------------------------------------------------- #

def eye_mask(lum: np.ndarray, alpha: np.ndarray, cx: float, cy: float,
             thresh: float = 150.0, box: int = 95) -> np.ndarray:
    x0, x1 = int(cx - box), int(cx + box)
    y0, y1 = int(cy - box), int(cy + box)
    sub = ((lum < thresh) & alpha)[y0:y1, x0:x1]
    comp = connected(sub, (cy - y0, cx - x0))
    full = np.zeros(lum.shape, bool)
    full[y0:y1, x0:x1] = comp
    return fill_holes(full)


def mouth_mask(lum: np.ndarray, alpha: np.ndarray, cx: float, cy: float,
               thresh: float = 115.0, half_w: float = 85.0,
               half_h: float = 34.0) -> np.ndarray:
    """The dark, connected region of the mouth inside a window around (cx, cy).

    The search window keeps the flood fill from leaking into the hair / the
    shadow under the chin, and only the largest component is kept.
    """
    H, W = lum.shape
    x0, x1 = int(max(cx - half_w, 0)), int(min(cx + half_w, W))
    y0, y1 = int(max(cy - half_h, 0)), int(min(cy + half_h, H))
    win = np.zeros(lum.shape, bool)
    win[y0:y1, x0:x1] = True
    dark = (lum < thresh) & alpha & win
    best, seen = None, np.zeros(lum.shape, bool)
    for y, x in zip(*np.where(dark)):
        if seen[y, x]:
            continue
        comp = connected(dark, (y, x))
        seen |= comp
        if best is None or int(comp.sum()) > int(best.sum()):
            best = comp
    if best is None:
        return fill_holes(dark)
    return fill_holes(best)


def inpaint_region(rgba: np.ndarray, mask: np.ndarray, ring: int = 12,
                   feather: int = 4) -> np.ndarray:
    """Fill `mask` with the median colour of the surrounding skin ring."""
    out = rgba.copy()
    solid = dilate(mask, 1)
    ring_mask = dilate(mask, ring) & ~dilate(mask, 3)
    ring_mask &= rgba[..., 3] > 128
    if ring_mask.sum() > 32:
        cols = rgba[..., :3][ring_mask]
        col = np.median(cols, axis=0)
    else:
        col = np.array([250.0, 215.0, 190.0])
    alpha_in = soft_alpha(solid, feather)
    # paint the flat colour, then blur so the patch melts into the skin shading
    tmp = rgba[..., :3].clip(0, 255).astype(np.uint8)
    tmp[solid] = np.clip(col, 0, 255).astype(np.uint8)
    blurred = np.array(Image.fromarray(tmp).filter(ImageFilter.GaussianBlur(5))).astype(np.float32)
    a = alpha_in[..., None]
    out[..., :3] = rgba[..., :3] * (1 - a) + blurred * a
    out[..., 3] = np.maximum(out[..., 3], alpha_in * 255.0)
    return out


# --------------------------------------------------------------------------- #
# mouth synthesis
# --------------------------------------------------------------------------- #

def make_open_mouth(size, lip_rgb, open_w, open_h):
    """Procedural 'open mouth' art: dark interior + tongue + lip outline.

    `size` is the patch the mesh covers; the visible mouth itself is
    `open_w` x `open_h` and stays inside that patch with a margin.
    """
    w, h = size
    img = np.zeros((h, w, 4), np.float32)
    cx, cy = w / 2.0, h / 2.0
    rx, ry = open_w / 2.0, open_h / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u = (xx - cx) / rx
    v = (yy - cy) / ry
    # the upper lip is flatter than the lower one -> squash the top half
    vv = np.where(v < 0.0, v / 0.72, v)
    e = u ** 2 + vv ** 2                                   # 1 = edge
    # interior: darker at the top, warmer towards the throat
    t = np.clip((yy - (cy - 0.72 * ry)) / (1.72 * ry), 0, 1)[..., None]
    top = np.array([58.0, 22.0, 28.0])
    bot = np.array([126.0, 48.0, 56.0])
    col = top[None, None, :] * (1 - t) + bot[None, None, :] * t
    # tongue
    trx, try_ = rx * 0.60, ry * 0.42
    tcy = cy + ry * 0.44
    te = ((xx - cx) / trx) ** 2 + ((yy - tcy) / try_) ** 2
    tongue = np.clip((1.12 - te) / 0.30, 0, 1)[..., None]
    col = col * (1 - tongue) + np.array([198.0, 106.0, 114.0])[None, None, :] * tongue
    # soft alpha, slightly wider than the dark interior so nothing is clipped
    a = np.clip((1.10 - e) / 0.18, 0, 1) * (e < 1.28)
    # lip outline ring
    lip = np.array(lip_rgb, np.float32)
    outline = 1.0 - np.clip(np.abs(np.sqrt(np.maximum(e, 0.0)) - 1.0) / 0.075, 0.0, 1.0)
    outline = outline[..., None]
    col = col * (1 - outline) + lip[None, None, :] * outline
    a = np.maximum(a, outline[..., 0] * (e < 1.5))
    img[..., :3] = col
    img[..., 3] = np.clip(a, 0.0, 1.0) * 255.0
    return img.astype(np.uint8)


# --------------------------------------------------------------------------- #
# atlas building
# --------------------------------------------------------------------------- #

def _ensure(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)


def blit(dst: np.ndarray, tile: np.ndarray, x: int, y: int) -> None:
    """Straight-alpha 'over' of `tile` (float RGBA, 0..255) onto `dst`.

    PIL's ``Image.paste(img, box, img)`` multiplies the colour by the mask in
    addition to the alpha, which darkens every feathered edge of a patch - so
    the compositing is done by hand here.
    """
    h, w = tile.shape[:2]
    y1, x1 = min(y + h, dst.shape[0]), min(x + w, dst.shape[1])
    hh, ww = y1 - y, x1 - x
    if hh <= 0 or ww <= 0:
        return
    t = tile[:hh, :ww]
    reg = dst[y:y1, x:x1]
    a = t[..., 3:4] / 255.0
    da = reg[..., 3:4] / 255.0
    oa = a + da * (1.0 - a)
    oc = (t[..., :3] * a + reg[..., :3] * da * (1.0 - a)) / np.maximum(oa, 1e-6)
    dst[y:y1, x:x1] = np.concatenate([oc, oa * 255.0], axis=2)


def build_texture(src_png: str, out_png: str, meta_json: str, geom: dict,
                  cfg: dict) -> dict:
    src = Image.open(src_png).convert("RGBA")
    W, H = src.size
    rgba = np.array(src).astype(np.float32)
    alpha = rgba[..., 3] > 128
    lum = 0.299 * rgba[..., 0] + 0.587 * rgba[..., 1] + 0.114 * rgba[..., 2]

    # NOTE: nothing is in-painted and nothing is copied into separate tiles
    # any more.  The eye / mouth meshes sample the original art in place, so
    # the model is pixel-identical to the source picture at rest.

    # ----- eyes ----------------------------------------------------------- #
    # `box`     - tight bbox of the eye (used for the closing centre)
    # `rig_box` - the same box grown by a margin of real skin: that skin is
    #             what the eyelid is made of when the eye closes.
    eyes = geom["eyes"]
    eye_boxes = []
    for i, e in enumerate(eyes):
        m = eye_mask(lum, alpha, e["cx"], e["cy"],
                     thresh=cfg["eye_dark_thresh"], box=cfg["eye_search_box"])
        ys, xs = np.where(m)
        pad = cfg["eye_pad"]
        box = (max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad),
               min(W, int(xs.max()) + pad + 1), min(H, int(ys.max()) + pad + 1))
        mx, mt, mb = cfg["eye_margin_x"], cfg["eye_margin_top"], cfg["eye_margin_bottom"]
        rig = (max(0, int(xs.min()) - mx), max(0, int(ys.min()) - mt),
               min(W, int(xs.max()) + 1 + mx), min(H, int(ys.max()) + 1 + mb))
        eye_boxes.append({"mask": m, "box": box, "rig_box": rig,
                          "center": (e["cx"], e["cy"]),
                          "cy": float(ys.min() + ys.max()) / 2.0,
                          "mask_y0": float(ys.min()), "mask_y1": float(ys.max()),
                          "size": (box[2] - box[0], box[3] - box[1])})

    # ----- mouth ---------------------------------------------------------- #
    # The closed mouth stays in the base art; the mesh with the open mouth
    # grows out of the mouth's own centre and hides it as it opens.
    mx, my = float(geom["mouth"][0]), float(geom["mouth"][1])
    mmask = mouth_mask(lum, alpha, mx, my,
                       thresh=cfg["mouth_dark_thresh"],
                       half_w=cfg["mouth_search_w"],
                       half_h=cfg["mouth_search_h"])
    if int(mmask.sum()) > 16:
        lip_rgb = rgba[..., :3][mmask].mean(axis=0)
        mys, mxs = np.where(mmask)
        mcx = float(mxs.min() + mxs.max()) / 2.0
        mcy = float(mys.min() + mys.max()) / 2.0
        mw = int(mxs.max() - mxs.min() + 1)
        mh = int(mys.max() - mys.min() + 1)
    else:
        lip_rgb = np.array([60.0, 25.0, 25.0])
        mcx, mcy, mw, mh = mx, my, 90, 24
    pw = mw + 2 * cfg["mouth_pad_x"]
    ph = max(mh + 2 * cfg["mouth_pad_y"],
             cfg["mouth_open_h"] + 2 * cfg["mouth_open_margin"])
    px0 = int(round(mcx - pw / 2.0))
    py0 = int(round(mcy - ph / 2.0))
    mouth_box = (px0, py0, px0 + pw, py0 + ph)
    mouth_bbox = (int(mxs.min()), int(mys.min()), int(mxs.max()) + 1, int(mys.max()) + 1)
    # the open mouth is at least as wide as the closed one, otherwise the
    # corners of the closed mouth would peek out next to it
    open_w = min(pw - 2 * cfg["mouth_open_margin"],
                 mw * cfg["mouth_open_w_scale"])
    open_h = cfg["mouth_open_h"]
    open_mouth = make_open_mouth((pw, ph), lip_rgb, open_w, open_h)

    # ----- compose atlas --------------------------------------------------- #
    # Left half: the untouched original art.  Right half: the only piece of
    # art that cannot come from the picture - the inside of the open mouth.
    AW = cfg["atlas_width"]
    atlas = np.zeros((H, AW, 4), np.float32)
    blit(atlas, rgba, 0, 0)

    cur_x, cur_y = W + 24, 24
    tile = open_mouth.astype(np.float32)
    h, w = tile.shape[:2]
    blit(atlas, tile, cur_x, cur_y)
    placed = {"mouth_open": dict(rect=(cur_x, cur_y, cur_x + w, cur_y + h),
                                 size=(w, h))}

    _ensure(out_png)
    Image.fromarray(atlas.astype(np.uint8)).save(out_png)

    meta = dict(atlas_size=(AW, H), base_rect=(0, 0, W, H),
                eyes=[{k: v for k, v in eb.items() if k != "mask"}
                      for eb in eye_boxes],
                mouth_box=list(mouth_box), mouth_bbox=list(mouth_bbox),
                mouth_center=(mcx, mcy), mouth_size=(mw, mh),
                mouth_open_size=(open_w, open_h),
                placed=placed, lip_rgb=list(map(float, lip_rgb)))
    _ensure(meta_json)
    with open(meta_json, "w") as f:
        json.dump(meta, f, indent=1)
    return meta


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="cutout.png")
    ap.add_argument("--geom", default="geom.json")
    ap.add_argument("--out", default="build/texture_atlas.png")
    ap.add_argument("--meta", default="build/atlas.json")
    args = ap.parse_args()
    cfg = dict(eye_dark_thresh=150.0, eye_search_box=95, eye_pad=8,
               eye_margin_x=12, eye_margin_top=6, eye_margin_bottom=20,
               mouth_dark_thresh=115.0, mouth_search_w=85, mouth_search_h=34,
               mouth_pad_x=8, mouth_pad_y=10,
               mouth_open_w_scale=0.95, mouth_open_h=62, mouth_open_margin=6,
               atlas_width=2048)
    meta = build_texture(args.src, args.out, args.meta, json.load(open(args.geom)), cfg)
    print(json.dumps(meta["placed"], indent=1))
    print("mouth box", meta["mouth_box"], "bbox", meta["mouth_bbox"])
    for i, e in enumerate(meta["eyes"]):
        print("eye %d box %s rig %s cy %.1f" % (i, e["box"], e["rig_box"], e["cy"]))


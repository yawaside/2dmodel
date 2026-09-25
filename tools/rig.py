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
# key frames ("раскадровка") of the drawn mouth
# --------------------------------------------------------------------------- #

def mouth_frames(crop: np.ndarray, mmask: np.ndarray, drops, lip: int = 5):
    """One key frame per drop value: *the mouth that is drawn on the model
    opens*.

    Nothing is pasted over the picture.  The drawn upper lip stays where it
    is, the drawn lower lip slides down by ``d`` pixels, the gap in between is
    filled with the mouth's own colour (the ink of the drawing, deepened
    towards the throat and warmed up as the mouth opens wider) and everything
    below the lip is left exactly as it is.  So the frames are the character's
    own mouth at different degrees of opening.
    """
    H, W = crop.shape[:2]
    cols = []
    for x in range(W):
        ys = np.where(mmask[:, x])[0]
        if len(ys):
            cols.append((x, int(ys.min()), int(ys.max()) + 1))
    if not cols:
        return [crop.copy() for _ in drops]
    ink = crop[..., :3][mmask].mean(axis=0)
    frames = []
    for d in drops:
        f = crop.copy()
        dd = int(round(d))
        open_t = min(1.0, dd / 30.0)
        top_c = ink * 0.50                       # throat: darker
        bot_c = ink * 1.30 + np.array([70.0, 34.0, 40.0]) * open_t
        for (x, yt, yb) in cols:
            a0 = min(H, yt + lip)
            a1 = min(H, yb + dd - lip)
            if a1 > a0:
                t = np.linspace(0.0, 1.0, a1 - a0)[:, None]
                f[a0:a1, x, :3] = top_c[None, :] * (1 - t) + bot_c[None, :] * t
                f[a0:a1, x, 3] = 255.0
            b0 = yb + dd - lip
            b1 = min(H, yb + dd)
            if b1 > b0 and b0 >= 0:
                s0 = max(0, yb - lip)
                f[b0:b1, x] = crop[s0:s0 + (b1 - b0), x]
        frames.append(f)
    return frames


# --------------------------------------------------------------------------- #
# atlas building
# --------------------------------------------------------------------------- #

def load_override(path: str, size):
    """Optional hand-drawn replacement for a generated key frame."""
    if not os.path.exists(path):
        return None
    im = Image.open(path).convert("RGBA")
    if im.size != size:
        im = im.resize(size, Image.LANCZOS)
    return np.array(im).astype(np.float32)


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


def crop_nontransparent(arr):
    """Crop RGBA float array to tight bbox around non-transparent pixels."""
    alpha = arr[..., 3] > 10
    if alpha.sum() < 1:
        return arr, (0, 0)
    ys, xs = np.where(alpha)
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    return arr[y0:y1, x0:x1].copy(), (x0, y0)

def build_texture(src_png: str, out_png: str, meta_json: str, geom: dict,
                  cfg: dict) -> dict:
    """Build the expression atlas.

    Stage-1 fixes (eyes / mouth):
      * the base art is NOT inpainted anymore - a flat patch under the face
        used to leak through the blended stack and read as a flickering
        square around the eyes and the mouth;
      * every variant tile is composited over the neutral crop, so all tiles
        are opaque: the stack never reveals whatever lies underneath;
      * boxes are grown to the full hand-drawn art (brows, lips) with a
        margin, so no stroke is ever cut by the mesh edge, and the eye band
        stops exactly where the mouth band starts (the open mouth's upper
        lip used to be covered by the eye tiles);
      * tile edges are replicated a few pixels into the atlas gaps, so
        bilinear filtering in the engine cannot sample the transparent
        gap and draw a dark seam around the blocks.
    """
    src = Image.open(src_png).convert("RGBA")
    W, H = src.size
    rgba = np.array(src).astype(np.float32)
    alpha = rgba[..., 3] > 128
    lum = 0.299 * rgba[..., 0] + 0.587 * rgba[..., 1] + 0.114 * rgba[..., 2]

    box_margin = int(cfg.get("box_margin", 8))
    tile_pad = int(cfg.get("tile_pad", 4))
    col_gap = int(cfg.get("column_gap", 24))

    # ----- layer helpers --------------------------------------------------- #
    def load_layer(fname: str):
        arr = np.array(Image.open(os.path.join("psd_layers", fname))
                       .convert("RGBA")).astype(np.float32)
        return clean_layer(arr)

    # -- corner smudge cleanup ------------------------------------------- #
    def _components(mask):
        H_, W_ = mask.shape
        seen = np.zeros_like(mask)
        out = []
        for y, x in zip(*np.where(mask)):
            if seen[y, x]:
                continue
            q = deque([(y, x)])
            seen[y, x] = True
            comp = []
            while q:
                cy, cx = q.popleft()
                comp.append((cy, cx))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < H_ and 0 <= nx < W_ and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            out.append(comp)
        return out

    def clean_layer(layer):
        """Erase the export-artifact diagonal smudges ("quote marks") every
        PSD expression layer carries in the top corners of its own art bbox.

        A line-guard mask (solid dark ink of the drawn outline + 5px) is
        excluded from EVERY phase: without it a smudge that touches the
        mouth outline merges with it and the erase eats thousands of
        outline pixels - which showed up as a pixelated mouth."""
        ys, xs = np.where(layer[..., 3] > 10)
        if not len(xs):
            return layer
        bx0, by0 = int(xs.min()), int(ys.min())
        bx1, by1 = int(xs.max()) + 1, int(ys.max()) + 1
        zones = [(max(0, bx0 - 10), max(0, by0 - 10), min(W, bx0 + 64), min(H, by0 + 64)),
                 (max(0, bx1 - 64), max(0, by0 - 10), min(W, bx1 + 10), min(H, by0 + 64))]

        l_lum = (0.299 * layer[..., 0] + 0.587 * layer[..., 1]
                 + 0.114 * layer[..., 2])
        ink_core = (layer[..., 3] > 200) & (l_lum < 110)
        guard = np.array(Image.fromarray((ink_core * 255).astype(np.uint8))
                         .filter(ImageFilter.MaxFilter(11))) > 127  # +/-5px

        kill = np.zeros(layer.shape[:2], bool)
        for (x0, y0, x1, y1) in zones:
            b = rgba[y0:y1, x0:x1]
            l = layer[y0:y1, x0:x1]
            a = l[..., 3] > 20
            d = np.abs(l[..., :3] - b[..., :3]).max(axis=2)
            lum = 0.299 * l[..., 0] + 0.587 * l[..., 1] + 0.114 * l[..., 2]
            # phase 1: compact smudge blobs, never the line art
            cand = a & (d > 8) & (lum > 40) & (lum < 215) & ~guard[y0:y1, x0:x1]
            for comp in _components(cand):
                n = len(comp)
                cys = [c[0] for c in comp]
                cxs = [c[1] for c in comp]
                bw = max(cxs) - min(cxs) + 1
                bh = max(cys) - min(cys) + 1
                if 40 <= n <= 900 and max(bw, bh) <= 50 and min(bw, bh) >= 6:
                    for cy, cx in comp:
                        kill[y0 + cy, x0 + cx] = True
        if not kill.any():
            return layer

        # phase 2: light airbrush halo within 6px of a confirmed stroke
        near = np.array(Image.fromarray((kill * 255).astype(np.uint8))
                        .filter(ImageFilter.MaxFilter(13))) > 127
        # phase 3: the dark tail of the stroke within 8px, again off the line
        near2 = np.array(Image.fromarray((kill * 255).astype(np.uint8))
                         .filter(ImageFilter.MaxFilter(17))) > 127
        for (x0, y0, x1, y1) in zones:
            b = rgba[y0:y1, x0:x1]
            l = layer[y0:y1, x0:x1]
            a = l[..., 3] > 15
            d = np.abs(l[..., :3] - b[..., :3]).max(axis=2)
            lum = 0.299 * l[..., 0] + 0.587 * l[..., 1] + 0.114 * l[..., 2]
            g = guard[y0:y1, x0:x1]
            halo = (a & (d > 4) & (lum > 150) & (lum < 234)
                    & near[y0:y1, x0:x1] & ~g)
            tail = (a & (d > 8) & (lum > 40) & (lum < 150)
                    & near2[y0:y1, x0:x1] & ~g)
            kill[y0:y1, x0:x1] |= halo | tail

        out = layer.copy()
        out[..., 3][kill] = 0.0
        # repair: light pixels around the cut get the BASE colour, so the
        # composite equals the base exactly - no feather ramps can survive
        near3 = np.array(Image.fromarray((kill * 255).astype(np.uint8))
                         .filter(ImageFilter.MaxFilter(21))) > 127
        repair = (near3 & (layer[..., 3] > 15) & (l_lum > 160)
                  & (l_lum < 232) & ~guard)
        out[..., 0][repair] = rgba[..., 0][repair]
        out[..., 1][repair] = rgba[..., 1][repair]
        out[..., 2][repair] = rgba[..., 2][repair]
        return out

    def art_bbox(arr) -> tuple | None:
        a = arr[..., 3] > 10
        if not a.any():
            return None
        ys, xs = np.where(a)
        return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)

    def union(bbs):
        bs = [b for b in bbs if b]
        if not bs:
            return None
        return (min(b[0] for b in bs), min(b[1] for b in bs),
                max(b[2] for b in bs), max(b[3] for b in bs))

    def clamp_box(b):
        return (max(0, b[0]), max(0, b[1]), min(W, b[2]), min(H, b[3]))

    def make_tile(box, layer):
        """Neutral crop of the base art with `layer` composited on top.

        Result is opaque wherever the face is opaque, so stacked tiles fully
        cover the head beneath them.
        """
        x0, y0, x1, y1 = box
        neutral = rgba[y0:y1, x0:x1].copy()
        if layer is None:
            return neutral
        lay = layer[y0:y1, x0:x1]
        a = lay[..., 3:4] / 255.0
        out = neutral.copy()
        out[..., :3] = lay[..., :3] * a + neutral[..., :3] * (1.0 - a)
        da = neutral[..., 3:4] / 255.0
        out[..., 3] = (a + da * (1.0 - a))[..., 0] * 255.0
        return out

    def place(key, tile, x, y):
        """Blit a tile with `tile_pad` px of replicated edge around it."""
        p = tile_pad
        if p > 0:
            padded = np.pad(tile, ((p, p), (p, p), (0, 0)), mode="edge")
            blit(atlas, padded, x - p, y - p)
        else:
            blit(atlas, tile, x, y)
        th, tw = tile.shape[:2]
        placed[key] = dict(rect=(x, y, x + tw, y + th), size=(tw, th))

    # ----- eye detection (mask boxes, kept for reference) ------------------ #
    eyes = geom["eyes"]
    eye_variants = ["neutral", "half", "blink", "happy", "squint"]
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
        eye_boxes.append({"box": box, "rig_box": rig,
                          "center": (e["cx"], e["cy"]),
                          "cy": float(ys.min() + ys.max()) / 2.0,
                          "mask_y0": float(ys.min()), "mask_y1": float(ys.max()),
                          "size": (box[2] - box[0], box[3] - box[1]),
                          "variants": eye_variants})

    # ----- mouth detection ------------------------------------------------- #
    mouth_shapes = [
        ("closed", None),           # neutral from base
        ("slight", "M_slight_V1.png"),
        ("half", "M_half_V2.png"),
        ("A_open", "M_A_open_V3.png"),
        ("smile", "M_I_grin_V5.png"),
        ("smirk", "M_smirk_V8.png"),
    ]

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
        mouth_bbox = (int(mxs.min()), int(mys.min()), int(mxs.max()) + 1, int(mys.max()) + 1)
    else:
        lip_rgb = np.array([60.0, 25.0, 25.0])
        mcx, mcy, mw, mh = mx, my, 90, 24
        mys, mxs = np.where(mmask)
        mouth_bbox = (int(mx) - 45, int(my) - 12, int(mx) + 45, int(my) + 12)

    # ----- load hand-drawn layers once ------------------------------------- #
    variant_file = {
        "wide": "E_%s_wide_V4.png",
        "half": "E_%s_half_V7.png",
        "blink": "E_%s_blink_V6.png",
        "happy": "E_%s_happy_V5.png",
        "squint": "E_%s_squint_V8.png",
    }
    layer_cache = {}

    def eye_layer(side: str, vname: str):
        key = (side, vname)
        if key not in layer_cache:
            layer_cache[key] = load_layer(variant_file[vname] % side)
        return layer_cache[key]

    # mouth_src[i] = layer for mouth shape i (None -> neutral base crop)
    igrin = load_layer("M_I_grin_V5.png")
    mouth_src = [None]
    for name, fname in mouth_shapes[1:]:
        if fname is None:
            mouth_src.append(igrin)                       # smile
        else:
            mouth_src.append(load_layer(fname))

    # ----- final boxes ------------------------------------------------------ #
    # Eye box: mask rig box UNION full layer art (brows etc.) + margin, but
    # never below the top of the mouth art - the open mouth's lip and the
    # eye tiles would otherwise fight over the same rows.
    mouth_art_tops = [b[1] for b in (art_bbox(a) for a in mouth_src if a is not None) if b]
    mouth_art_tops.append(mouth_bbox[1])
    mouth_art_top = min(mouth_art_tops)

    for i, side in enumerate(("L", "R")):
        lb = union([art_bbox(eye_layer(side, v)) for v in variant_file])
        r = eye_boxes[i]["rig_box"]
        b = (min(r[0], lb[0]) - box_margin,
             min(r[1], lb[1]) - box_margin,
             max(r[2], lb[2]) + box_margin,
             max(r[3], lb[3]) + box_margin)
        b = clamp_box(b)
        eye_boxes[i]["rig_box"] = (b[0], b[1], b[2], min(b[3], mouth_art_top))

    eye_bottom = max(eb["rig_box"][3] for eb in eye_boxes)

    mb = union([art_bbox(a) for a in mouth_src if a is not None] + [mouth_bbox])
    mouth_box = clamp_box((mb[0] - box_margin, mb[1] - box_margin,
                           mb[2] + box_margin, mb[3] + box_margin))
    # the eye band ends at eye_bottom; the mouth band starts there too
    mouth_box = (mouth_box[0], max(mouth_box[1], eye_bottom),
                 mouth_box[2], mouth_box[3])
    assert mouth_box[3] - mouth_box[1] > 20, "mouth box collapsed: %s" % (mouth_box,)

    # ----- compose atlas ---------------------------------------------------- #
    # Left half: the original art untouched (no inpainting - a flat patch
    # under the face only ever showed up as a flickering square).
    AW = cfg["atlas_width"]
    AH = cfg["atlas_height"]
    atlas = np.zeros((AH, AW, 4), np.float32)
    blit(atlas, rgba, 0, 0)

    placed = {}
    mx0, my0, mx1, my1 = mouth_box
    mw_box, mh_box = mx1 - mx0, my1 - my0
    eye_w = [eb["rig_box"][2] - eb["rig_box"][0] for eb in eye_boxes]
    eye_h = [eb["rig_box"][3] - eb["rig_box"][1] for eb in eye_boxes]

    x_mouth = W + col_gap
    x_eye0 = x_mouth + mw_box + col_gap
    x_eye1 = x_eye0 + eye_w[0] + col_gap
    assert max(x_eye1 + eye_w[1], x_mouth + mw_box) + tile_pad <= AW, \
        "atlas columns do not fit into %d px" % AW

    # mouth column (bottom -> top matches mouth shape indices)
    y = 24
    mouth_tiles = []
    for i, (tile_src) in enumerate(mouth_src):
        tile = make_tile(mouth_box, tile_src)
        place("mouth_%d" % i, tile, x_mouth, y)
        mouth_tiles.append(("mouth_%d" % i, tile))
        y += tile.shape[0] + 12
    assert y + tile_pad <= AH, "mouth column overflows atlas height"

    # eye columns: one per eye, neutral first
    for i, side in enumerate(("L", "R")):
        rb = eye_boxes[i]["rig_box"]
        ex = (x_eye0, x_eye1)[i]
        y = 24
        for vname in eye_variants:
            layer = None if vname == "neutral" else eye_layer(side, vname)
            tile = make_tile(rb, layer)
            place("eye_%d_%s" % (i, vname), tile, ex, y)
            y += tile.shape[0] + 10
        assert y + tile_pad <= AH, "eye column %d overflows atlas height" % i

    _ensure(out_png)
    Image.fromarray(atlas.astype(np.uint8)).save(out_png)

    meta = dict(atlas_size=(AW, AH), base_rect=(0, 0, W, H),
                eyes=[{k: v for k, v in eb.items()}
                      for eb in eye_boxes],
                mouth_box=list(mouth_box), mouth_bbox=list(mouth_bbox),
                mouth_center=(mcx, mcy), mouth_size=(mw, mh),
                mouth_shapes=[n for n, _ in mouth_shapes],
                mouth_frame_count=len(mouth_tiles),
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
               mouth_pad_x=8, mouth_pad_y=8, mouth_lip=5,
               mouth_drops=(0, 10, 22, 34),
               atlas_width=2048, atlas_height=2048)
    meta = build_texture(args.src, args.out, args.meta, json.load(open(args.geom)), cfg)
    print(json.dumps(meta["placed"], indent=1))
    print("mouth box", meta["mouth_box"], "bbox", meta["mouth_bbox"])
    for i, e in enumerate(meta["eyes"]):
        print("eye %d box %s rig %s cy %.1f" % (i, e["box"], e["rig_box"], e["cy"]))


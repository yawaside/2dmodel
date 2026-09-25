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
    src = Image.open(src_png).convert("RGBA")
    W, H = src.size
    rgba = np.array(src).astype(np.float32)
    alpha = rgba[..., 3] > 128
    lum = 0.299 * rgba[..., 0] + 0.587 * rgba[..., 1] + 0.114 * rgba[..., 2]

    # ----- eyes ----------------------------------------------------------- #
    # Use hand-drawn PSD layers for all eye states instead of procedural deformation.
    # Layers in psd_layers/ are full 1024x1024 transparent overlays.
    eyes = geom["eyes"]
    eye_boxes = []
    eye_variants = ["neutral", "wide", "half", "blink", "happy", "squint"]
    placed_eyes = {0: {}, 1: {}}  # side -> variant -> rect in atlas
    for i, e in enumerate(eyes):
        side = "L" if i == 0 else "R"
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

    # ----- mouth ---------------------------------------------------------- #
    # Use hand-drawn PSD layers for mouth shapes instead of procedural generation.
    # Vowel shapes for perfect lip-sync: closed, slight, half, A_open, O, I_grin, plus smile/smirk.
    mouth_shapes = [
        ("closed", None),           # neutral from base
        ("slight", "M_slight_V1.png"),
        ("half", "M_half_V2.png"),
        ("A_open", "M_A_open_V3.png"),
        ("O", "M_O_V4.png"),
        ("I_grin", "M_I_grin_V5.png"),
        ("smile", None),            # will blend I + happy eyes
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
        # use the largest bbox across all mouth layers
        all_mouth_bboxes = []
        for _, fname in mouth_shapes:
            if fname is None:
                continue
            p = os.path.join("psd_layers", fname)
            if os.path.exists(p):
                larr = np.array(Image.open(p).convert("RGBA")).astype(np.float32)
                crop, (lx, ly) = crop_nontransparent(larr)
                all_mouth_bboxes.append((lx, ly, lx + crop.shape[1], ly + crop.shape[0]))
        if all_mouth_bboxes:
            bb = np.array(all_mouth_bboxes)
            px0_m = int(bb[:, 0].min())
            py0_m = int(bb[:, 1].min())
            px1_m = int(bb[:, 2].max())
            py1_m = int(bb[:, 3].max())
        else:
            px0_m, py0_m, px1_m, py1_m = int(mxs.min()) - 8, int(mys.min()) - 8, int(mxs.max()) + 1 + 8, int(mys.max()) + 1 + 40
    else:
        lip_rgb = np.array([60.0, 25.0, 25.0])
        mcx, mcy, mw, mh = mx, my, 90, 24
        px0_m, py0_m, px1_m, py1_m = int(mx) - 95, int(my) - 50, int(mx) + 95, int(my) + 60

    pw = px1_m - px0_m
    ph = py1_m - py0_m
    mouth_box = (px0_m, py0_m, px1_m, py1_m)
    mouth_bbox = (int(mxs.min()), int(mys.min()), int(mxs.max()) + 1, int(mys.max()) + 1)

    # ----- compose atlas --------------------------------------------------- #
    # Left half: base art with original eyes/mouth inpainted (covered by overlays).
    # Right half: all hand-drawn expression layers cropped to their bounding boxes.
    AW = cfg["atlas_width"]
    AH = cfg["atlas_height"]
    atlas = np.zeros((AH, AW, 4), np.float32)

    # Inpaint eyes and mouth on base so overlays can fully replace them
    base_rgba = rgba.copy()
    # inpaint eyes
    for eb in eye_boxes:
        x0, y0, x1, y1 = eb["box"]
        eye_mask_region = np.zeros((H, W), bool)
        eye_mask_region[y0:y1, x0:x1] = True
        base_rgba = inpaint_region(base_rgba.astype(np.uint8), eye_mask_region, ring=15, feather=3).astype(np.float32)
    # inpaint mouth
    base_rgba = inpaint_region(base_rgba.astype(np.uint8), mmask, ring=10, feather=3).astype(np.float32)
    blit(atlas, base_rgba, 0, 0)

    # Neutral eye/mouth variants are cut from original untouched art, so rest pose matches exactly
    # They will be overlaid on the inpainted base at full opacity, so neutral state is 1:1 source

    cur_x, cur_y = W + 24, 24
    col2_x = cur_x + 250  # second column for eye tiles
    placed = {}
    # Place mouth frames in first column
    mouth_tiles = []
    # 0: closed = base crop
    closed_crop = rgba[py0_m:py1_m, px0_m:px1_m].copy()
    mouth_tiles.append(("mouth_0", closed_crop))
    for idx, (name, fname) in enumerate(mouth_shapes[1:], start=1):
        if fname is not None:
            p = os.path.join("psd_layers", fname)
            if os.path.exists(p):
                larr = np.array(Image.open(p).convert("RGBA")).astype(np.float32)
                tile = larr[py0_m:py1_m, px0_m:px1_m]
                mouth_tiles.append((f"mouth_{idx}", tile))
            else:
                # fallback to closed
                mouth_tiles.append((f"mouth_{idx}", closed_crop))
        else:
            # smile = blend I_grin a bit wider, reuse I_grin for now placeholder
            igrin_p = os.path.join("psd_layers", "M_I_grin_V5.png")
            if os.path.exists(igrin_p):
                larr = np.array(Image.open(igrin_p).convert("RGBA")).astype(np.float32)
                tile = larr[py0_m:py1_m, px0_m:px1_m]
                mouth_tiles.append((f"mouth_{idx}", tile))
            else:
                mouth_tiles.append((f"mouth_{idx}", closed_crop))

    for mid, tile in mouth_tiles:
        th, tw = tile.shape[:2]
        blit(atlas, tile, cur_x, cur_y)
        placed[mid] = dict(rect=(cur_x, cur_y, cur_x + tw, cur_y + th), size=(tw, th))
        cur_y += th + 12

    # Place eye variants in second column (left eye first, then right eye)
    cur_y2 = 24
    for i, e in enumerate(eyes):
        side = "L" if i == 0 else "R"
        ex0, ey0, ex1, ey1 = e_box = eye_boxes[i]["rig_box"]
        ew, eh = ex1 - ex0, ey1 - ey0
        # neutral = crop from base art
        neutral_tile = rgba[ey0:ey1, ex0:ex1].copy()
        key = f"eye_{i}_neutral"
        blit(atlas, neutral_tile, col2_x, cur_y2)
        placed[key] = dict(rect=(col2_x, cur_y2, col2_x + ew, cur_y2 + eh), size=(ew, eh))
        cur_y2 += eh + 8
        # other variants
        variant_map = {
            "wide": f"E_{side}_wide_V4.png",
            "half": f"E_{side}_half_V7.png",
            "blink": f"E_{side}_blink_V6.png",
            "happy": f"E_{side}_happy_V5.png",
            "squint": f"E_{side}_squint_V8.png",
        }
        for vname, fname in variant_map.items():
            p = os.path.join("psd_layers", fname)
            if os.path.exists(p):
                larr = np.array(Image.open(p).convert("RGBA")).astype(np.float32)
                tile = larr[ey0:ey1, ex0:ex1]
                key = f"eye_{i}_{vname}"
                blit(atlas, tile, col2_x, cur_y2)
                placed[key] = dict(rect=(col2_x, cur_y2, col2_x + ew, cur_y2 + eh), size=(ew, eh))
                cur_y2 += eh + 8
            else:
                # fallback to neutral
                key = f"eye_{i}_{vname}"
                blit(atlas, neutral_tile, col2_x, cur_y2)
                placed[key] = dict(rect=(col2_x, cur_y2, col2_x + ew, cur_y2 + eh), size=(ew, eh))
                cur_y2 += eh + 8

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


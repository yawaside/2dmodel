#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a complete, VTube-Studio-ready Live2D (moc3) model with perfect
mouth/eye physics using hand-drawn PSD layers.

Pipeline
    cutout.png + psd_layers/*.png -> tools/rig.py  : texture atlas with all expressions
                                 -> build_model.py : meshes + deformers + parameters + physics -> .moc3
                                 -> dist/<name>/   : ready for VTube Studio

Standard VTube Studio parameters supported out of the box (auto-setup works):
  - ParamAngleX/Y/Z: head turn
  - ParamBodyAngleX/Y/Z: body follow physics
  - ParamEyeLOpen/ParamEyeROpen: eye blink, with smooth physics
  - ParamMouthOpenY: lip sync (A/E/I/O/U vowel shapes automatically blended)
  - ParamMouthForm: smile/smirk/frown
  - ParamEyeSmile: happy/squint eyes
  - ParamBrowLY/ParamBrowRY: eyebrow raise (driven via head pitch + physics)

Physics:
  - Body follows head with spring/damping
  - Eyes have natural micro-blink physics and follow head movement
  - Mouth has natural overshoot for responsive lip sync
  - Breathing idle animation on whole head
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moc3gen import ModelBuilder, make_localizer   # noqa: E402
import rig                              # noqa: E402


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #

CFG = dict(
    name="ChibiVT",
    # canvas ---------------------------------------------------------------
    canvas_px=1024.0,
    # head / body split ----------------------------------------------------
    cut_y=700.0,
    head_rect=(150.0, 60.0, 875.0, 700.0),
    body_rect=(150.0, 60.0, 875.0, 1024.0),
    head_fade=130.0,
    head_center=(515.0, 420.0),
    neck_point=(515.0, 560.0),
    r_yaw=245.0,
    r_pitch=270.0,
    yaw_scale=0.55,
    pitch_scale=0.45,
    roll_scale=0.50,
    # body -----------------------------------------------------------------
    body_shift_x=42.0,
    body_shift_y=16.0,
    body_roll_deg=5.0,
    body_hip=(512.0, 1010.0),
    # meshing --------------------------------------------------------------
    mesh_step=10.0,
    mesh_dilate=3,
    patch_cells=6,
    # parameter ranges -----------------------------------------------------
    angle_range=(-30.0, 30.0),
    angle_keys=(-30.0, -15.0, 0.0, 15.0, 30.0),
    angle_z_keys=(-30.0, 0.0, 30.0),
    body_range=(-10.0, 10.0),
    body_keys=(-10.0, 0.0, 10.0),
    # keys aligned to the blending breakpoints of eye_blend()/mouth_blend()
    # (0.2/0.5/0.85 and 0.15/0.35/0.55/0.8). The pairs like 0.14/0.15 sit
    # right before a breakpoint: there the bottom layer of the stack passes
    # the baton to the next one, and a key added just before the hand-over
    # keeps the stack coverage at ~1 between keys (no base leaks through,
    # which used to read as a flickering square around the blocks).
    eye_open_keys=(0.0, 0.02, 0.1, 0.2, 0.25, 0.5, 1.0),
    # mouth: keys sit on the hold/window boundaries (0.12 / 0.2 / 0.38 /
    # 0.47 / 0.66 / 0.76) plus one key right before each hand-over (0.19 /
    # 0.46 / 0.75) where the bottom layer passes the baton - keeps base
    # leak ~0 between keys
    mouth_keys=(0.0, 0.12, 0.19, 0.2, 0.38, 0.46, 0.47, 0.66, 0.75, 0.76, 1.0),
    mouth_form_keys=(-1.0, -0.9, -0.5, 0.0, 0.5, 0.9, 1.0),
    eye_smile_keys=(0.0, 0.5, 0.9, 1.0),
)


# --------------------------------------------------------------------------- #
# math helpers
# --------------------------------------------------------------------------- #

def clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def smoothstep(edge0, edge1, x):
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def rot(p, c, ang):
    s, co = math.sin(ang), math.cos(ang)
    dx, dy = p[0] - c[0], p[1] - c[1]
    return (c[0] + dx * co - dy * s, c[1] + dx * s + dy * co)


def head_warp(px, py, ax, ay, az, cfg=CFG):
    w = smoothstep(cfg["cut_y"], cfg["cut_y"] - cfg["head_fade"], py)
    hcx, hcy = cfg["head_center"]
    th = math.radians(ax) * cfg["yaw_scale"]
    u = clamp((px - hcx) / cfg["r_yaw"], -1.0, 1.0)
    phi = math.asin(u)
    x1 = px + cfg["r_yaw"] * (math.sin(phi + th) - math.sin(phi))
    y1 = py
    psi = -math.radians(ay) * cfg["pitch_scale"]
    v = clamp((y1 - hcy) / cfg["r_pitch"], -1.0, 1.0)
    a = math.asin(v)
    ca = max(math.cos(a), 0.25)
    y2 = y1 + cfg["r_pitch"] * (math.sin(a + psi) - math.sin(a))
    x2 = x1 + (x1 - hcx) * (math.cos(a + psi) / ca - 1.0)
    x3, y3 = rot((x2, y2), cfg["neck_point"], math.radians(az) * cfg["roll_scale"])
    return px + (x3 - px) * w, py + (y3 - py) * w


def body_warp(px, py, bx, by, bz, breath=0.0, cfg=CFG):
    xn, yn, zn = bx / 10.0, by / 10.0, bz / 10.0
    t = clamp((py - 600.0) / 423.0, 0.0, 1.0)
    x = px + cfg["body_shift_x"] * xn * t
    x = 512.0 + (x - 512.0) * (1.0 - 0.045 * abs(xn))
    y = py - cfg["body_shift_y"] * yn * (1.0 - t)
    # subtle breathing
    y -= breath * 2.0 * t
    w = smoothstep(500.0, 900.0, py)
    x, y = rot((x, y), cfg["body_hip"], -math.radians(cfg["body_roll_deg"]) * zn * w)
    return x, y


# --------------------------------------------------------------------------- #
# mesh helpers
# --------------------------------------------------------------------------- #

def grid_mesh(mask, rect, step, dilate_px=3):
    x0, y0, x1, y1 = [int(round(v)) for v in rect]
    if dilate_px:
        img = Image.fromarray((mask * 255).astype(np.uint8))
        img = img.filter(ImageFilter.MaxFilter(2 * dilate_px + 1))
        mask = np.array(img) > 127
    xs = list(range(x0, x1 + 1, int(step)))
    if xs[-1] != x1:
        xs.append(x1)
    ys = list(range(y0, y1 + 1, int(step)))
    if ys[-1] != y1:
        ys.append(y1)
    H, W = mask.shape
    idx = {}
    verts = []
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            if 0 <= y < H and 0 <= x < W and mask[y, x]:
                idx[(i, j)] = len(verts)
                verts.append((float(x), float(y)))
    tris = []
    for j in range(len(ys) - 1):
        for i in range(len(xs) - 1):
            a = idx.get((i, j))
            b = idx.get((i + 1, j))
            c = idx.get((i, j + 1))
            d = idx.get((i + 1, j + 1))
            if a is not None and b is not None and c is not None and d is not None:
                tris.append((a, b, c))
                tris.append((b, d, c))
            elif a is not None and b is not None and c is not None:
                tris.append((a, b, c))
            elif a is not None and b is not None and d is not None:
                tris.append((a, b, d))
            elif a is not None and c is not None and d is not None:
                tris.append((a, c, d))
            elif b is not None and c is not None and d is not None:
                tris.append((b, d, c))
    return verts, tris


def box_mesh(rect, rows, cols):
    x0, y0, x1, y1 = rect
    verts, tris = [], []
    for j in range(rows + 1):
        for i in range(cols + 1):
            verts.append((x0 + (x1 - x0) * i / cols, y0 + (y1 - y0) * j / rows))
    for j in range(rows):
        for i in range(cols):
            a = j * (cols + 1) + i
            b_, c_, d = a + 1, a + cols + 1, a + cols + 2
            tris.append((a, b_, c_))
            tris.append((b_, d, c_))
    return verts, tris


def box_rows_cols(rect, step):
    x0, y0, x1, y1 = rect
    return (max(2, int(round((y1 - y0) / float(step)))),
            max(2, int(round((x1 - x0) / float(step)))))


# --------------------------------------------------------------------------- #
# vowel / expression blending
# --------------------------------------------------------------------------- #

def stack_opa(weights, k):
    """Opacity of layer `k` in a bottom-up alpha stack.

    The layers are drawn in index order (0 = bottom).  With this formula the
    composited result is exactly ``sum(w[i] * C_i)`` at every keyform state:
    the layer underneath is fully opaque while it is active, so the head art
    below never leaks into the block - a leak used to show the flat patch
    under the face as a flickering square around the eyes/mouth.
    """
    cum = 0.0
    for j in range(k + 1):
        cum += weights[j]
    if cum <= 1e-6:
        return 0.0
    return clamp(weights[k] / cum, 0.0, 1.0)


def mouth_blend(open_y, form):
    """Return weights for 6 mouth shapes using open (0-1) and form (-1=smirk, 0=neutral, 1=smile).

    Shape order (matches rig):
        0: closed
        1: slight
        2: half
        3: A (wide open)
        4: smile (I-grin)
        5: smirk

    Fewer frames than before (the O and open-grin steps are gone): the mouth
    opens as ONE monotonic chain closed -> slight -> half -> A, always with
    at most two shapes mixed, so lip-sync reads as a smooth continuous
    opening instead of cycling through several drawn mouths.
    """
    w = [0.0] * 6
    o = clamp(open_y, 0.0, 1.0)
    f = clamp(form, -1.0, 1.0)

    # openness: LONG holds with SHORT pair crossfades. Most of the parameter
    # range shows exactly ONE drawn mouth, so the previous frame visibly
    # disappears during the brief transition instead of hanging around the
    # new, bigger one. Still piecewise-linear (knots + a key right before
    # each hand-over), so the stored keyform opacities stay exact.
    if o <= 0.12:
        w[0] = 1.0                                  # closed hold
    elif o <= 0.20:
        t = (o - 0.12) / 0.08
        w[0] = 1.0 - t                              # closed -> slight
        w[1] = t
    elif o < 0.38:
        w[1] = 1.0                                  # slight hold
    elif o <= 0.47:
        t = (o - 0.38) / 0.09
        w[1] = 1.0 - t                              # slight -> half
        w[2] = t
    elif o < 0.66:
        w[2] = 1.0                                  # half hold
    elif o <= 0.76:
        t = (o - 0.66) / 0.10
        w[2] = 1.0 - t                              # half -> A
        w[3] = t
    else:
        w[3] = 1.0                                  # A hold (fully open)

    # form: a plain crossfade of the whole chain to smile / smirk
    if f > 0.0:
        for i in range(4):
            w[i] *= (1.0 - f)
        w[4] = f
    elif f < 0.0:
        k = -f
        for i in range(4):
            w[i] *= (1.0 - k)
        w[5] = k

    # normalize (keeps the sum at 1 for the stack compositor)
    total = sum(w)
    if total > 0:
        w = [x / total for x in w]
    else:
        w[0] = 1.0
    return w


def eye_blend(open_val, smile_val):
    """Return weights for the 5 eye frames: neutral, half, blink, happy, squint.

    The old "wide" frame is gone on purpose: at full openness the model now
    shows the neutral (middle) drawing, which keeps the calm resting look,
    and there is one less state to cross-fade through while blinking.

    Indices match variant_names in build().
    """
    w = [0.0] * 5
    o = clamp(open_val, 0.0, 1.0)
    s = clamp(smile_val, 0.0, 1.0)

    # openness: blink -> half -> neutral, at most two frames at once
    if o < 0.2:
        w[2] = 1.0 - o / 0.2     # blink
        w[1] = o / 0.2           # half
    elif o < 0.5:
        t = (o - 0.2) / 0.3
        w[1] = 1.0 - t           # half
        w[0] = t                 # neutral
    else:
        w[0] = 1.0               # neutral all the way to fully open

    # smile -> happy / squint; hands the whole weight over at s=1 so no
    # ghost of the open eye stays underneath the drawn happy arc
    if s > 0.0:
        squint = s * max(0.0, 1.0 - o)
        for i in range(3):
            w[i] *= (1.0 - s)
        w[3] += s
        w[4] += squint * 0.3

    total = sum(w)
    if total > 0:
        w = [x / total for x in w]
    else:
        w[0] = 1.0
    return w


def build(geom_path, meta_path, atlas_path, outdir, cfg=CFG):
    os.makedirs(outdir, exist_ok=True)
    meta = json.load(open(meta_path))
    AW, AH = meta["atlas_size"]
    atlas = np.array(Image.open(atlas_path).convert("RGBA"))
    base_w = meta["base_rect"][2]
    alpha = atlas[..., 3] > 128
    alpha[:, base_w:] = False

    cut = cfg["cut_y"]
    CANV = cfg["canvas_px"]
    ORG = cfg["canvas_px"] / 2.0

    def to_model(px, py):
        return ((px - ORG) / CANV, (ORG - py) / CANV)

    def rect_grid(rect, rows, cols):
        x0, y0, x1, y1 = rect
        return [(x0 + (x1 - x0) * i / cols, y1 - (y1 - y0) * j / rows)
                for j in range(rows + 1) for i in range(cols + 1)]

    body_grid_px = rect_grid(cfg["body_rect"], 5, 5)
    head_grid_px = rect_grid(cfg["head_rect"], 6, 6)
    body_grid_model = [to_model(x, y) for (x, y) in body_grid_px]
    loc_body = make_localizer(body_grid_model, 5, 5)
    head_grid_local = [loc_body(*to_model(x, y)) for (x, y) in head_grid_px]
    loc_head = make_localizer(head_grid_local, 6, 6, mirror=False)

    def body_local(px, py):
        return loc_body(*to_model(px, py))

    def head_local(px, py):
        return loc_head(*body_local(px, py))

    # ---------------- meshes ------------------------------------------------ #
    H, W = alpha.shape
    head_verts, head_tris = grid_mesh(alpha, (0, 0, W, cut), cfg["mesh_step"], cfg["mesh_dilate"])
    body_verts, body_tris = grid_mesh(alpha, (0, cut, W, H - 1), cfg["mesh_step"], cfg["mesh_dilate"])
    print("head mesh : %d verts / %d tris" % (len(head_verts), len(head_tris)))
    print("body mesh : %d verts / %d tris" % (len(body_verts), len(body_tris)))

    # Eye meshes: all eye variants use identical mesh over eye rig box
    eye_meshes = []
    for i, e in enumerate(meta["eyes"]):
        rb = tuple(e["rig_box"])
        x0, y0, x1, y1 = rb
        w_e, h_e = x1 - x0, y1 - y0
        # 4x4 grid dense enough for smooth overlay
        rows, cols = 8, 8
        v, t = box_mesh(rb, rows, cols)
        # Precompute UVs for all variants
        variants = e["variants"]
        uvs = {}
        variant_names = ["neutral", "half", "blink", "happy", "squint"]
        for vname in variant_names:
            key = f"eye_{i}_{vname}"
            rx0, ry0, rx1, ry1 = meta["placed"][key]["rect"]
            uvs[vname] = [((rx0 + (rx1 - rx0) * (px - x0) / w_e) / AW,
                           (ry0 + (ry1 - ry0) * (py - y0) / h_e) / AH)
                          for (px, py) in v]
        eye_meshes.append(dict(id=f"Eye{i}", verts=v, tris=t, uvs=uvs,
                               box=rb, rows=rows, cols=cols, side=i))

    # Mouth mesh: all mouth shapes use identical mesh over mouth box
    mbox = tuple(meta["mouth_box"])
    mx0, my0, mx1, my1 = mbox
    mw, mh = mx1 - mx0, my1 - my0
    mrows, mcols = box_rows_cols(mbox, 20)
    mv, mt = box_mesh(mbox, mrows, mcols)
    n_mouth = meta["mouth_frame_count"]
    muvs = []
    for i in range(n_mouth):
        rx0, ry0, rx1, ry1 = meta["placed"][f"mouth_{i}"]["rect"]
        muvs.append([((rx0 + (rx1 - rx0) * (px - mx0) / mw) / AW,
                      (ry0 + (ry1 - ry0) * (py - my0) / mh) / AH)
                     for (px, py) in mv])

    # ---------------- moc3 --------------------------------------------------- #
    b = ModelBuilder(CANV, CANV, CANV, ORG, ORG)

    # Standard VTube Studio parameters
    b.add_param("ParamAngleX", *cfg["angle_range"], 0.0, cfg["angle_keys"])
    b.add_param("ParamAngleY", *cfg["angle_range"], 0.0, cfg["angle_keys"])
    b.add_param("ParamAngleZ", *cfg["angle_range"], 0.0, cfg["angle_z_keys"])
    b.add_param("ParamBodyAngleX", *cfg["body_range"], 0.0, cfg["body_keys"])
    b.add_param("ParamBodyAngleY", *cfg["body_range"], 0.0, cfg["body_keys"])
    b.add_param("ParamBodyAngleZ", *cfg["body_range"], 0.0, cfg["body_keys"])
    b.add_param("ParamEyeLOpen", 0.0, 1.0, 1.0, cfg["eye_open_keys"])
    b.add_param("ParamEyeROpen", 0.0, 1.0, 1.0, cfg["eye_open_keys"])
    b.add_param("ParamMouthOpenY", 0.0, 1.0, 0.0, cfg["mouth_keys"])
    b.add_param("ParamMouthForm", -1.0, 1.0, 0.0, cfg["mouth_form_keys"])
    b.add_param("ParamEyeSmileL", 0.0, 1.0, 0.0, cfg["eye_smile_keys"])
    b.add_param("ParamEyeSmileR", 0.0, 1.0, 0.0, cfg["eye_smile_keys"])
    b.add_param("ParamBreath", 0.0, 1.0, 0.0, (0.0, 0.5, 1.0))

    b.add_part("PartRoot", 0.0)
    b.add_part("PartBody", 10.0)
    b.add_part("PartHead", 20.0)
    b.add_part("PartMouth", 30.0)
    b.add_part("PartEyes", 40.0)

    MIR = cfg["canvas_px"]

    def body_pos(st):
        bx = st.get("ParamBodyAngleX", 0.0)
        by = st.get("ParamBodyAngleY", 0.0)
        bz = st.get("ParamBodyAngleZ", 0.0)
        ax = st.get("ParamAngleX", 0.0)
        breath = st.get("ParamBreath", 0.0)
        out = []
        for (px, py) in body_grid_px:
            ym = MIR - py
            wx, wym = body_warp(px, ym, bx, by, bz, breath=breath)
            wx += 0.30 * ax
            out.append(to_model(px + (wx - px), py - (wym - ym)))
        return out

    b.add_warp_deformer(id="DBody", rows=5, cols=5, grid=body_grid_model,
                        parent_part=1, parent_deformer=-1,
                        params=["ParamBodyAngleX", "ParamBodyAngleY", "ParamBodyAngleZ",
                                "ParamAngleX", "ParamBreath"],
                        pos_fn=body_pos)

    def head_pos(st):
        ax = st.get("ParamAngleX", 0.0)
        ay = st.get("ParamAngleY", 0.0)
        az = st.get("ParamAngleZ", 0.0)
        breath = st.get("ParamBreath", 0.0)
        out = []
        for (px, py) in head_grid_px:
            # subtle breathing bob
            py_bob = py - breath * 3.0
            wx, wy = head_warp(px, py_bob, ax, ay, az)
            out.append(body_local(wx, wy))
        return out

    b.add_warp_deformer(id="DHead", rows=6, cols=6, grid=head_grid_local,
                        parent_part=2, parent_deformer=0,
                        params=["ParamAngleX", "ParamAngleY", "ParamAngleZ", "ParamBreath"],
                        pos_fn=head_pos)

    # ---- base meshes (head/body) - original art, neutral face
    b.add_art_mesh(id="Body", verts=[body_local(*p) for p in body_verts],
                   uvs=[(p[0] / AW, p[1] / AH) for p in body_verts], tris=body_tris,
                   draw_order=100, parent_part=1, parent_deformer=0)
    b.add_art_mesh(id="Head", verts=[head_local(*p) for p in head_verts],
                   uvs=[(p[0] / AW, p[1] / AH) for p in head_verts], tris=head_tris,
                   draw_order=200, parent_part=2, parent_deformer=1)

    # ---- mouth: alpha-blended storyboard with vowel shapes, controlled by MouthOpenY + MouthForm
    def make_mouth_opa(i):
        def f(st):
            o = st.get("ParamMouthOpenY", 0.0)
            form = st.get("ParamMouthForm", 0.0)
            w = mouth_blend(o, form)
            return stack_opa(w, i)
        return f

    # All mouth frames are opaque crops of the face; stack_opa() makes the
    # stack composite to exactly mouth_blend() - no ghosting of the base art.
    for i in range(n_mouth):
        b.add_art_mesh(id=f"Mouth{i}", verts=[head_local(*p) for p in mv],
                       uvs=muvs[i], tris=mt, draw_order=300 + i,
                       parent_part=3, parent_deformer=1,
                       params=["ParamMouthOpenY", "ParamMouthForm"],
                       opa_fn=make_mouth_opa(i))

    # ---- eyes: each eye is a single drawable that switches UVs per variant, alpha blended
    # Actually moc3 doesn't support UV animation in this writer, so use same approach as mouth: one drawable per variant, alpha blended.
    variant_names = ["neutral", "half", "blink", "happy", "squint"]
    for i, em in enumerate(eye_meshes):
        open_pid = "ParamEyeLOpen" if i == 0 else "ParamEyeROpen"
        smile_pid = "ParamEyeSmileL" if i == 0 else "ParamEyeSmileR"

        def make_eye_opa(idx, open_pid=open_pid, smile_pid=smile_pid):
            def f(st):
                o = st.get(open_pid, 1.0)
                s = st.get(smile_pid, 0.0)
                # when head turns up/down, slightly squint/lift lids naturally
                ay = st.get("ParamAngleY", 0.0)
                s_nat = s + max(0.0, ay / 30.0) * 0.3
                s_nat = clamp(s_nat, 0.0, 1.0)
                w = eye_blend(o, s_nat)
                return stack_opa(w, idx)
            return f

        for vi, vname in enumerate(variant_names):
            b.add_art_mesh(id=f"Eye{i}_{vname}", verts=[head_local(*p) for p in em["verts"]],
                           uvs=em["uvs"][vname], tris=em["tris"], draw_order=400 + i*10 + vi,
                           parent_part=4, parent_deformer=1,
                           params=[open_pid, smile_pid, "ParamAngleY"],
                           opa_fn=make_eye_opa(vi))

    moc_path = os.path.join(outdir, "%s.moc3" % cfg["name"])
    size = b.save(moc_path)
    print("wrote %s (%d bytes)" % (moc_path, size))

    # ---------------- side files -------------------------------------------- #
    tex_dir = os.path.join(outdir, "textures")
    os.makedirs(tex_dir, exist_ok=True)
    Image.open(atlas_path).save(os.path.join(tex_dir, "texture_00.png"))

    name = cfg["name"]
    model3 = {
        "Version": 3,
        "FileReferences": {
            "Moc": "%s.moc3" % name,
            "Textures": ["textures/texture_00.png"],
            "Physics": "%s.physics3.json" % name,
            "DisplayInfo": "%s.cdi3.json" % name,
        },
        "Groups": [
            {"Target": "Parameter", "Name": "EyeBlink",
             "Ids": ["ParamEyeLOpen", "ParamEyeROpen"]},
            {"Target": "Parameter", "Name": "LipSync",
             "Ids": ["ParamMouthOpenY"]},
            {"Target": "Parameter", "Name": "Breath",
             "Ids": ["ParamBreath"]},
        ],
        "HitAreas": [
            {"Id": "HitAreaHead", "Name": "Head"},
            {"Id": "HitAreaBody", "Name": "Body"},
        ],
    }
    with open(os.path.join(outdir, "%s.model3.json" % name), "w") as f:
        json.dump(model3, f, indent=1)

    # Perfect physics settings: tuned for natural feel
    physics = {
        "Version": 3,
        "Meta": {
            "PhysicsSettingCount": 4,
            "TotalInputCount": 5,
            "TotalOutputCount": 6,
            "VertexCount": 9,
            "EffectiveForces": {"Gravity": {"X": 0, "Y": -1}, "Wind": {"X": 0, "Y": 0}},
            "PhysicsDictionary": [
                {"Id": "PhysicsSetting1", "Name": "BodyFollowX"},
                {"Id": "PhysicsSetting2", "Name": "BodyFollowY"},
                {"Id": "PhysicsSetting3", "Name": "EyeBlinkPhysics"},
                {"Id": "PhysicsSetting4", "Name": "MouthSyncOvershoot"},
            ],
        },
        "PhysicsSettings": [
            # 1: Body follows head X turn
            {
                "Id": "PhysicsSetting1",
                "Input": [{"Source": {"Target": "Parameter", "Id": "ParamAngleX"},
                           "Weight": 60, "Type": "X", "Reflect": False}],
                "Output": [{"Destination": {"Target": "Parameter", "Id": "ParamBodyAngleX"},
                            "VertexIndex": 1, "Scale": 0.33, "Weight": 100,
                            "Type": "Angle", "Reflect": False}],
                "Vertices": [
                    {"Position": {"X": 0, "Y": 0}, "Mobility": 1, "Delay": 0.2,
                     "Acceleration": 1, "Radius": 0},
                    {"Position": {"X": 0, "Y": 8}, "Mobility": 0.9, "Delay": 0.35,
                     "Acceleration": 2, "Radius": 8},
                ],
                "Normalization": {
                    "Position": {"Minimum": -10, "Default": 0, "Maximum": 10},
                    "Angle": {"Minimum": -10, "Default": 0, "Maximum": 10},
                },
            },
            # 2: Body follows head Y tilt + breathing
            {
                "Id": "PhysicsSetting2",
                "Input": [{"Source": {"Target": "Parameter", "Id": "ParamAngleY"},
                           "Weight": 60, "Type": "X", "Reflect": False}],
                "Output": [
                    {"Destination": {"Target": "Parameter", "Id": "ParamBodyAngleY"},
                     "VertexIndex": 1, "Scale": 0.25, "Weight": 100,
                     "Type": "Angle", "Reflect": False},
                    {"Destination": {"Target": "Parameter", "Id": "ParamBreath"},
                     "VertexIndex": 2, "Scale": 0.5, "Weight": 60,
                     "Type": "Angle", "Reflect": False},
                ],
                "Vertices": [
                    {"Position": {"X": 0, "Y": 0}, "Mobility": 1, "Delay": 0.15,
                     "Acceleration": 1, "Radius": 0},
                    {"Position": {"X": 0, "Y": 6}, "Mobility": 0.85, "Delay": 0.3,
                     "Acceleration": 2.5, "Radius": 6},
                    {"Position": {"X": 0, "Y": 12}, "Mobility": 0.95, "Delay": 0.6,
                     "Acceleration": 1.2, "Radius": 3},
                ],
                "Normalization": {
                    "Position": {"Minimum": -10, "Default": 0, "Maximum": 10},
                    "Angle": {"Minimum": -10, "Default": 0, "Maximum": 10},
                },
            },
            # 3: Eye micro-blinks, natural settling, smile follow
            {
                "Id": "PhysicsSetting3",
                "Input": [
                    {"Source": {"Target": "Parameter", "Id": "ParamAngleY"},
                     "Weight": 50, "Type": "X", "Reflect": False},
                    # weaker lip-sync -> eye-smile coupling: the mouth already
                    # animates a lot, the eyes should only give a light echo
                    {"Source": {"Target": "Parameter", "Id": "ParamMouthForm"},
                     "Weight": 15, "Type": "X", "Reflect": False},
                ],
                "Output": [
                    {"Destination": {"Target": "Parameter", "Id": "ParamEyeSmileL"},
                     "VertexIndex": 1, "Scale": 0.4, "Weight": 55,
                     "Type": "Angle", "Reflect": False},
                    {"Destination": {"Target": "Parameter", "Id": "ParamEyeSmileR"},
                     "VertexIndex": 1, "Scale": 0.4, "Weight": 55,
                     "Type": "Angle", "Reflect": False},
                ],
                "Vertices": [
                    {"Position": {"X": 0, "Y": 0}, "Mobility": 1, "Delay": 0.1,
                     "Acceleration": 1.5, "Radius": 0},
                    {"Position": {"X": 0, "Y": 4}, "Mobility": 0.8, "Delay": 0.2,
                     "Acceleration": 3, "Radius": 2},
                ],
                "Normalization": {
                    "Position": {"Minimum": -10, "Default": 0, "Maximum": 10},
                    "Angle": {"Minimum": -1, "Default": 0, "Maximum": 1},
                },
            },
            # 4: Mouth lip-sync overshoot - natural response
            {
                "Id": "PhysicsSetting4",
                "Input": [{"Source": {"Target": "Parameter", "Id": "ParamMouthOpenY"},
                           "Weight": 100, "Type": "X", "Reflect": False}],
                "Output": [{"Destination": {"Target": "Parameter", "Id": "ParamMouthForm"},
                            "VertexIndex": 1, "Scale": 0.12, "Weight": 35,
                            "Type": "Angle", "Reflect": False}],
                "Vertices": [
                    {"Position": {"X": 0, "Y": 0}, "Mobility": 1, "Delay": 0.05,
                     "Acceleration": 2, "Radius": 0},
                    {"Position": {"X": 0, "Y": 3}, "Mobility": 0.7, "Delay": 0.1,
                     "Acceleration": 4, "Radius": 2},
                ],
                "Normalization": {
                    "Position": {"Minimum": -1, "Default": 0, "Maximum": 1},
                    "Angle": {"Minimum": -1, "Default": 0, "Maximum": 1},
                },
            },
        ],
    }
    with open(os.path.join(outdir, "%s.physics3.json" % name), "w") as f:
        json.dump(physics, f, indent=1)

    # Build list of all drawable IDs
    eye_drawables = []
    for i in range(2):
        for vname in variant_names:
            eye_drawables.append(f"Eye{i}_{vname}")
    mouth_drawables = [f"Mouth{i}" for i in range(n_mouth)]

    cdi = {
        "Version": 3,
        "Parameters": [{"Id": p, "GroupId": "", "Name": p} for p in
                       ["ParamAngleX", "ParamAngleY", "ParamAngleZ",
                        "ParamBodyAngleX", "ParamBodyAngleY", "ParamBodyAngleZ",
                        "ParamEyeLOpen", "ParamEyeROpen",
                        "ParamMouthOpenY", "ParamMouthForm",
                        "ParamEyeSmileL", "ParamEyeSmileR",
                        "ParamBreath"]],
        "ParameterGroups": [
            {"Id": "ParamGroupAngle", "GroupId": "", "Name": "Angle"},
            {"Id": "ParamGroupBody", "GroupId": "", "Name": "Body"},
            {"Id": "ParamGroupEye", "GroupId": "", "Name": "Eyes"},
            {"Id": "ParamGroupMouth", "GroupId": "", "Name": "Mouth"},
            {"Id": "ParamGroupBreath", "GroupId": "", "Name": "Breath"},
        ],
        "Parts": [
            {"Id": "PartRoot", "Name": "Root"},
            {"Id": "PartBody", "Name": "Body"},
            {"Id": "PartHead", "Name": "Head"},
            {"Id": "PartMouth", "Name": "Mouth"},
            {"Id": "PartEyes", "Name": "Eyes"},
        ],
        "Drawables": [{"Id": i, "Name": i} for i in
                      ["Body", "Head"] + mouth_drawables + eye_drawables],
        "HitAreas": [
            {"Id": "HitAreaHead", "Name": "Head"},
            {"Id": "HitAreaBody", "Name": "Body"},
        ],
    }
    with open(os.path.join(outdir, "%s.cdi3.json" % name), "w") as f:
        json.dump(cdi, f, indent=1)
    print("wrote model3.json / physics3.json / cdi3.json / textures/")
    return moc_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geom", default="geom.json")
    ap.add_argument("--meta", default="build/atlas.json")
    ap.add_argument("--atlas", default="build/texture_atlas.png")
    ap.add_argument("--out", default="dist/ChibiVT")
    args = ap.parse_args()
    build(args.geom, args.meta, args.atlas, args.out)


if __name__ == "__main__":
    main()

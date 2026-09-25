#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a complete, VTube-Studio-ready Live2D (moc3) model out of a single
front-facing character cut-out.

Pipeline
    cutout.png  ->  tools/rig.py   : texture atlas (eyes in-painted, open mouth)
                 ->  build_model.py : meshes + deformers + parameters -> .moc3
                 ->  dist/<name>/   : .model3.json / .moc3 / textures / physics

Everything is authored in the pixel space of the source art; the conversion to
the moc3 coordinate systems happens in the helpers below.
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
    canvas_px=1024.0,          # canvas size in pixels  (== source art size)
    # head / body split ----------------------------------------------------
    cut_y=700.0,               # pixel row where head mesh ends / body begins
    head_rect=(150.0, 60.0, 875.0, 700.0),   # warp grid D_Head (px)
    body_rect=(150.0, 60.0, 875.0, 1024.0),  # warp grid D_Body (px)
    head_fade=130.0,           # px over which head motion fades to 0 at the cut
    head_center=(515.0, 420.0),
    neck_point=(515.0, 560.0),
    r_yaw=245.0,               # "cylinder" radius used for left/right turn
    r_pitch=270.0,             # "sphere" radius used for up/down turn
    yaw_scale=0.55,            # how much of ParamAngleX becomes cylinder yaw
    pitch_scale=0.45,
    roll_scale=0.50,
    # body -----------------------------------------------------------------
    body_shift_x=42.0,         # px horizontal shift at the bottom (|X| = 10)
    body_shift_y=16.0,         # px vertical shift at the shoulders (|Y| = 10)
    body_roll_deg=5.0,
    body_hip=(512.0, 1010.0),
    # meshing --------------------------------------------------------------
    mesh_step=10.0,            # silhouette grid step in px
    mesh_dilate=3,
    patch_cells=6,             # eye / mouth patch grid resolution
    # blink ----------------------------------------------------------------
    eye_keys=(0.0, 0.5, 1.0),
    eye_closed_scale=0.12,     # высота глаза в закрытом состоянии
    # mouth ---------------------------------------------------------------
    mouth_grow=2.2,            # как быстро рот набирает полную высоту
    mouth_fade=0.10,           # и как быстро становится непрозрачным
    # parameter ranges -----------------------------------------------------
    angle_range=(-30.0, 30.0),
    angle_keys=(-30.0, -15.0, 0.0, 15.0, 30.0),
    angle_z_keys=(-30.0, 0.0, 30.0),
    body_range=(-10.0, 10.0),
    body_keys=(-10.0, 0.0, 10.0),
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
    """Fake-3D head rotation.  Returns the deformed pixel position.

    Every term is written as a *displacement* relative to the neutral pose so
    that (0, 0, 0) is exactly the identity - otherwise the neutral keyform of
    the deformer would not match the mesh layout and the model would be
    mis-assembled.
    """
    w = smoothstep(cfg["cut_y"], cfg["cut_y"] - cfg["head_fade"], py)
    hcx, hcy = cfg["head_center"]
    # --- yaw: wrap the flat art onto a cylinder and spin it -----------------
    th = math.radians(ax) * cfg["yaw_scale"]
    u = clamp((px - hcx) / cfg["r_yaw"], -1.0, 1.0)
    phi = math.asin(u)
    x1 = px + cfg["r_yaw"] * (math.sin(phi + th) - math.sin(phi))
    y1 = py
    # --- pitch: sphere rotation about the horizontal axis -------------------
    psi = -math.radians(ay) * cfg["pitch_scale"]   # +AngleY looks up
    v = clamp((y1 - hcy) / cfg["r_pitch"], -1.0, 1.0)
    a = math.asin(v)
    ca = max(math.cos(a), 0.25)
    y2 = y1 + cfg["r_pitch"] * (math.sin(a + psi) - math.sin(a))
    x2 = x1 + (x1 - hcx) * (math.cos(a + psi) / ca - 1.0)
    # --- roll: tilt around the neck ----------------------------------------
    x3, y3 = rot((x2, y2), cfg["neck_point"], math.radians(az) * cfg["roll_scale"])
    return px + (x3 - px) * w, py + (y3 - py) * w


def body_warp(px, py, bx, by, bz, cfg=CFG):
    """Body lean / turn / tilt. Returns deformed pixel position."""
    xn, yn, zn = bx / 10.0, by / 10.0, bz / 10.0
    t = clamp((py - 600.0) / 423.0, 0.0, 1.0)
    x = px + cfg["body_shift_x"] * xn * t
    x = 512.0 + (x - 512.0) * (1.0 - 0.045 * abs(xn))
    y = py - cfg["body_shift_y"] * yn * (1.0 - t)
    w = smoothstep(500.0, 900.0, py)
    x, y = rot((x, y), cfg["body_hip"], -math.radians(cfg["body_roll_deg"]) * zn * w)
    return x, y


# --------------------------------------------------------------------------- #
# mesh helpers
# --------------------------------------------------------------------------- #

def grid_mesh(mask, rect, step, dilate_px=3):
    """Triangulate `mask` with a regular grid inside `rect` (pixel space).

    Returns verts [(x, y)], tris [(a, b, c)] in pixel coordinates.
    """
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
    """Regular rows x cols grid over `rect` (pixel space)."""
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
    """Grid resolution matching the silhouette mesh, so the patches and the
    face are interpolated by exactly the same deformation."""
    x0, y0, x1, y1 = rect
    return (max(2, int(round((y1 - y0) / float(step)))),
            max(2, int(round((x1 - x0) / float(step)))))


def patch_mesh(rect, cells):
    """Regular grid over a rectangle (pixel space)."""
    x0, y0, x1, y1 = rect
    verts, tris = [], []
    for j in range(cells + 1):
        for i in range(cells + 1):
            verts.append((x0 + (x1 - x0) * i / cells, y0 + (y1 - y0) * j / cells))
    for j in range(cells):
        for i in range(cells):
            a = j * (cells + 1) + i
            b = a + 1
            c = a + cells + 1
            d = c + 1
            tris.append((a, b, c))
            tris.append((b, d, c))
    return verts, tris


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #

def build(geom_path, meta_path, atlas_path, outdir, cfg=CFG):
    os.makedirs(outdir, exist_ok=True)
    meta = json.load(open(meta_path))
    AW, AH = meta["atlas_size"]
    atlas = np.array(Image.open(atlas_path).convert("RGBA"))
    base_w = meta["base_rect"][2]
    alpha = atlas[..., 3] > 128
    alpha[:, base_w:] = False          # only the character half is a mesh source

    cut = cfg["cut_y"]
    CANV = cfg["canvas_px"]
    ORG = cfg["canvas_px"] / 2.0

    def to_model(px, py):
        """source-art pixel -> moc3 model space"""
        return ((px - ORG) / CANV, (ORG - py) / CANV)

    def rect_grid(rect, rows, cols):
        # Rows are emitted bottom -> top.  Cubism flips the y axis once when a
        # root deformer maps into model space, so with this order the local
        # coordinate of a point grows with the image row, which keeps the
        # deformation maths below readable.
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

    # Eyes: a mesh as dense as the face mesh, sampling the original art right
    # where the eye is - no copied tile, no in-painting, no feathered edge.
    eye_meshes = []
    for i, e in enumerate(meta["eyes"]):
        rb = tuple(e["rig_box"])
        h = float(rb[3] - rb[1])
        rows, cols = box_rows_cols(rb, cfg["mesh_step"])
        v, t = box_mesh(rb, rows, cols)
        uvs = [(px / AW, py / AH) for (px, py) in v]
        eye_meshes.append(dict(id="Eye%d" % i, verts=v, tris=t, uvs=uvs,
                               box=rb, rows=rows, cols=cols,
                               te0=(e["mask_y0"] - rb[1]) / h,
                               te1=(e["mask_y1"] - rb[1]) / h,
                               tc=(e["cy"] - rb[1]) / h))

    mbox = tuple(meta["mouth_box"])
    rows, cols = box_rows_cols(mbox, cfg["mesh_step"])
    mv, mt = box_mesh(mbox, rows, cols)

    def uvs_for(box, rect, verts):
        x0, y0, x1, y1 = box
        rx0, ry0, rx1, ry1 = rect
        return [((rx0 + (rx1 - rx0) * (px - x0) / (x1 - x0)) / AW,
                 (ry0 + (ry1 - ry0) * (py - y0) / (y1 - y0)) / AH)
                for (px, py) in verts]

    muvs = uvs_for(mbox, meta["placed"]["mouth_open"]["rect"], mv)

    # ---------------- moc3 --------------------------------------------------- #
    b = ModelBuilder(CANV, CANV, CANV, ORG, ORG)
    b.add_param("ParamAngleX", *cfg["angle_range"], 0.0, cfg["angle_keys"])
    b.add_param("ParamAngleY", *cfg["angle_range"], 0.0, cfg["angle_keys"])
    b.add_param("ParamAngleZ", *cfg["angle_range"], 0.0, cfg["angle_z_keys"])
    b.add_param("ParamBodyAngleX", *cfg["body_range"], 0.0, cfg["body_keys"])
    b.add_param("ParamBodyAngleY", *cfg["body_range"], 0.0, cfg["body_keys"])
    b.add_param("ParamBodyAngleZ", *cfg["body_range"], 0.0, cfg["body_keys"])
    b.add_param("ParamEyeLOpen", 0.0, 1.0, 1.0, cfg["eye_keys"])
    b.add_param("ParamEyeROpen", 0.0, 1.0, 1.0, cfg["eye_keys"])
    b.add_param("ParamMouthOpenY", 0.0, 1.0, 0.0, (0.0, 1.0))
    b.add_part("PartRoot", 0.0)

    # D_Body : root warp deformer (body turn + a bit of head-driven lean).
    # Cubism flips the y axis when a *root* deformer maps its local space into
    # model space, so the deformation has to be evaluated at the row mirrored
    # about the canvas centre (and the y part of the displacement flipped back).
    MIR = cfg["canvas_px"]

    def body_pos(st):
        bx = st.get("ParamBodyAngleX", 0.0)
        by = st.get("ParamBodyAngleY", 0.0)
        bz = st.get("ParamBodyAngleZ", 0.0)
        ax = st.get("ParamAngleX", 0.0)
        out = []
        for (px, py) in body_grid_px:
            ym = MIR - py                       # mirrored authoring row
            wx, wym = body_warp(px, ym, bx, by, bz)
            wx += 0.30 * ax                     # subtle follow of the head turn
            out.append(to_model(px + (wx - px), py - (wym - ym)))
        return out

    b.add_warp_deformer(id="DBody", rows=5, cols=5, grid=body_grid_model,
                        parent_part=0, parent_deformer=-1,
                        params=["ParamBodyAngleX", "ParamBodyAngleY", "ParamBodyAngleZ",
                                "ParamAngleX"],
                        pos_fn=body_pos)

    # D_Head : child of D_Body
    def head_pos(st):
        ax = st.get("ParamAngleX", 0.0)
        ay = st.get("ParamAngleY", 0.0)
        az = st.get("ParamAngleZ", 0.0)
        return [body_local(*head_warp(px, py, ax, ay, az))
                for (px, py) in head_grid_px]

    b.add_warp_deformer(id="DHead", rows=6, cols=6, grid=head_grid_local,
                        parent_part=0, parent_deformer=0,
                        params=["ParamAngleX", "ParamAngleY", "ParamAngleZ"],
                        pos_fn=head_pos)

    # ---- silhouette meshes
    b.add_art_mesh(id="Body", verts=[body_local(*p) for p in body_verts],
                   uvs=[(p[0] / AW, p[1] / AH) for p in body_verts], tris=body_tris,
                   draw_order=100, parent_part=0, parent_deformer=0)
    b.add_art_mesh(id="Head", verts=[head_local(*p) for p in head_verts],
                   uvs=[(p[0] / AW, p[1] / AH) for p in head_verts], tris=head_tris,
                   draw_order=200, parent_part=0, parent_deformer=1)

    # ---- mouth: the only art that cannot come from the picture, so it stays
    # invisible at rest and grows out of the mouth's own centre when it opens.
    my0, my1 = mbox[1], mbox[3]
    mcy = (my0 + my1) / 2.0

    def mouth_pos(st):
        r = float(st["ParamMouthOpenY"])
        s = min(1.0, r * cfg["mouth_grow"])
        return [head_local(px, mcy + (py - mcy) * s) for (px, py) in mv]

    def mouth_op(st):
        return min(1.0, float(st["ParamMouthOpenY"]) / cfg["mouth_fade"])

    b.add_art_mesh(id="MouthOpen", verts=[head_local(*p) for p in mv],
                   uvs=muvs, tris=mt, draw_order=350, parent_part=0,
                   parent_deformer=1, params=["ParamMouthOpenY"],
                   pos_fn=mouth_pos, opa_fn=mouth_op)

    # ---- eyes: the lid closes over the eye using the model's own skin.
    # The outer rows of the mesh stay pinned to the face, the rows inside the
    # eye collapse into a lash line and the rows of skin below stretch up, so
    # a closed eye is built from real skin taken from around the eye.
    for i, em in enumerate(eye_meshes):
        pid = "ParamEyeLOpen" if i == 0 else "ParamEyeROpen"
        rb = em["box"]
        y0 = float(rb[1])
        h = float(rb[3] - rb[1])
        te0, te1, tc = em["te0"], em["te1"], em["tc"]
        k = cfg["eye_closed_scale"]

        def lid_t(t, te0=te0, te1=te1, tc=tc, k=k):
            if t <= te0:
                return tc * (t / te0) if te0 > 1e-6 else 0.0
            off = tc + (te1 - te0) * k
            if t >= te1:
                return off + (1.0 - off) * (t - te1) / (1.0 - te1) \
                    if te1 < 1.0 - 1e-6 else 1.0
            return tc + (t - te0) * k

        def eye_pos(st, verts=em["verts"], cols=em["cols"], rows=em["rows"],
                    y0=y0, h=h, lid_t=lid_t, pid=pid):
            o = float(st[pid])
            out = []
            for n, (px, py) in enumerate(verts):
                t = (n // (cols + 1)) / float(rows)
                out.append(head_local(px, y0 + h * (t + (lid_t(t) - t) * (1.0 - o))))
            return out

        b.add_art_mesh(id=em["id"], verts=[head_local(*p) for p in em["verts"]],
                       uvs=em["uvs"], tris=em["tris"], draw_order=400 + i,
                       parent_part=0, parent_deformer=1,
                       params=[pid], pos_fn=eye_pos)

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
        ],
    }
    with open(os.path.join(outdir, "%s.model3.json" % name), "w") as f:
        json.dump(model3, f, indent=1)

    physics = {
        "Version": 3,
        "Meta": {
            "PhysicsSettingCount": 1,
            "TotalInputCount": 1,
            "TotalOutputCount": 1,
            "VertexCount": 2,
            "EffectiveForces": {"Gravity": {"X": 0, "Y": -1}, "Wind": {"X": 0, "Y": 0}},
            "PhysicsDictionary": [{"Id": "PhysicsSetting1", "Name": "BodyFollow"}],
        },
        "PhysicsSettings": [{
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
        }],
    }
    with open(os.path.join(outdir, "%s.physics3.json" % name), "w") as f:
        json.dump(physics, f, indent=1)

    cdi = {
        "Version": 3,
        "Parameters": [{"Id": p, "GroupId": "", "Name": p} for p in
                       ["ParamAngleX", "ParamAngleY", "ParamAngleZ",
                        "ParamBodyAngleX", "ParamBodyAngleY", "ParamBodyAngleZ",
                        "ParamEyeLOpen", "ParamEyeROpen", "ParamMouthOpenY"]],
        "ParameterGroups": [],
        "Parts": [{"Id": "PartRoot", "Name": "Root"}],
        "Drawables": [{"Id": i, "Name": i} for i in
                      ["Body", "Head", "MouthOpen", "Eye0", "Eye1"]],
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

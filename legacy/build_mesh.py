#!/usr/bin/env python3
"""Build a Live2D moc3 (v1) for the chibi VTuber: body + breathing torso + blink eyes + mouth crossfade."""
import json
import numpy as np
from PIL import Image

# ---------------------------------------------------------------- load inputs
cut = Image.open("cutout.png").convert("RGBA")
ALPHA = np.array(cut)[:, :, 3] > 0
H, W = ALPHA.shape
geom = json.load(open("geom.json"))
EYE_L = geom["eyes"][0]   # cx, cy, w, h
EYE_R = geom["eyes"][1]
MOUTH = geom["mouth"]     # cx, cy
NECK_Y = 640.0            # torso starts here (below chin)
BREATH_AMP = 6.0          # px at bottom
EYE_SQUASH = 0.82         # fraction of eye height closed
EYE_FADE = 45.0           # px below eye where squash fades to 0
MOUTH_BOX = (20, 25, 130, 85)  # where open mouth was drawn in texture corner


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


# ---------------------------------------------------------------- mesh builder
def row_runs(row_mask):
    """Runs of True in a 1-D bool array -> list of (start, end) inclusive."""
    runs = []
    x = 0
    n = len(row_mask)
    while x < n:
        if row_mask[x]:
            x0 = x
            while x + 1 < n and row_mask[x + 1]:
                x += 1
            runs.append((x0, x))
            x += 1
        else:
            x += 1
    return runs


def build_region(y_top, y_bot, row_step, col_step, x_lo=None, x_hi=None):
    """Row-strip mesh over the silhouette (optionally clipped to a rect).
    Returns verts [(x,y)], uvs [(u,v)], tris [(i,j,k)]."""
    x_lo = 0 if x_lo is None else int(x_lo)
    x_hi = W - 1 if x_hi is None else int(x_hi)
    verts, uvs, tris = [], [], []
    rows = []  # list of (y, [points]) where point = (x, vid)
    y = y_top
    while y <= y_bot:
        ypix = int(round(y))
        ypix = min(max(ypix, 0), H - 1)
        runs = row_runs(ALPHA[ypix, x_lo:x_hi + 1])
        pts = []
        for (a, b) in runs:
            a += x_lo
            b += x_lo
            if b - a < 1:
                pts.append((a, a))
                continue
            xs = list(range(a, b, col_step))
            if xs[-1] != b:
                xs.append(b)
            if xs[0] != a:
                xs = [a] + xs
            for x in xs:
                pts.append((x, len(verts)))
                verts.append((float(x), float(ypix)))
                uvs.append((x / W, 1.0 - ypix / H))
        rows.append((ypix, pts))
        y += row_step
    # triangulate between consecutive rows
    for ri in range(len(rows) - 1):
        yT, top = rows[ri]
        yB, bot = rows[ri + 1]
        if not top or not bot:
            continue
        # top runs: group consecutive top points with gap <= col_step
        truns = []
        cur = [top[0]]
        for p in top[1:]:
            if p[0] - cur[-1][0] <= col_step:
                cur.append(p)
            else:
                truns.append(cur)
                cur = [p]
        truns.append(cur)
        bot_x = np.array([p[0] for p in bot])
        for ti, run in enumerate(truns):
            A = run[0][1]
            B = run[-1][1]
            ax, bx = run[0][0], run[-1][0]
            if len(run) >= 2:
                sel = np.where((bot_x >= ax) & (bot_x <= bx))[0]
                sel = [s for s in sel]
                m = len(sel)
                if m == 1:
                    tris.append((A, B, bot[sel[0]][1]))
                elif m >= 2:
                    tris.append((A, B, bot[sel[m - 1]][1]))
                    for j in range(m - 1, 0, -1):
                        tris.append((A, bot[sel[j]][1], bot[sel[j - 1]][1]))
            else:
                # single top point (tip): V region between gaps
                pass
            # gap to next top run
            if ti + 1 < len(truns):
                next_run = truns[ti + 1]
                Bx = bx
                Cx = next_run[0][0]
                if Cx - Bx > 1:
                    sel = np.where((bot_x > Bx) & (bot_x < Cx))[0].tolist()
                    k = len(sel)
                    C = next_run[0][1]
                    if k == 1:
                        tris.append((B, C, bot[sel[0]][1]))
                    elif k >= 2:
                        for j in range(0, k - 1):
                            tris.append((B, bot[sel[j]][1], bot[sel[j + 1]][1]))
                        tris.append((B, bot[sel[k - 1]][1], C))
    return verts, uvs, tris


def eye_field(cx, cy, ew, eh, x, y):
    """Blink squash: displacement dy (negative = up) for a vertex at (x,y)."""
    eye_top = cy - eh / 2
    eye_bot = cy + eh / 2
    rx = ew / 2
    tx = np.abs(x - cx) / (rx + 35.0)
    inside = 1.0 - smoothstep(0.75, 1.0, tx)
    ty = (y - eye_top) / eh
    base = np.where(ty <= 1.0, np.clip(ty, 0, 1), 1.0 - smoothstep(1.0, 1.0 + EYE_FADE / eh, ty))
    base = np.clip(base, 0, 1)
    return -EYE_SQUASH * eh * base * inside


def build_mesh(name, verts, uvs, tris, kfs):
    """kfs: list of arrays (nv,2) — one per keyform (absolute positions)."""
    return {
        "name": name,
        "verts": np.array(verts, dtype=np.float32),
        "uvs": np.array(uvs, dtype=np.float32),
        "tris": np.array(tris, dtype=np.int32),
        "kfs": [np.asarray(k, dtype=np.float32) for k in kfs],
    }


# ---------------------------------------------------------------- regions
print("building body mesh...")
vb, ub, tb = build_region(104, 1023, 6, 14)
body = build_mesh("Body", vb, ub, tb, [np.array(vb, dtype=np.float32)])
print("body:", len(vb), "verts,", len(tb), "tris")

print("building torso mesh...")
vt, ut, tt = build_region(NECK_Y, 1023, 5, 10)
base_t = np.array(vt, dtype=np.float32)
yy = base_t[:, 1]
shift = BREATH_AMP * smoothstep(NECK_Y, NECK_Y + 120.0, yy)
torso1 = base_t.copy()
torso1[:, 1] += shift
torso = build_mesh("Torso", vt, ut, tt, [base_t, torso1])
print("torso:", len(vt), "verts,", len(tt), "tris, max shift", shift.max())

def eye_mesh(e, name):
    cx, cy, ew, eh = e["cx"], e["cy"], e["w"], e["h"]
    x_lo = cx - ew / 2 - 35
    x_hi = cx + ew / 2 + 35
    y_top = cy - eh / 2 - 25
    y_bot = cy + eh / 2 + 45
    v, u, t = build_region(y_top, y_bot, 5, 8, x_lo, x_hi)
    base = np.array(v, dtype=np.float32)
    dy = eye_field(cx, cy, ew, eh, base[:, 0], base[:, 1])
    closed = base.copy()
    closed[:, 1] += dy  # kf0 = closed (squashed)
    return build_mesh(name, v, u, t, [closed, base]), dy

print("building eye meshes...")
eyeL, dyL = eye_mesh(EYE_L, "EyeL")
eyeR, dyR = eye_mesh(EYE_R, "EyeR")
print("eyeL:", len(eyeL["verts"]), "verts, max squash", -dyL.min())
print("eyeR:", len(eyeR["verts"]), "verts, max squash", -dyR.min())

# mouth quad
print("building mouth mesh...")
mcx, mcy = MOUTH[0], MOUTH[1]
mw, mh = 120, 80
mverts = [
    (mcx - mw / 2, mcy - mh / 2),
    (mcx + mw / 2, mcy - mh / 2),
    (mcx + mw / 2, mcy + mh / 2),
    (mcx - mw / 2, mcy + mh / 2),
]
muv = [
    (MOUTH_BOX[0] / W, 1 - MOUTH_BOX[1] / H),
    (MOUTH_BOX[2] / W, 1 - MOUTH_BOX[1] / H),
    (MOUTH_BOX[2] / W, 1 - MOUTH_BOX[3] / H),
    (MOUTH_BOX[0] / W, 1 - MOUTH_BOX[3] / H),
]
mtris = [(0, 1, 2), (0, 2, 3)]
mbase = np.array(mverts, dtype=np.float32)
mouth = build_mesh("MouthOpen", mverts, muv, mtris, [mbase, mbase])

MESHES = [body, torso, eyeL, eyeR, mouth]
total_v = sum(len(m["verts"]) for m in MESHES)
total_t = sum(len(m["tris"]) for m in MESHES)
print("TOTAL:", total_v, "verts,", total_t, "tris (i16 max 65535 ok:", total_v < 65536, ")")
assert total_v < 65536

data = {}
for i, m in enumerate(MESHES):
    data[f"m{i}_v"] = m["verts"]
    data[f"m{i}_u"] = m["uvs"]
    data[f"m{i}_t"] = m["tris"]
    for k, kf in enumerate(m["kfs"]):
        data[f"m{i}_k{k}"] = kf
np.savez("mesh_data.npz", **data)
print("saved mesh_data.npz")

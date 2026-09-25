#!/usr/bin/env python3
"""Assemble model.moc3 (MOC3 v1) from mesh_data.npz using py-moc3 (validated round-trip on real model)."""
import numpy as np
from moc3 import Moc3

d = np.load("mesh_data.npz")
MESHES = []
i = 0
while True:
    key = f"m{i}_v"
    if key not in d:
        break
    vc = d[f"m{i}_v"].shape[0]
    tris = d[f"m{i}_t"]
    kfs = []
    k = 0
    while f"m{i}_k{k}" in d:
        kfs.append(d[f"m{i}_k{k}"])
        k += 1
    MESHES.append({
        "verts": d[f"m{i}_v"],
        "uvs": d[f"m{i}_u"],
        "tris": tris,
        "kfs": kfs,
    })
    i += 1
N = len(MESHES)
assert N == 5, N

NAMES = ["Body", "Torso", "EyeL", "EyeR", "MouthOpen"]
BINDINGS = [0, 1, 2, 3, 4]          # per mesh -> binding idx
DO = [0, 1, 2, 3, 4]                # draw order per mesh
OPA = [[1], [1, 1], [1, 1], [1, 1], [0, 1]]  # opacity per keyform

# keyform table (global order)
kf_list = []  # (mesh_idx, kf_idx, opa, do, pos_floats)
kf_begin = [0]
for mi in range(N):
    for ki in range(len(MESHES[mi]["kfs"])):
        kf_list.append((mi, ki))
    kf_begin.append(len(kf_list))
KF_COUNT = len(kf_list)

# counts
vc = [m["verts"].shape[0] for m in MESHES]
tric = [m["tris"].shape[0] for m in MESHES]
uvf = [2 * m["verts"].shape[0] for m in MESHES]
kff = [2 * m["verts"].shape[0] * len(m["kfs"]) for m in MESHES]

counts = [
    1,        # parts
    0,        # deformers
    0,        # warp deformers
    0,        # rotation deformers
    N,        # art meshes
    8,        # parameters
    1,        # part keyforms
    0,        # warp deformer keyforms
    0,        # rotation deformer keyforms
    KF_COUNT, # art mesh keyforms
    sum(kff), # keyform positions (floats)
    4,        # parameter binding indices (flat curve ids)
    5,        # keyform bindings
    4,        # parameter bindings (curves)
    8,        # keys (floats)
    sum(uvf), # uvs (floats)
    sum(tric),# position indices (i16)
    0, 0, 0, 0, 0, 0, 0,  # masks, draw order groups, objects, glues...
]

moc = Moc3.from_file("/home/user/l2d_refs/aier.moc3")
moc.header.version = 1
moc.counts = counts
moc.canvas.pixels_per_unit = 1.0
moc.canvas.origin_x = 512.0
moc.canvas.origin_y = 512.0
moc.canvas.canvas_width = 1024.0
moc.canvas.canvas_height = 1024.0
moc.canvas.canvas_flag = 0

def Z(n, f=False):
    return [0.0] * n if f else [0] * n

# ---- part
moc["part.runtime_space"] = Z(1)
moc["part.ids"] = ["Body"]
moc["part.keyform_binding_band_indices"] = [0]
moc["part.keyform_begin_indices"] = [0]
moc["part.keyform_counts"] = [1]
moc["part.visibles"] = [1]
moc["part.enables"] = [1]
moc["part.parent_part_indices"] = [-1]

# ---- deformers (all empty)
for sec in ["deformer.runtime_space", "deformer.ids", "deformer.keyform_binding_band_indices",
            "deformer.visibles", "deformer.enables", "deformer.parent_part_indices",
            "deformer.parent_deformer_indices", "deformer.types", "deformer.specific_indices",
            "warp_deformer.keyform_binding_band_indices", "warp_deformer.keyform_begin_indices",
            "warp_deformer.keyform_counts", "warp_deformer.vertex_counts", "warp_deformer.rows",
            "warp_deformer.cols",
            "rotation_deformer.keyform_binding_band_indices", "rotation_deformer.keyform_begin_indices",
            "rotation_deformer.keyform_counts", "rotation_deformer.base_angles"]:
    moc[sec] = []

# ---- art mesh
for r in range(4):
    moc[f"art_mesh.runtime_space_{r}"] = Z(N)
moc["art_mesh.ids"] = NAMES
moc["art_mesh.keyform_binding_band_indices"] = BINDINGS
moc["art_mesh.keyform_begin_indices"] = kf_begin[:-1]
moc["art_mesh.keyform_counts"] = [len(m["kfs"]) for m in MESHES]
moc["art_mesh.visibles"] = [1] * N
moc["art_mesh.enables"] = [1] * N
moc["art_mesh.parent_part_indices"] = [0] * N
moc["art_mesh.parent_deformer_indices"] = [-1] * N
moc["art_mesh.texture_indices"] = [0] * N
moc["art_mesh.drawable_flags"] = [4] * N  # double-sided
moc["art_mesh.position_index_counts"] = vc  # NOTE: py-moc3 name = actual vertex_counts (sot43)
moc["art_mesh.uv_begin_indices"] = np.cumsum([0] + uvf)[:-1].tolist()
moc["art_mesh.position_index_begin_indices"] = np.cumsum([0] + tric)[:-1].tolist()
moc["art_mesh.vertex_counts"] = tric      # NOTE: py-moc3 name = actual idx_len (sot46)
moc["art_mesh.mask_begin_indices"] = [0] * N
moc["art_mesh.mask_counts"] = [0] * N

# ---- parameters
PARAMS = [
    ("ParamAngleX", -30, 30, 0),
    ("ParamAngleY", -30, 30, 0),
    ("ParamAngleZ", -30, 30, 0),
    ("ParamEyeLOpen", 0, 1, 1),
    ("ParamEyeROpen", 0, 1, 1),
    ("ParamMouthOpenY", 0, 1, 0),
    ("ParamMouthForm", -1, 1, 0),
    ("ParamBreath", 0, 1, 0),
]
moc["parameter.runtime_space"] = Z(len(PARAMS))
moc["parameter.ids"] = [p[0] for p in PARAMS]
moc["parameter.min_values"] = [float(p[1]) for p in PARAMS]
moc["parameter.max_values"] = [float(p[2]) for p in PARAMS]
moc["parameter.default_values"] = [float(p[3]) for p in PARAMS]
moc["parameter.repeats"] = [0] * len(PARAMS)
moc["parameter.decimal_places"] = [2] * len(PARAMS)
# parameter -> bindings it drives (flat list of binding indices)
p_bind_flat = [2, 3, 4, 1]  # EyeL->B2, EyeR->B3, Mouth->B4, Breath->B1
p_bind = {3: (0, 1), 4: (1, 1), 5: (2, 1), 7: (3, 1)}
moc["parameter.keyform_binding_begin_indices"] = [p_bind[i][0] if i in p_bind else 0 for i in range(8)]
moc["parameter.keyform_binding_counts"] = [p_bind[i][1] if i in p_bind else 0 for i in range(8)]

# ---- keyforms (parts/warps/rotations empty)
moc["part_keyform.draw_orders"] = [1.0]
for sec in ["warp_deformer_keyform.opacities", "warp_deformer_keyform.keyform_position_begin_indices",
            "rotation_deformer_keyform.opacities", "rotation_deformer_keyform.angles",
            "rotation_deformer_keyform.origin_xs", "rotation_deformer_keyform.origin_ys",
            "rotation_deformer_keyform.scales", "rotation_deformer_keyform.reflect_xs",
            "rotation_deformer_keyform.reflect_ys"]:
    moc[sec] = []

kf_opa, kf_do, kf_pos_begin = [], [], []
pos_flat = []
off = 0
for (mi, ki) in kf_list:
    kf_opa.append(float(OPA[mi][ki]))
    kf_do.append(float(DO[mi]))
    kf_pos_begin.append(off)
    pos_flat.extend(MESHES[mi]["kfs"][ki].flatten().tolist())
    off += 2 * vc[mi]
moc["art_mesh_keyform.opacities"] = kf_opa
moc["art_mesh_keyform.draw_orders"] = kf_do
moc["art_mesh_keyform.keyform_position_begin_indices"] = kf_pos_begin
moc["keyform_position.xys"] = pos_flat

# ---- bindings / curves / keys
# binding B: list of curve ids
moc["keyform_binding_index.indices"] = [0, 1, 2, 3]      # B1->C0, B2->C1, B3->C2, B4->C3
moc["keyform_binding_band.begin_indices"] = [0, 0, 1, 2, 3]
moc["keyform_binding_band.counts"] = [0, 1, 1, 1, 1]
# curves C: key breakpoint lists (param values), 2 keys each
moc["keyform_binding.keys_begin_indices"] = [0, 2, 4, 6]
moc["keyform_binding.keys_counts"] = [2, 2, 2, 2]
moc["keys.values"] = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]

# ---- uv / indices
uv_flat = []
for m in MESHES:
    uv_flat.extend(m["uvs"].flatten().tolist())
moc["uv.xys"] = uv_flat
idx_flat = []
for m in MESHES:
    idx_flat.extend(m["tris"].flatten().tolist())
moc["position_index.indices"] = idx_flat

# ---- empty groups
moc["drawable_mask.art_mesh_indices"] = []
moc["draw_order_group.object_begin_indices"] = []
moc["draw_order_group.object_counts"] = []
moc["draw_order_group.object_total_counts"] = []
moc["draw_order_group.min_draw_orders"] = []
moc["draw_order_group.max_draw_orders"] = []
moc["draw_order_group_object.types"] = []
moc["draw_order_group_object.indices"] = []
moc["draw_order_group_object.group_indices"] = []
for sec in ["glue.runtime_space", "glue.ids", "glue.keyform_binding_band_indices",
            "glue.keyform_begin_indices", "glue.keyform_counts", "glue.art_mesh_index_as",
            "glue.art_mesh_index_bs", "glue.info_begin_indices", "glue.info_counts",
            "glue_info.weights", "glue_info.position_indices", "glue_keyform.intensities"]:
    moc[sec] = []

out = "live2d/model.moc3"
import os
os.makedirs("live2d", exist_ok=True)
moc.to_file(out)
print("wrote", out, "sizes:", len(moc.to_bytes()))
print("counts:", counts)

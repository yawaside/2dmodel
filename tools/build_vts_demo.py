#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ассеты интерактивного предпросмотра «как в VTube Studio» (папка vts/).

Готовит ``vts/pack`` — ровно те кусочки арта, из которых модель рисуется
в Live2D, плюс машинопитаемый манифест с геометрией деформеров:

    tiles/head_base.png   голова (кроп 150,60–875,700)
    tiles/body_base.png   корпус (кроп 150,700–875,1024)
    tiles/mouth_0..5.png  6 кадров рта   — тайлы атласа
    tiles/eye_{0,1}_*.png 5 кадров глаза x 2 глаза — тайлы атласа
    manifest.json         геометрия, боксы, параметры, деформеры

Браузерный рендер (vts/model.js) читает этот манифест и воспроизводит ту же
математику, что записана в .moc3: head_warp / body_warp / сеточные деформеры
из tools/build_model.py и mouth_blend / eye_blend / stack_opa. Cubism Core
не нужен — модель не меняется, меняется только способ показать её.

    python3 tools/rig.py                # атлас модели (если ещё не собран)
    python3 tools/build_vts_demo.py     # ассеты предпросмотра
    python3 tools/serve.py . 8000       # открыть /vts/
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

# Те же прямоугольники, что в tools/build_model.py (CFG)
HEAD_RECT = (150, 60, 875, 700)     # head_deformer + меши лица
BODY_RECT = (150, 700, 875, 1024)   # низ персонажа (mesh ниже cut_y)
MOUTH_VARIANTS = ("closed", "slight", "half", "A_open", "smile", "smirk")
EYE_VARIANTS = ("neutral", "half", "blink", "happy", "squint")

# параметры модели: id, min, max, default, группа, подпись, что делает в VTS
PARAMETERS = [
    ("ParamAngleX", -30.0, 30.0, 0.0, "Углы", "поворот головы (рысканье)",
     "FaceAngle Yaw — трекинг лица или мыши"),
    ("ParamAngleY", -30.0, 30.0, 0.0, "Углы", "наклон головы (тангаж)",
     "FaceAngle Pitch — трекинг лица или мыши"),
    ("ParamAngleZ", -30.0, 30.0, 0.0, "Углы", "крен головы",
     "FaceAngle Roll — трекинг лица или мыши"),
    ("ParamBodyAngleX", -10.0, 10.0, 0.0, "Корпус", "корпус по X",
     "обычно ведёт физика (PhysicsSetting1)"),
    ("ParamBodyAngleY", -10.0, 10.0, 0.0, "Корпус", "корпус по Y",
     "обычно ведёт физика (PhysicsSetting2)"),
    ("ParamBodyAngleZ", -10.0, 10.0, 0.0, "Корпус", "крен корпуса",
     "обычно ведёт физика"),
    ("ParamEyeLOpen", 0.0, 1.0, 1.0, "Глаза", "левый глаз закрыт ↔ открыт",
     "группа EyeBlink — авто-моргание VTube Studio"),
    ("ParamEyeROpen", 0.0, 1.0, 1.0, "Глаза", "правый глаз закрыт ↔ открыт",
     "группа EyeBlink — авто-моргание VTube Studio"),
    ("ParamEyeSmileL", 0.0, 1.0, 0.0, "Глаза", "левый глаз «аркой»",
     "экспрессии/физика (PhysicsSetting3)"),
    ("ParamEyeSmileR", 0.0, 1.0, 0.0, "Глаза", "правый глаз «аркой»",
     "экспрессии/физика (PhysicsSetting3)"),
    ("ParamMouthOpenY", 0.0, 1.0, 0.0, "Рот", "открытие рта",
     "группа LipSync — липсинк с микрофона"),
    ("ParamMouthForm", -1.0, 1.0, 0.0, "Рот", "−1 ухмылка … +1 улыбка",
     "экспрессии + лёгкий довод физикой (PhysicsSetting4)"),
    ("ParamBreath", 0.0, 1.0, 0.0, "Дыхание", "вдох/выдох",
     "группа Breath — включается в VTube Studio"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# доступ к атласу
# --------------------------------------------------------------------------- #

def ensure_atlas(atlas: Path, meta: Path, src: Path, geom: Path) -> dict:
    """Атлас модели + его метаданные (собирает, если build/ пуст)."""
    if atlas.exists() and meta.exists():
        return json.loads(meta.read_text(encoding="utf-8"))
    spec = importlib.util.spec_from_file_location("rig", ROOT / "tools" / "rig.py")
    rig = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rig)
    cfg = dict(eye_dark_thresh=150.0, eye_search_box=95, eye_pad=8,
               eye_margin_x=12, eye_margin_top=6, eye_margin_bottom=20,
               mouth_dark_thresh=115.0, mouth_search_w=85, mouth_search_h=34,
               mouth_pad_x=8, mouth_pad_y=8, mouth_lip=5,
               mouth_drops=(0, 10, 22, 34),
               atlas_width=2048, atlas_height=2048)
    atlas.parent.mkdir(parents=True, exist_ok=True)
    print("собираю атлас:", atlas)
    return rig.build_texture(str(src), str(atlas), str(meta),
                             json.loads(geom.read_text(encoding="utf-8")), cfg)


def crop(atlas: Image.Image, rect) -> Image.Image:
    return atlas.crop((int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])))


def save(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, optimize=True)


# --------------------------------------------------------------------------- #
# сборка
# --------------------------------------------------------------------------- #

def build(out: Path, atlas: Path, meta: Path, src: Path, geom: Path,
          step: int = 10) -> dict:
    meta_json = ensure_atlas(atlas, meta, src, geom)
    image = Image.open(atlas).convert("RGBA")
    base = Image.open(src).convert("RGBA")

    tiles = out / "tiles"
    tiles.mkdir(parents=True, exist_ok=True)

    # --- базовые кропы (голова и корпус) --------------------------------- #
    head = base.crop(HEAD_RECT)
    body = base.crop(BODY_RECT)
    save(head, tiles / "head_base.png")
    save(body, tiles / "body_base.png")

    # --- тайлы лица: те же пиксели, что в атласе .moc3 -------------------- #
    mouth_tiles = []
    for i, name in enumerate(MOUTH_VARIANTS):
        rect = meta_json["placed"]["mouth_%d" % i]["rect"]
        tile = crop(image, rect)
        save(tile, tiles / ("mouth_%d.png" % i))
        mouth_tiles.append("tiles/mouth_%d.png" % i)

    eye_tiles = {"L": [], "R": []}
    for side in (0, 1):
        key = "L" if side == 0 else "R"
        for vname in EYE_VARIANTS:
            rect = meta_json["placed"]["eye_%d_%s" % (side, vname)]["rect"]
            tile = crop(image, rect)
            save(tile, tiles / ("eye_%d_%s.png" % (side, vname)))
            eye_tiles[key].append("tiles/eye_%d_%s.png" % (side, vname))

    # --- параметры и группы VTube Studio из собранной модели -------------- #
    model3_path = ROOT / "dist" / "ChibiVT" / "ChibiVT.model3.json"
    groups = []
    if model3_path.exists():
        model3 = json.loads(model3_path.read_text(encoding="utf-8"))
        groups = [{"name": g["Name"], "ids": g["Ids"]}
                  for g in model3.get("Groups", [])]

    from build_model import CFG, V_LIMIT  # единственный источник геометрии рига

    manifest = {
        "schema_version": 1,
        "name": "ChibiVT · интерактивный предпросмотр VTube Studio",
        "kind": "vtube_studio_preview_matches_live2d_rig",
        "provenance": (
            "Ассеты — кропы cutout.png и тайлы атласа модели (tools/rig.py); "
            "деформация в браузере — порт head_warp/body_warp и сеточных "
            "деформеров из tools/build_model.py. Модель .moc3 не меняется."),
        "canvas": {
            "size": 1024,
            "head_rect": list(HEAD_RECT),
            "body_rect": list(BODY_RECT),
            "cut_y": float(CFG["cut_y"]),
        },
        "deformers": {
            "head": {"rect": [float(v) for v in CFG["head_rect"]],
                     "rows": 6, "cols": 6},
            "body": {"rect": [float(v) for v in CFG["body_rect"]],
                     "rows": 5, "cols": 5},
        },
        "rig": {
            "head_fade": CFG["head_fade"],
            "head_center": list(CFG["head_center"]),
            "neck_point": list(CFG["neck_point"]),
            "r_yaw": CFG["r_yaw"], "r_pitch": CFG["r_pitch"],
            "v_max": V_LIMIT,
            "yaw_scale": CFG["yaw_scale"], "pitch_scale": CFG["pitch_scale"],
            "roll_scale": CFG["roll_scale"],
            "body_shift_x": CFG["body_shift_x"], "body_shift_y": CFG["body_shift_y"],
            "body_roll_deg": CFG["body_roll_deg"], "body_hip": list(CFG["body_hip"]),
        },
        "browser_mesh": {
            "step": step,
            "dilate_steps": 2,
            "note": "сетка для warp-рендера в WebGL: шаг в пикселях исходника",
        },
        "boxes": {
            "head_rect": list(HEAD_RECT),
            "mouth_box": [float(v) for v in meta_json["mouth_box"]],
            "eye_boxes": [[float(v) for v in e["rig_box"]]
                          for e in meta_json["eyes"]],
        },
        "files": {
            "head": "tiles/head_base.png",
            "body": "tiles/body_base.png",
            "mouth_tiles": mouth_tiles,
            "eye_tiles": eye_tiles,
        },
        "mouth_variants": list(MOUTH_VARIANTS),
        "eye_variants": list(EYE_VARIANTS),
        "parameters": [
            {"id": pid, "min": lo, "max": hi, "default": dflt,
             "group": grp, "label": label, "vts": vts}
            for (pid, lo, hi, dflt, grp, label, vts) in PARAMETERS
        ],
        "vts_groups": groups,
        "source_sha256": {
            "cutout.png": sha256(src),
            "atlas": sha256(atlas),
        },
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "vts" / "pack")
    ap.add_argument("--atlas", type=Path, default=ROOT / "build" / "texture_atlas.png")
    ap.add_argument("--meta", type=Path, default=ROOT / "build" / "atlas.json")
    ap.add_argument("--src", type=Path, default=ROOT / "cutout.png")
    ap.add_argument("--geom", type=Path, default=ROOT / "geom.json")
    ap.add_argument("--mesh-step", type=int, default=10)
    args = ap.parse_args()

    manifest = build(args.out, args.atlas, args.meta, args.src, args.geom,
                     step=args.mesh_step)
    print("готово:", args.out)
    print("  голова %s, корпус %s" % (
        manifest["boxes"]["head_rect"], manifest["canvas"]["body_rect"]))
    print("  тайлов рта %d, глаз %d x2" % (
        len(manifest["files"]["mouth_tiles"]),
        len(manifest["files"]["eye_tiles"]["L"])))
    print("  параметров %d, групп VTS %d" % (
        len(manifest["parameters"]), len(manifest["vts_groups"])))


if __name__ == "__main__":
    main()

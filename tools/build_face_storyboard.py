#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Раскадровка глаз и рта (face storyboard) для липсинка в VTube Studio.

Собирает пакет `face_storyboard/pack/` из ТЕХ ЖЕ тайлов текстурного атласа,
которыми рисуется Live2D-модель (tools/rig.py -> build/texture_atlas.png),
поэтому каждый кадр раскадровки пиксель-в-пиксель совпадает с тем, что
рендерит .moc3 при соответствующих значениях параметров.

Выход (по образцу пакета ракурсов turnaround/):
  boards/mouth_board.png        лист рта: 6 кадров + шкала ParamMouthOpenY/Form
  boards/eye_board.png          лист глаз: 5 кадров + шкала ParamEyeLOpen/Smile
  frames/mouth/*.png            кадры рта на голове (нативное разрешение)
  frames/eyes/*.png             кадры глаз на голове
  frames/expressions/*.png      комбинации для проверки настроения
  mouth_sprite_sheet.png        сырые тайлы рта (как в атласе модели)
  eyes_sprite_sheet.png         сырые тайлы глаз
  lipsync_demo.gif              реальный блендинг цепочки липсинка (0->1->0)
  blink_cycle.gif               цикл моргания
  expressions.gif               улыбка <-> ухмылка (ParamMouthForm)
  storyboard.csv                кадр -> параметр -> диапазон (для риггера)
  manifest.json                 все зоны hold/transition из mouth_blend/eye_blend
  storyboard_boards.pdf         оба листа одним PDF
  ChibiVT_face_storyboard.psd   слой на каждый кадр (--psd, нужен psd-tools)
  ChibiVT_face_storyboard.zip   архив всего пакета

Использование:
  python3 tools/build_face_storyboard.py [--psd] [--out face_storyboard/pack]
Требует предварительно собранный риг: tools/rig.py (build/texture_atlas.png + atlas.json).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, HERE)
from build_model import CFG, mouth_blend, eye_blend, stack_opa  # noqa: E402

# ---- палитра в стиле пакета ракурсов -------------------------------------- #
BG = "#f8f6ef"
INK = "#28332d"
ACCENT = "#b05939"
HOLD = "#7ba05b"
TRANS = "#d9a441"
CARD = "#ffffff"
CARD_EDGE = "#deded4"
MUTED = "#6a716a"

MOUTH_FRAMES = [
    # (id, label, psd-источник, диапазон в котором кадр виден)
    ("closed", "Закрыт", "BASE (нейтральный рот)", "0.00 – 0.12"),
    ("slight", "Слегка приоткрыт", "M_slight_V1.png", "0.20 – 0.38"),
    ("half", "Полуоткрыт", "M_half_V2.png", "0.47 – 0.66"),
    ("A_open", "«А» широко открыт", "M_A_open_V3.png", "0.76 – 1.00"),
    ("smile", "Улыбка (зубы)", "M_I_grin_V5.png", "Form = +1"),
    ("smirk", "Ухмылка", "M_smirk_V8.png", "Form = −1"),
]

EYE_FRAMES = [
    # (id, label, psd-источник L/R, диапазон ParamEyeLOpen/R)
    ("neutral", "Нейтральный", "E_{side}_wide_V4.png", "0.50 – 1.00"),
    ("half", "Полуприкрыт", "E_{side}_half_V7.png", "0.20 – 0.50"),
    ("blink", "Закрыт (моргание)", "E_{side}_blink_V6.png", "0.00 – 0.20"),
    ("happy", "Улыбка (арка)", "E_{side}_happy_V5.png", "Smile = 1"),
    ("squint", "Прищур", "E_{side}_squint_V8.png", "Smile=1, веки≈0 (23%)"),
]

EXPRESSIONS = [
    ("smile_combo", "Улыбка", dict(eye="happy", mouth="smile")),
    ("smirk_combo", "Ухмылка", dict(eye="squint", mouth="smirk")),
    ("talk_A", "Говорит «А»", dict(eye="neutral", mouth="A_open")),
    ("blink_rest", "Моргает (покой)", dict(eye="blink", mouth="closed")),
]

# Зоны цепочки открытия рта: (от, до, тип, кадр A, кадр B)
MOUTH_TIMELINE = [
    (0.00, 0.12, "hold", "closed", None),
    (0.12, 0.20, "trans", "closed", "slight"),
    (0.20, 0.38, "hold", "slight", None),
    (0.38, 0.47, "trans", "slight", "half"),
    (0.47, 0.66, "hold", "half", None),
    (0.66, 0.76, "trans", "half", "A_open"),
    (0.76, 1.00, "hold", "A_open", None),
]

EYE_TIMELINE = [
    (0.00, 0.20, "trans", "blink", "half"),
    (0.20, 0.50, "trans", "half", "neutral"),
    (0.50, 1.00, "hold", "neutral", None),
]

MOUTH_IDX = {name: i for i, (name, *_r) in enumerate(MOUTH_FRAMES)}
EYE_IDX = {name: i for i, (name, *_r) in enumerate(EYE_FRAMES)}

HEAD_RECT = tuple(int(v) for v in CFG["head_rect"])  # (150, 60, 875, 700)


def font(size: int, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        raise RuntimeError("Нужны шрифты DejaVu Sans для русских подписей.")


class Tile:
    """Тайл атласа + его положение в пикселях исходника."""
    __slots__ = ("img", "box")

    def __init__(self, img: Image.Image, box: tuple):
        self.img = img
        self.box = box


# --------------------------------------------------------------------------- #
# доступ к ригу и композитинг состояний лица
# --------------------------------------------------------------------------- #

class Rig:
    """Доступ к тайлам атласа и композитинг состояний лица."""

    def __init__(self, atlas_path: Path, meta_path: Path, src_path: Path):
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.atlas = Image.open(atlas_path).convert("RGBA")
        self.base = Image.open(src_path).convert("RGBA")
        self.head = self.base.crop(HEAD_RECT)
        self.mbox = tuple(self.meta["mouth_box"])
        self.eye_boxes = [tuple(e["rig_box"]) for e in self.meta["eyes"]]

    def tile(self, key: str, box: tuple) -> Image.Image:
        rx0, ry0, rx1, ry1 = self.meta["placed"][key]["rect"]
        assert rx1 - rx0 == box[2] - box[0] and ry1 - ry0 == box[3] - box[1], key
        return self.atlas.crop((rx0, ry0, rx1, ry1))

    def mouth_tile(self, i: int) -> Image.Image:
        return self.tile(f"mouth_{i}", self.mbox)

    def eye_tile(self, side: int, vname: str) -> Image.Image:
        return self.tile(f"eye_{side}_{vname}", self.eye_boxes[side])

    def mouth_tiles(self) -> list[Tile]:
        return [Tile(self.mouth_tile(i), self.mbox) for i in range(6)]

    def eye_tiles(self, side: int) -> list[Tile]:
        return [Tile(self.eye_tile(side, n), self.eye_boxes[side])
                for n, *_r in EYE_FRAMES]

    # -- композитинг -------------------------------------------------------- #
    @staticmethod
    def _fade(tile: Image.Image, opacity: float) -> Image.Image:
        if opacity >= 0.999:
            return tile
        t = tile.copy()
        a = t.getchannel("A").point(lambda v: int(round(v * opacity)))
        t.putalpha(a)
        return t

    def paste_stack(self, head: Image.Image, tiles, weights):
        """Альфа-стек тайлов — ровно то, что делает stack_opa() в .moc3."""
        for i, tile in enumerate(tiles):
            opa = stack_opa(weights, i)
            if opa > 0.001:
                x = tile.box[0] - HEAD_RECT[0]
                y = tile.box[1] - HEAD_RECT[1]
                head.alpha_composite(self._fade(tile.img, opa), (x, y))

    def face(self, mouth: str | None = None, form: float = 0.0,
             eye: str | None = None, smile: float = 0.0) -> Image.Image:
        """Голова-кроп с наложенными состояниями (как рендерит модель)."""
        head = self.head.copy()
        if eye is not None:
            w = [0.0] * 5
            w[EYE_IDX[eye]] = 1.0
            if eye == "squint":  # как в модели: happy 77% + squint 23%
                w = [0.0, 0.0, 0.0, 0.77, 0.23]
            for side in (0, 1):
                self.paste_stack(head, self.eye_tiles(side), w)
        if mouth is not None:
            w = [0.0] * 6
            w[MOUTH_IDX[mouth]] = 1.0
            if mouth in ("closed", "slight", "half", "A_open") and form:
                # crossfade всей цепочки к форме — как в mouth_blend()
                k = abs(form)
                for i in range(4):
                    w[i] *= (1.0 - k)
                w[MOUTH_IDX["smile" if form > 0 else "smirk"]] = k
            total = sum(w) or 1.0
            self.paste_stack(head, self.mouth_tiles(), [x / total for x in w])
        return head


# --------------------------------------------------------------------------- #
# листы (boards)
# --------------------------------------------------------------------------- #

def zone_color(kind: str) -> str:
    return HOLD if kind == "hold" else TRANS


class Board:
    """Лист с ImageDraw и paste в одном объекте."""

    def __init__(self, w: int, h: int):
        self.img = Image.new("RGB", (w, h), BG)
        self.d = ImageDraw.Draw(self.img)

    def text(self, *a, **k):
        self.d.text(*a, **k)

    def rrect(self, *a, **k):
        self.d.rounded_rectangle(*a, **k)

    def paste(self, img, xy, mask=None):
        self.img.paste(img, xy, mask)

    def head_title(self, small: str, title: str, subtitle: str):
        self.text((48, 28), small, font=font(18, True), fill=ACCENT)
        self.text((48, 60), title, font=font(44, True), fill=INK)
        self.text((48, 122), subtitle, font=font(20), fill=MUTED)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.img.save(path)


def timeline_bar(board: Board, x: float, y: float, w: float, h: float,
                 timeline, title: str) -> float:
    """Шкала параметра с зонами hold/transition; возвращает новый y."""
    board.text((x, y - 4), title, font=font(17, True), fill=INK)
    y += 32
    small = font(15)
    for (a, b, kind, fa, fb) in timeline:
        zx = x + a * w
        zw = (b - a) * w
        board.d.rectangle((zx, y, zx + zw, y + h), fill=zone_color(kind), outline=BG)
        label = fa if fb is None else f"{fa}→{fb}"
        if zw > 96:
            board.text((zx + 8, y + h / 2 - 9), label,
                       font=small, fill="#ffffff" if kind == "hold" else INK)
        board.text((zx + 2, y + h + 6), f"{a:.2f}", font=small, fill=MUTED)
    board.text((x + w - 34, y + h + 6), "1.0", font=small, fill=MUTED)
    return y + h + 34


def frame_card(board: Board, head_img: Image.Image, title: str,
               lines: list[str], x: int, y: int, cw: int, ch: int, img_h: int):
    board.rrect((x, y, x + cw, y + ch), radius=16, fill=CARD, outline=CARD_EDGE, width=2)
    tw = cw - 24
    th = round(img_h * tw / head_img.width)
    thumb = head_img.resize((tw, th), Image.Resampling.LANCZOS)
    iy = y + 14
    # фон карточки под прозрачной головой — тёплый, не чёрный
    back = Image.new("RGBA", (tw + 8, th + 8), (239, 236, 226, 255))
    back.alpha_composite(thumb, (4, 4))
    board.rrect((x + 12, iy, x + cw - 12, iy + th + 8), radius=8, fill="#efece2")
    board.paste(back.convert("RGB"), (x + 12, iy))
    ty = iy + th + 14
    board.text((x + 14, ty), title, font=font(21, True), fill=INK)
    ty += 29
    for line in lines:
        board.text((x + 14, ty), line, font=font(15), fill=MUTED)
        ty += 22


def make_mouth_board(rig: Rig, heads: dict) -> Board:
    W, H = 1720, 1215
    board = Board(W, H)
    board.head_title("CHIBI / FACE STORYBOARD", "Раскадровка рта · липсинк",
                     "6 нарисованных кадров · ParamMouthOpenY 0…1 + ParamMouthForm −1…+1 · "
                     "кадры = слои PSD = тайлы атласа модели")

    cw, ch, img_h = 536, 366, 290
    for i, (fid, label, layer, rng) in enumerate(MOUTH_FRAMES):
        r, c = divmod(i, 3)
        x, y = 40 + c * (cw + 16), 176 + r * (ch + 14)
        param = "ParamMouthOpenY: %s" % rng if "Form" not in rng else "Значение: %s" % rng
        frame_card(board, heads["mouth"][fid], "%02d · %s" % (i, label),
                   ["PSD-слой: %s" % layer, param, "hold — кадр виден целиком"],
                   x, y, cw, ch, img_h)

    bx, bw = 40, W - 80
    ty = 176 + 2 * (ch + 14) + 6
    ty = timeline_bar(board, bx, ty, bw, 44, MOUTH_TIMELINE,
                      "Цепочка открытия рта · ParamMouthOpenY "
                      "(hold — зелёный, переход — жёлтый; одновременно смешано не более 2 кадров)")
    board.text((bx, ty - 4),
               "Форма рта · ParamMouthForm (−1 ухмылка … +1 улыбка): плавный crossfade всей цепочки",
               font=font(17, True), fill=INK)
    ty += 32
    grad = Image.new("RGB", (int(bw), 24))
    gp = grad.load()
    a, bcol = (176, 84, 57), (123, 160, 91)
    for gx in range(grad.width):
        t = gx / (grad.width - 1)
        col = tuple(round(a[k] + (bcol[k] - a[k]) * t) for k in range(3))
        for gy in range(24):
            gp[gx, gy] = col
    board.paste(grad, (int(bx), int(ty)))
    board.text((bx + 8, ty + 3), "−1  ухмылка", font=font(15, True), fill="#ffffff")
    board.text((bx + bw / 2 - 40, ty + 3), "0 нейтраль", font=font(15, True), fill="#ffffff")
    board.text((bx + bw - 136, ty + 3), "улыбка  +1", font=font(15, True), fill="#ffffff")
    ty += 52
    board.text((bx, ty), "Правило плавности: кадры открываются одной монотонной цепочкой "
                         "закрыт → слегка → полуоткрыт → «А».", font=font(19), fill=INK)
    board.text((bx, ty + 30), "Паузы (hold) длинные, переходы короткие — «двойного рта» "
                              "и полупрозрачных призраков нет.", font=font(19), fill=INK)
    return board


def make_eye_board(rig: Rig, heads: dict) -> Board:
    W, H = 1720, 1250
    board = Board(W, H)
    board.head_title("CHIBI / FACE STORYBOARD", "Раскадровка глаз · моргание и улыбка",
                     "5 состояний на каждый глаз · ParamEyeLOpen / ParamEyeROpen 0…1 + "
                     "ParamEyeSmileL/R 0…1")

    cw, ch, img_h = 320, 262, 320
    for i, (fid, label, layer, rng) in enumerate(EYE_FRAMES):
        x, y = 40 + i * (cw + 16), 176
        frame_card(board, heads["eyes"][fid], "%02d · %s" % (i, label),
                   ["PSD: %s / …R" % layer.format(side="L"), "Диапазон: %s" % rng],
                   x, y, cw, ch, img_h)

    bx, bw = 40, W - 80
    ty = 176 + ch + 26
    ty = timeline_bar(board, bx, ty, bw, 44, EYE_TIMELINE,
                      "Цепочка открытия глаз · ParamEyeLOpen / R "
                      "(моргание = быстрый проход 1 → 0 → 1, ~0.15 с)")
    board.text((bx, ty - 4),
               "ParamEyeSmileL/R: 0 → 1 плавно заменяет открытый глаз «аркой» (happy); "
               "при закрытых веках подмешивается прищур (≈23%).",
               font=font(17, True), fill=INK)
    ty += 42
    board.text((bx, ty), "Физика: микродвижения век, глаза прищуриваются при взгляде вверх; "
                         "кадр «широко раскрыт» не используется — покой = нейтраль.",
               font=font(19), fill=INK)
    ty += 52
    board.text((bx, ty), "Готовые выражения (глаза + рот):", font=font(24, True), fill=INK)
    ty += 46
    cw2, ch2, img_h2 = 400, 252, 300
    for i, (fid, label, spec) in enumerate(EXPRESSIONS):
        x = 40 + i * (cw2 + 16)
        frame_card(board, heads["expr"][fid], label,
                   ["глаза: %s" % spec["eye"], "рот: %s" % spec["mouth"]],
                   x, ty, cw2, ch2, img_h2)
    return board


# --------------------------------------------------------------------------- #
# GIF из реального блендинга модели
# --------------------------------------------------------------------------- #

def _gif_page(head: Image.Image, size: int) -> Image.Image:
    head = head.copy()
    head.thumbnail((size, size), Image.Resampling.LANCZOS)
    page = Image.new("RGB", (size, size), BG)
    page.paste(head, ((size - head.width) // 2, (size - head.height) // 2), head)
    return page


def make_gifs(rig: Rig, out: Path, size: int = 380):
    # липсинк: реальный проход ParamMouthOpenY 0→1→0 + улыбка/ухмылка
    seq = [0.0, 0.06, 0.16, 0.30, 0.44, 0.58, 0.72, 0.90, 1.0, 0.88, 0.70,
           0.55, 0.42, 0.30, 0.18, 0.08, 0.0, 0.0,
           0.25, 0.50, 0.75, 0.95, 0.70, 0.40, 0.15, 0.0]
    forms = [0.0] * 18 + [0.8] * 4 + [0.0] * 2 + [-0.9] + [0.0]
    frames = []
    for o, f in zip(seq, forms):
        head = rig.head.copy()
        rig.paste_stack(head, rig.mouth_tiles(), mouth_blend(o, f))
        frames.append(_gif_page(head, size))
    frames[0].save(out / "lipsync_demo.gif", save_all=True, append_images=frames[1:],
                   duration=95, loop=0, disposal=2)

    # моргание: ParamEyeLOpen/R 1 → 0 → 1
    frames = []
    for o in [1, 1, 0.75, 0.45, 0.12, 0.0, 0.12, 0.45, 0.8, 1, 1, 1]:
        head = rig.head.copy()
        for side in (0, 1):
            rig.paste_stack(head, rig.eye_tiles(side), eye_blend(o, 0.0))
        frames.append(_gif_page(head, size))
    frames[0].save(out / "blink_cycle.gif", save_all=True, append_images=frames[1:],
                   duration=70, loop=0, disposal=2)

    # улыбка <-> ухмылка (ParamMouthForm), глаза следуют как в физике модели
    frames = []
    for f in [0, 0.3, 0.6, 0.85, 1, 1, 0.7, 0.3, 0, -0.4, -0.8, -1, -1, -0.6, -0.2, 0]:
        head = rig.head.copy()
        rig.paste_stack(head, rig.mouth_tiles(), mouth_blend(0.14, f))
        for side in (0, 1):
            rig.paste_stack(head, rig.eye_tiles(side), eye_blend(1.0, max(0.0, f)))
        frames.append(_gif_page(head, size))
    frames[0].save(out / "expressions.gif", save_all=True, append_images=frames[1:],
                   duration=110, loop=0, disposal=2)


# --------------------------------------------------------------------------- #
# спрайт-листы, CSV, манифест, PSD, архив
# --------------------------------------------------------------------------- #

def sprite_sheet(tiles: list[Image.Image], labels: list[str], cols: int,
                 title: str) -> Image.Image:
    tw = max(t.width for t in tiles)
    th = max(t.height for t in tiles)
    label_h = 28
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (tw + 8) + 8, 44 + rows * (th + label_h + 8)),
                      BG)
    d = ImageDraw.Draw(sheet)
    d.text((8, 8), title, font=font(20, True), fill=INK)
    for i, (t, lab) in enumerate(zip(tiles, labels)):
        r, c = divmod(i, cols)
        x, y = 8 + c * (tw + 8), 44 + r * (th + label_h + 8)
        sheet.paste(t, (x, y), t)
        d.rectangle((x, y, x + t.width, y + t.height), outline=CARD_EDGE, width=1)
        d.text((x + 2, y + th + 5), lab, font=font(15), fill=MUTED)
    return sheet


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_psd(out: Path, heads: dict):
    from psd_tools import PSDImage
    from psd_tools.api.layers import Group, PixelLayer
    w = HEAD_RECT[2] - HEAD_RECT[0]
    h = HEAD_RECT[3] - HEAD_RECT[1]

    def on_bg(img: Image.Image) -> Image.Image:
        """Кадр на тёплом фоне, RGB без альфы — PSD получается вдвое меньше."""
        flat = Image.new("RGB", img.size, (239, 236, 226))
        flat.paste(img, (0, 0), img)
        return flat

    psd = PSDImage.new("RGB", (w, h))
    PixelLayer.frompil(on_bg(heads["base"]), psd, name="00_BASE_neutral")
    g_mouth = Group.new(psd, name="01_MOUTH_FRAMES_ParamMouthOpenY")
    for i, (fid, *_r) in enumerate(MOUTH_FRAMES):
        lay = PixelLayer.frompil(on_bg(heads["mouth"][fid]), g_mouth, name=f"M{i:02d}_{fid}")
        lay.visible = i == 0
    g_eyes = Group.new(psd, name="02_EYE_FRAMES_ParamEyeLOpenR")
    for i, (fid, *_r) in enumerate(EYE_FRAMES):
        lay = PixelLayer.frompil(on_bg(heads["eyes"][fid]), g_eyes, name=f"E{i:02d}_{fid}")
        lay.visible = False
    g_expr = Group.new(psd, name="03_EXPRESSIONS_combo")
    for i, (fid, *_r) in enumerate(EXPRESSIONS):
        lay = PixelLayer.frompil(on_bg(heads["expr"][fid]), g_expr, name=f"X{i:02d}_{fid}")
        lay.visible = False
    g_mouth.visible = True
    psd.save(out / "ChibiVT_face_storyboard.psd")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_small(img: Image.Image, path: Path):
    """PNG-8 (палитра 256) — кадры в 5-6 раз легче без заметной потери качества."""
    img.quantize(colors=256, method=Image.FASTOCTREE).save(path, optimize=True)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def build(out: Path, psd: bool, atlas: Path, meta: Path, src: Path) -> dict:
    for sub in ("boards", "frames/mouth", "frames/eyes", "frames/expressions"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    rig = Rig(atlas, meta, src)

    # --- кадры на голове --------------------------------------------------- #
    heads = {"mouth": {}, "eyes": {}, "expr": {}, "base": rig.head.copy()}
    for fid, *_r in MOUTH_FRAMES:
        img = rig.face(mouth=fid)
        heads["mouth"][fid] = img
        save_small(img, out / "frames" / "mouth" / ("m%02d_%s.png" % (MOUTH_IDX[fid], fid)))
    for fid, *_r in EYE_FRAMES:
        img = rig.face(eye=fid)
        heads["eyes"][fid] = img
        save_small(img, out / "frames" / "eyes" / ("e%02d_%s.png" % (EYE_IDX[fid], fid)))
    for i, (fid, _label, spec) in enumerate(EXPRESSIONS):
        img = rig.face(eye=spec["eye"], mouth=spec["mouth"])
        heads["expr"][fid] = img
        save_small(img, out / "frames" / "expressions" / ("x%02d_%s.png" % (i, fid)))

    # --- листы -------------------------------------------------------------- #
    mouth_board = make_mouth_board(rig, heads)
    eye_board = make_eye_board(rig, heads)
    mouth_board.save(out / "boards" / "mouth_board.png")
    eye_board.save(out / "boards" / "eye_board.png")
    mouth_board.img.save(out / "storyboard_boards.pdf", save_all=True,
                         append_images=[eye_board.img], resolution=150)

    # --- спрайт-листы ------------------------------------------------------- #
    sprite_sheet([rig.mouth_tile(i) for i in range(6)],
                 ["%d %s" % (i, fid) for i, (fid, *_r) in enumerate(MOUTH_FRAMES)],
                 3, "Mouth tiles — как в атласе модели (mouth_0…5)"
                 ).save(out / "mouth_sprite_sheet.png")
    eye_tiles, eye_labels = [], []
    for side, sname in ((0, "L"), (1, "R")):
        for fid, *_r in EYE_FRAMES:
            eye_tiles.append(rig.eye_tile(side, fid))
            eye_labels.append("%s %s" % (sname, fid))
    sprite_sheet(eye_tiles, eye_labels, 5,
                 "Eye tiles — как в атласе модели (eye_0_*, eye_1_*)"
                 ).save(out / "eyes_sprite_sheet.png")

    # --- тайлы для интерактивного просмотрщика /storyboard/ ----------------- #
    tiles_dir = out / "tiles"
    tiles_dir.mkdir(exist_ok=True)
    for i in range(6):
        rig.mouth_tile(i).save(tiles_dir / ("mouth_%d.png" % i), optimize=True)
    for side in (0, 1):
        for fid, *_r in EYE_FRAMES:
            rig.eye_tile(side, fid).save(
                tiles_dir / ("eye_%d_%s.png" % (side, fid)), optimize=True)
    rig.head.save(tiles_dir / "head_base.png", optimize=True)

    # --- GIF ---------------------------------------------------------------- #
    make_gifs(rig, out)

    # --- CSV ---------------------------------------------------------------- #
    rows = []
    for i, (fid, label, layer, rng) in enumerate(MOUTH_FRAMES):
        rows.append(dict(group="mouth", frame=i, label=label,
                         file="frames/mouth/m%02d_%s.png" % (i, fid),
                         psd_layer=layer,
                         parameter="ParamMouthOpenY" if "Form" not in rng else "ParamMouthForm",
                         range=rng))
    for i, (fid, label, layer, rng) in enumerate(EYE_FRAMES):
        rows.append(dict(group="eyes", frame=i, label=label,
                         file="frames/eyes/e%02d_%s.png" % (i, fid),
                         psd_layer=layer.format(side="L") + " + " + layer.format(side="R"),
                         parameter="ParamEyeLOpen/R" if "Smile" not in rng and "%" not in rng
                         else "ParamEyeSmileL/R",
                         range=rng))
    for i, (fid, label, spec) in enumerate(EXPRESSIONS):
        rows.append(dict(group="expressions", frame=i, label=label,
                         file="frames/expressions/x%02d_%s.png" % (i, fid),
                         psd_layer="combo", parameter="combo",
                         range="eyes=%s mouth=%s" % (spec["eye"], spec["mouth"])))
    write_csv(out / "storyboard.csv", rows)

    # --- манифест ----------------------------------------------------------- #
    manifest = {
        "schema_version": 1,
        "name": "ChibiVT · Раскадровка глаз и рта (липсинк)",
        "kind": "face_storyboard_matches_live2d_rig",
        "canvas": {"head_rect": list(HEAD_RECT), "source": "cutout.png"},
        "provenance": ("Кадры собраны из тайлов текстурного атласа Live2D-модели "
                       "(tools/rig.py), поэтому совпадают с рендером .moc3 один в один."),
        "parameters": {
            "ParamMouthOpenY": {"min": 0, "max": 1, "default": 0,
                                "keys": list(CFG["mouth_keys"])},
            "ParamMouthForm": {"min": -1, "max": 1, "default": 0,
                               "keys": list(CFG["mouth_form_keys"])},
            "ParamEyeLOpen": {"min": 0, "max": 1, "default": 1,
                              "keys": list(CFG["eye_open_keys"])},
            "ParamEyeROpen": {"min": 0, "max": 1, "default": 1,
                              "keys": list(CFG["eye_open_keys"])},
            "ParamEyeSmileL": {"min": 0, "max": 1, "default": 0,
                               "keys": list(CFG["eye_smile_keys"])},
            "ParamEyeSmileR": {"min": 0, "max": 1, "default": 0,
                               "keys": list(CFG["eye_smile_keys"])},
        },
        "mouth_timeline": [
            {"from": a, "to": b, "kind": k, "frame_a": fa, "frame_b": fb}
            for (a, b, k, fa, fb) in MOUTH_TIMELINE],
        "eye_timeline": [
            {"from": a, "to": b, "kind": k, "frame_a": fa, "frame_b": fb}
            for (a, b, k, fa, fb) in EYE_TIMELINE],
        "mouth_frames": [
            {"id": fid, "index": i, "label": label, "psd_layer": layer,
             "range": rng, "file": "frames/mouth/m%02d_%s.png" % (i, fid)}
            for i, (fid, label, layer, rng) in enumerate(MOUTH_FRAMES)],
        "eye_frames": [
            {"id": fid, "index": i, "label": label, "psd_layer": layer,
             "range": rng, "file": "frames/eyes/e%02d_%s.png" % (i, fid)}
            for i, (fid, label, layer, rng) in enumerate(EYE_FRAMES)],
        "expressions": [
            {"id": fid, "label": label, "eyes": spec["eye"], "mouth": spec["mouth"],
             "file": "frames/expressions/x%02d_%s.png" % (i, fid)}
            for i, (fid, label, spec) in enumerate(EXPRESSIONS)],
        "sprite_sheets": [
            {"file": "mouth_sprite_sheet.png", "tiles": 6, "columns": 3},
            {"file": "eyes_sprite_sheet.png", "tiles": 10, "columns": 5},
        ],
        "interactive": {
            "head": "tiles/head_base.png",
            "mouth_tiles": ["tiles/mouth_%d.png" % i for i in range(6)],
            "eye_tiles": {
                "L": ["tiles/eye_0_%s.png" % fid for fid, *_r in EYE_FRAMES],
                "R": ["tiles/eye_1_%s.png" % fid for fid, *_r in EYE_FRAMES],
            },
            "boxes": {
                "head_rect": list(HEAD_RECT),
                "mouth_box": [int(v) for v in rig.mbox],
                "eye_boxes": [[int(v) for v in b] for b in rig.eye_boxes],
            },
            "note": ("Тайлы и геометрия для покадрового блендинга в браузере — "
                     "та же математика, что в .moc3 (mouth_blend/eye_blend/stack_opa)."),
        },
        "animations": [
            {"file": "lipsync_demo.gif",
             "what": "реальный блендинг ParamMouthOpenY 0→1→0 + улыбка/ухмылка"},
            {"file": "blink_cycle.gif", "what": "цикл моргания ParamEyeLOpen/R 1→0→1"},
            {"file": "expressions.gif", "what": "ParamMouthForm −1…+1, глаза следуют"},
        ],
        "downloads": {
            "boards": "boards/", "pdf": "storyboard_boards.pdf",
            "csv": "storyboard.csv", "psd": "ChibiVT_face_storyboard.psd",
            "zip": "ChibiVT_face_storyboard.zip",
        },
        "source_sha256": {"cutout.png": sha256(src), "atlas": sha256(atlas)},
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if psd:
        make_psd(out, heads)

    # --- README пакета ------------------------------------------------------ #
    readme = ROOT / "face_storyboard" / "README.md"
    if readme.exists():
        (out / "README.md").write_text(readme.read_text(encoding="utf-8"),
                                       encoding="utf-8")

    # --- архив -------------------------------------------------------------- #
    files = [p for p in sorted(out.rglob("*")) if p.is_file()
             and p.name != "ChibiVT_face_storyboard.zip"]
    with zipfile.ZipFile(out / "ChibiVT_face_storyboard.zip", "w",
                         zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, "ChibiVT_face_storyboard/" + str(p.relative_to(out)))
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "storyboard" / "pack")
    ap.add_argument("--psd", action="store_true", help="собрать PSD со слоем на каждый кадр")
    ap.add_argument("--atlas", type=Path, default=ROOT / "build" / "texture_atlas.png")
    ap.add_argument("--meta", type=Path, default=ROOT / "build" / "atlas.json")
    ap.add_argument("--src", type=Path, default=ROOT / "cutout.png")
    args = ap.parse_args()
    if not (args.atlas.exists() and args.meta.exists()):
        sys.exit("Нет build/texture_atlas.png или build/atlas.json — "
                 "сначала запустите: python3 tools/rig.py")
    manifest = build(args.out.resolve(), args.psd, args.atlas, args.meta, args.src)
    print("Кадров рта: %d, глаз: %d, выражений: %d" % (
        len(manifest["mouth_frames"]), len(manifest["eye_frames"]),
        len(manifest["expressions"])))
    print("Готово: %s" % args.out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build the reference pack from AI-authored sheets; does not change the Live2D rig.

Usage: python3 tools/build_turnaround.py [--output turnaround/pack] [--psd]
Dependencies: Pillow, numpy; optional psd-tools for the layered reference PSD.
The source artwork is committed; the crops, labels and metadata are reproducible.
"""
from __future__ import annotations

import argparse
from collections import deque
import csv
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / 'turnaround' / 'source'
CANVAS = (768, 1024)
FEET_Y = 974
BG = '#f8f6ef'
INK = '#28332d'
ACCENT = '#b05939'

# The image model did not obey all angle labels. These are based on VISUAL
# inspection of the drawings, not the labels it baked into the images.
VIEWS = [
    ('front', 0, 'Спереди', 'front_to_side.png', (42, 75, 356, 750), False),
    ('front_left', 45, 'Спереди ¾ · влево', 'front_to_side.png', (539, 75, 850, 750), True),
    ('profile_left', 90, 'Профиль · влево', 'front_to_side.png', (996, 75, 1316, 750), False),
    ('back_left', 135, 'Сзади ¾ · влево', 'back_views.png', (58, 75, 382, 750), True),
    ('back', 180, 'Сзади', 'back_views.png', (536, 75, 842, 750), False),
    ('back_right', 225, 'Сзади ¾ · вправо', 'back_views.png', (58, 75, 382, 750), False),
    ('profile_right', 270, 'Профиль · вправо', 'front_to_side.png', (996, 75, 1316, 750), True),
    ('front_right', 315, 'Спереди ¾ · вправо', 'front_to_side.png', (539, 75, 850, 750), False),
]


def font(size: int, bold: bool = False):
    name = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        raise RuntimeError('Install DejaVu Sans fonts to render Russian labels.')


def cutout(source: Image.Image, box: tuple[int, ...]) -> Image.Image:
    """Remove border-connected near-ivory background, not cream areas in art."""
    crop = source.convert('RGB').crop(box)
    rgb = np.asarray(crop).copy()
    h, w = rgb.shape[:2]
    background = np.all(rgb > 222, axis=2)
    # The sheets contain a 1px ground rule at y=735. Remove only its isolated
    # horizontal extensions; preserve boot silhouettes above/below the rule.
    gy = 735 - box[1]
    if 4 <= gy < h - 5:
        support = (~background[max(0, gy - 40):gy - 3]).any(axis=0) | (~background[gy + 3:gy + 6]).any(axis=0)
        background[gy - 1:gy + 3, ~support] = True
    outside = np.zeros((h, w), dtype=bool)
    queue = deque()
    for x in range(w):
        for y in (0, h - 1):
            if background[y, x] and not outside[y, x]:
                outside[y, x] = True
                queue.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if background[y, x] and not outside[y, x]:
                outside[y, x] = True
                queue.append((y, x))
    while queue:
        y, x = queue.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and background[ny, nx] and not outside[ny, nx]:
                outside[ny, nx] = True
                queue.append((ny, nx))
    alpha = np.where(outside, 0, 255).astype('uint8')
    # Clear RGB in transparent pixels to avoid ivory fringes on resizing.
    rgb[outside] = 0
    result = Image.fromarray(np.dstack((rgb, alpha)))
    bbox = result.getbbox()
    if not bbox:
        raise ValueError('Empty character crop')
    return result.crop(bbox)


def normalize(figure: Image.Image) -> Image.Image:
    height = 880
    figure = figure.resize((round(figure.width * height / figure.height), height), Image.Resampling.LANCZOS)
    canvas = Image.new('RGBA', CANVAS)
    canvas.alpha_composite(figure, ((CANVAS[0] - figure.width) // 2, FEET_Y - height))
    return canvas


def make_board(frames: list[dict], out: Path):
    w, h = 1600, 1520
    board = Image.new('RGB', (w, h), BG)
    d = ImageDraw.Draw(board)
    d.text((48, 28), 'CHIBI / TURNAROUND', font=font(18, True), fill=ACCENT)
    d.text((48, 63), 'Персонаж со всех сторон', font=font(44, True), fill=INK)
    d.text((48, 126), '8 опорных кадров · шаг 45° · единый холст 768 × 1024 px', font=font(21), fill=INK)
    for i, frame in enumerate(frames):
        x, y = 40 + (i % 4) * 390, 192 + (i // 4) * 598
        d.rounded_rectangle((x, y, x + 370, y + 572), radius=18, fill='#ffffff', outline='#deded4', width=2)
        image = Image.open(out / frame['file']).convert('RGBA').resize((345, 460), Image.Resampling.LANCZOS)
        board.paste(image, (x + 12, y + 31), image)
        d.text((x + 20, y + 17), f"{i + 1:02d} / {frame['angle']:03d}°", font=font(16, True), fill=ACCENT)
        d.text((x + 20, y + 500), frame['label'], font=font(20, True), fill=INK)
        d.text((x + 20, y + 534), 'Зеркальный черновик' if frame['mirrored'] else 'AI-референс', font=font(16), fill='#6a716a')
    d.text((48, 1415), 'Референсы, не готовый 360° риг. Спина и ноги — предложенный дизайн.', font=font(20), fill=INK)
    d.text((48, 1451), 'Перед нарезкой согласовать гарнитуру, узел платка и подсумки между ракурсами.', font=font(19), fill='#6a716a')
    board.save(out / 'turnaround_board.png')
    return board


def make_heads(out: Path) -> tuple[list[dict], Image.Image]:
    source = Image.open(SOURCES / 'head_angles.png').convert('RGB')
    board = Image.new('RGB', (1200, 1460), BG)
    d = ImageDraw.Draw(board)
    d.text((40, 30), 'HEAD / KEYFORMS', font=font(18, True), fill=ACCENT)
    d.text((40, 68), 'Сетка поворотов головы', font=font(36, True), fill=INK)
    d.text((40, 124), 'Целевые ключи X/Y · углы приблизительные, не калиброваны', font=font(20), fill='#6a716a')
    heads = []
    for row, py in enumerate((30, 0, -30)):
        for col, px in enumerate((-30, 0, 30)):
            i = row * 3 + col
            # Leave grid lines behind. Each image includes the opaque reference
            # background, intentionally unlike the transparent body cutouts.
            box = (round(col * source.width / 3) + 5, round(row * source.height / 3) + 5,
                   round((col + 1) * source.width / 3) - 5, round((row + 1) * source.height / 3) - 5)
            head = source.crop(box).resize((512, 512), Image.Resampling.LANCZOS)
            name = f'heads/{i + 1:02d}_x{px:+d}_y{py:+d}.png'
            head.save(out / name)
            heads.append({'id': f'head_{i+1:02d}', 'label': f'X {px:+d} / Y {py:+d}',
                          'file': name, 'target_parameters': {'ParamAngleX': px, 'ParamAngleY': py},
                          'calibrated': False, 'background': 'opaque'})
            x, y = 40 + col * 380, 185 + row * 395
            board.paste(head.resize((360, 360), Image.Resampling.LANCZOS), (x, y))
            d.text((x + 12, y + 366), f'{i+1:02d}    X {px:+d} / Y {py:+d}', font=font(18, True), fill=INK)
    d.text((40, 1395), 'Сверить перспективу век, козырька и микрофона перед созданием мешей.', font=font(20), fill=INK)
    board.save(out / 'head_board.png')
    return heads, board


def make_psd(frames: list[dict], heads: list[dict], out: Path):
    try:
        from psd_tools import PSDImage
        from psd_tools.api.layers import Group, PixelLayer
    except ImportError:
        raise RuntimeError('--psd requires psd-tools: pip install psd-tools')
    psd = PSDImage.new('RGBA', CANVAS)
    bg = PixelLayer.frompil(Image.new('RGBA', CANVAS, BG), psd, name='00_Background_optional')
    bg.visible = False
    bodies = Group.new(psd, name='01_BODY_REFERENCES_NOT_CUT_PARTS')
    for i, frame in enumerate(frames):
        with Image.open(out / frame['file']) as image:
            name = f"{i+1:02d}_{frame['angle']:03d}_{frame['id']}" + ('_MIRROR_DRAFT' if frame['mirrored'] else '')
            layer = PixelLayer.frompil(image, bodies, name=name)
            layer.visible = i == 0
    head_group = Group.new(psd, name='02_HEAD_REFERENCES_TARGET_XY')
    for head in heads:
        with Image.open(out / head['file']) as image:
            layer = PixelLayer.frompil(image.convert('RGBA'), head_group, name=head['label'], left=128, top=94)
            layer.visible = False
    head_group.visible = False
    psd.save(out / 'ChibiVT_turnaround_references.psd')


def build(output: Path, psd: bool = False) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    (output / 'frames').mkdir(exist_ok=True)
    (output / 'heads').mkdir(exist_ok=True)
    frames = []
    images = []
    for i, (slug, angle, label, src, box, mirrored) in enumerate(VIEWS):
        with Image.open(SOURCES / src) as source:
            figure = cutout(source, box)
        if mirrored:
            figure = ImageOps.mirror(figure)
        canvas = normalize(figure)
        name = f'frames/{i+1:02d}_{angle:03d}_{slug}.png'
        canvas.save(output / name)
        frames.append({'id': slug, 'angle': angle, 'label': label, 'file': name,
                       'mirrored': mirrored, 'source': src, 'source_crop': list(box),
                       'status': 'mirror_draft' if mirrored else 'ai_reference',
                       'duration_ms': 500, 'pivot': [384, FEET_Y]})
        images.append(canvas)
    board = make_board(frames, output)
    heads, head_board = make_heads(output)
    sheet = Image.new('RGBA', (CANVAS[0] * 4, CANVAS[1] * 2))
    for i, image in enumerate(images):
        sheet.alpha_composite(image, ((i % 4) * CANVAS[0], (i // 4) * CANVAS[1]))
    sheet.save(output / 'body_sprite_sheet.png')
    animation = []
    for i, image in enumerate(images):
        page = Image.new('RGB', (384, 552), BG)
        thumb = image.resize((384, 512), Image.Resampling.LANCZOS)
        page.paste(thumb, (0, 0), thumb)
        d = ImageDraw.Draw(page)
        d.text((14, 514), f"{frames[i]['angle']:03d}° · " + ('MIRROR DRAFT' if frames[i]['mirrored'] else 'AI REFERENCE'), font=font(15), fill=INK)
        animation.append(page)
    animation[0].save(output / 'turnaround.gif', save_all=True, append_images=animation[1:], duration=500, loop=0, disposal=2)
    board.save(output / 'reference_boards.pdf', save_all=True, append_images=[head_board], resolution=150)
    manifest = {
        'schema_version': 1, 'name': 'ChibiVT · Turnaround reference pack',
        'kind': 'reference_pack_not_live2d_rig',
        'canvas': {'width': CANVAS[0], 'height': CANVAS[1], 'feet_y': FEET_Y},
        'angle_convention': '0 front; 90 profile facing viewer left; 180 back; 270 profile facing viewer right',
        'provenance': 'AI-generated from cutout.png. Full body and rear are proposed designs. Three views are mirrored drafts.',
        'limitations': [
            'Not a 3D reconstruction, not a rig-ready separated PSD, and not a 360-degree moc3.',
            'Asymmetric headset, scarf and pouches need manual consistency cleanup on ALL views.',
            'Source figures are about 660px tall; 768x1024 exports are upscaled, not newly detailed.',
            'Head X/Y labels are target keys, not measured or calibrated angles.',
        ],
        'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(SOURCES.glob('*.png'))},
        'frames': frames, 'head_frames': heads,
        'sprite_sheet': {'file': 'body_sprite_sheet.png', 'columns': 4, 'rows': 2, 'order': 'row-major', 'cell': list(CANVAS)},
        'downloads': {
            'board': 'turnaround_board.png', 'heads': 'head_board.png', 'pdf': 'reference_boards.pdf',
            'animation': 'turnaround.gif', 'zip': 'ChibiVT_turnaround_pack.zip',
        },
    }
    if psd:
        make_psd(frames, heads, output)
        manifest['downloads']['psd'] = 'ChibiVT_turnaround_references.psd'
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    with (output / 'storyboard.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['frame', 'time_ms', 'angle_degrees', 'label', 'file', 'status', 'pivot_x', 'pivot_y'])
        for i, frame in enumerate(frames):
            writer.writerow([i + 1, i * 500, frame['angle'], frame['label'], frame['file'], frame['status'], *frame['pivot']])
    (output / 'README.md').write_text((ROOT / 'turnaround' / 'README.md').read_text(encoding='utf-8'), encoding='utf-8')
    # Export only the known deliverables, never stale files or the archive itself.
    files = [f['file'] for f in frames] + [h['file'] for h in heads]
    files += ['manifest.json', 'storyboard.csv', 'README.md', 'body_sprite_sheet.png']
    files += [v for k, v in manifest['downloads'].items() if k != 'zip']
    with zipfile.ZipFile(output / manifest['downloads']['zip'], 'w', zipfile.ZIP_DEFLATED) as archive:
        for file in files:
            archive.write(output / file, 'ChibiVT_turnaround/' + file)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'turnaround' / 'pack')
    parser.add_argument('--psd', action='store_true', help='Export a PSD with whole reference frames, not rigging parts')
    args = parser.parse_args()
    result = build(args.output.resolve(), args.psd)
    print(f"Built {len(result['frames'])} body frames + {len(result['head_frames'])} head references in {args.output}")


if __name__ == '__main__':
    main()

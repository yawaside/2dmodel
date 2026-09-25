#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Package the generated model: add an icon + README and build a .zip."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import zipfile

from PIL import Image

README_RU = """\
МОДЕЛЬ ДЛЯ VTube Studio — {name}
=====================================

ЧТО ВНУТРИ
  {name}.model3.json     главный файл модели (его ищет VTube Studio)
  {name}.moc3            геометрия, деформеры, параметры
  {name}.physics3.json   физика: корпус догоняет поворот головы
  {name}.cdi3.json       служебная информация (не обязательна)
  textures/texture_00.png  текстура персонажа (2048x1024)
  icon.png               иконка модели в списке VTube Studio

КАК УСТАНОВИТЬ
  1. Скопируйте ВСЮ папку {name} целиком (не отдельные файлы!) в папку
     Live2DModels вашего VTube Studio. Открыть её можно кнопкой
     «Open Data Folder»/«Open Folder» в настройках VTube Studio
     (Steam-версия), либо вручную:
       Windows : %%USERPROFILE%%\\Documents\\VTube Studio\\Live2DModels
                 (или Data\\Live2DModels рядом с приложением)
       macOS   : ~/Library/Application Support/VTube Studio/Data/Live2DModels
  2. Запустите VTube Studio — модель появится в списке моделей.
  3. При первом открытии согласитесь на «Auto-Setup»: VTube Studio сам
     привяжет трекинг лица к стандартным параметрам модели.

ПАРАМЕТРЫ (стандартные Live2D — трекинг работает сразу)
  ParamAngleX / Y / Z        поворот, наклон и крен головы   (-30..30)
  ParamBodyAngleX / Y / Z    корпус                          (-10..10)
  ParamEyeLOpen / ParamEyeROpen  моргание (1 — открыт, 0 — закрыт)
  ParamMouthOpenY            открытие рта (лип-синк)
  ParamMouthForm             не задействован (параметр можно добавить)

ЕСЛИ ЧТО-ТО НЕ РАБОТАЕТ
  • Модель не видна — проверьте, что папка скопирована целиком и лежит
    именно в Live2DModels (а не в подпапке подпапки).
  • Голова/рот двигаются слабо — в настройках модели VTube Studio
    (шестерёнка → Model Movement / VTS Parameter Setup) увеличьте OUT-значения
    у FaceAngleX/Y/Z и MouthOpen.
  • Корпус не двигается — включите Body Angle в настройках модели или
    увеличьте вес в {name}.physics3.json (поле Weight).
  • Не нравится физика — удалите {name}.physics3.json и ссылку на него
    в {name}.model3.json: модель продолжит работать без физики.

ПЕРЕСБОРКА (если нужно подправить риг)
  Модель собирается из одного изображения скриптами в каталоге tools/:
      python3 tools/rig.py               # текстура: глаза/рот вырезаны, атлас
      python3 tools/build_model.py       # меши, деформеры, параметры -> .moc3
      python3 tools/validate.py          # проверка через Cubism Core
      python3 tools/poses.py             # контактный лист поз для проверки
  Полный цикл пересборки — эти четыре команды по порядку.

ЧТО ГДЕ КРУТИТЬ
  tools/build_model.py, словарь CFG (в начале файла):
      yaw_scale / pitch_scale / roll_scale — амплитуда поворота головы
      r_yaw / r_pitch                     — «толщина» головы (перспектива)
      eye_closed_scale                    — насколько сжимается глаз при моргании
      body_shift_x / body_shift_y         — наклон корпуса
  tools/rig.py, cfg в конце файла (__main__):
      mouth_open_w / mouth_open_h         — размер открытого рта
      eye_dark_thresh / mouth_dark_thresh — порог, по которому ищутся глаза и рот
"""

README_EN = """
VTube Studio model — {name}
=====================================

Copy the WHOLE {name} folder into VTube Studio's "Live2DModels" folder
(Steam version: settings -> "Open Data Folder"), start VTube Studio and run
"Auto-Setup" when it asks. Parameters are the standard Live2D ones
(ParamAngleX/Y/Z, ParamBodyAngleX/Y/Z, ParamEyeLOpen/ROpen, ParamMouthOpenY),
so face tracking, blinking and lip-sync work out of the box.
"""


def make_icon(atlas_path, out_path, size=512, box=(150, 55, 880, 1023)):
    im = Image.open(atlas_path).convert('RGBA').crop(box)
    im = im.resize((size, size), Image.LANCZOS)
    im.save(out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='dist/ChibiVT')
    ap.add_argument('--atlas', default='build/texture_atlas.png')
    ap.add_argument('--zip', default='dist/ChibiVT_vtube_studio.zip')
    args = ap.parse_args()

    d = args.dir
    name = os.path.basename(d.rstrip('/\\'))
    make_icon(args.atlas, os.path.join(d, 'icon.png'))
    with open(os.path.join(d, 'README.txt'), 'w', encoding='utf-8') as f:
        f.write(README_RU.format(name=name))
    with open(os.path.join(d, 'README.en.txt'), 'w', encoding='utf-8') as f:
        f.write(README_EN.format(name=name))

    zpath = args.zip
    os.makedirs(os.path.dirname(zpath) or '.', exist_ok=True)
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(d):
            for fn in sorted(files):
                p = os.path.join(root, fn)
                z.write(p, os.path.join(name, os.path.relpath(p, d)))
    print('kit      : %s' % d)
    for root, _, files in os.walk(d):
        for fn in sorted(files):
            p = os.path.join(root, fn)
            print('   %-40s %8d B' % (os.path.relpath(p, d), os.path.getsize(p)))
    print('zip      : %s (%d B)' % (zpath, os.path.getsize(zpath)))


if __name__ == '__main__':
    main()

# Раскадровка глаз и рта · липсинк для VTube Studio

Пакет-референс **лицевой анимации**: все нарисованные кадры глаз и рта,
зоны параметров, шкалы блендинга и живые GIF-демонстрации.

Кадры собраны **из тайлов текстурного атласа Live2D-модели** (`tools/rig.py`),
поэтому каждый кадр пиксель-в-пиксель совпадает с тем, что рендерит
`.moc3` при соответствующих значениях параметров. Это не «примерные»
референсы — это точное содержимое модели.

## Состав пакета

| Файл | Что это |
|---|---|
| `boards/mouth_board.png` | Лист рта: 6 кадров + шкала `ParamMouthOpenY`/`ParamMouthForm` |
| `boards/eye_board.png` | Лист глаз: 5 состояний + шкала `ParamEyeLOpen/R`, готовые выражения |
| `frames/mouth/` | Кадры рта на голове (нативное разрешение, PNG с прозрачностью) |
| `frames/eyes/` | Кадры глаз на голове |
| `frames/expressions/` | Готовые комбинации «глаза + рот» |
| `mouth_sprite_sheet.png` / `eyes_sprite_sheet.png` | Сырые тайлы атласа модели |
| `lipsync_demo.gif` | Реальный блендинг цепочки липсинка 0→1→0 + улыбка/ухмылка |
| `blink_cycle.gif` | Цикл моргания 1→0→1 |
| `expressions.gif` | `ParamMouthForm` −1…+1, глаза следуют |
| `storyboard.csv` | Кадр → параметр → диапазон (таблица для риггера) |
| `manifest.json` | Машиночитаемая версия: зоны hold/transition, ключи, хеши |
| `storyboard_boards.pdf` | Оба листа одним PDF |
| `ChibiVT_face_storyboard.psd` | Слой на каждый кадр (группы MOUTH / EYES / EXPRESSIONS) |
| `ChibiVT_face_storyboard.zip` | Архив всего пакета |

## Как читать раскадровку

**Рот** открывается одной монотонной цепочкой
`закрыт → слегка → полуоткрыт → «А»` по `ParamMouthOpenY`:

| Кадр | Файл PSD | `ParamMouthOpenY` |
|---|---|---|
| 00 закрыт | BASE | 0.00 – 0.12 (hold) |
| — переход | | 0.12 – 0.20 (crossfade) |
| 01 слегка | `M_slight_V1.png` | 0.20 – 0.38 (hold) |
| — переход | | 0.38 – 0.47 (crossfade) |
| 02 полуоткрыт | `M_half_V2.png` | 0.47 – 0.66 (hold) |
| — переход | | 0.66 – 0.76 (crossfade) |
| 03 «А» | `M_A_open_V3.png` | 0.76 – 1.00 (hold) |

`ParamMouthForm` −1…+1 плавно перекрашивает всю цепочку в ухмылку/улыбку
(`M_smirk_V8.png` / `M_I_grin_V5.png`). В каждый момент смешано **не более
двух соседних кадров** — «двойного рта» нет.

**Глаза**: `blink (0–0.2) → half (0.2–0.5) → neutral (0.5–1)` по
`ParamEyeLOpen/R`; `ParamEyeSmileL/R` заменяет открытый глаз «аркой»
(`E_*_happy_V5.png`), при закрытых веках подмешивается прищур (≈23%).

## Пересборка

```bash
pip install -r requirements-turnaround.txt
python3 tools/rig.py                      # 1. атлас + мета
python3 tools/build_face_storyboard.py --psd   # 2. пакет раскадровки
python3 -m unittest discover -s tests -v  # 3. проверки
```

Просмотр: поднять `python3 tools/serve.py . 8000` и открыть **`/storyboard/`**.

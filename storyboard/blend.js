/* Блендинг кадров лица — ОБЩИЙ файл для браузерного превью и тестов.
   Это точный порт mouth_blend / eye_blend / stack_opa из tools/build_model.py:
   та же математика работает внутри .moc3 в VTube Studio. */
'use strict';

function fclamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }

/* Прозрачность слоя k в альфа-стеке (0 = нижний слой). */
function stackOpa(weights, k) {
  let cum = 0;
  for (let j = 0; j <= k; j++) cum += weights[j];
  return cum <= 1e-6 ? 0 : fclamp(weights[k] / cum, 0, 1);
}

/* Вес 6 кадров рта: open = ParamMouthOpenY 0..1, form = ParamMouthForm -1..1.
   Порядок: closed, slight, half, A_open, smile, smirk. */
function mouthBlend(openY, form) {
  const w = [0, 0, 0, 0, 0, 0];
  const o = fclamp(openY, 0, 1);
  const f = fclamp(form, -1, 1);
  if (o <= 0.12) w[0] = 1;
  else if (o <= 0.20) { const t = (o - 0.12) / 0.08; w[0] = 1 - t; w[1] = t; }
  else if (o < 0.38) w[1] = 1;
  else if (o <= 0.47) { const t = (o - 0.38) / 0.09; w[1] = 1 - t; w[2] = t; }
  else if (o < 0.66) w[2] = 1;
  else if (o <= 0.76) { const t = (o - 0.66) / 0.10; w[2] = 1 - t; w[3] = t; }
  else w[3] = 1;
  if (f > 0) { for (let i = 0; i < 4; i++) w[i] *= 1 - f; w[4] = f; }
  else if (f < 0) { const k = -f; for (let i = 0; i < 4; i++) w[i] *= 1 - k; w[5] = k; }
  const total = w.reduce((a, b) => a + b, 0);
  return total > 0 ? w.map((x) => x / total) : [1, 0, 0, 0, 0, 0];
}

/* Вес 5 кадров глаза: open = ParamEyeLOpen/R 0..1, smile = ParamEyeSmileL/R 0..1.
   Порядок: neutral, half, blink, happy, squint. */
function eyeBlend(openVal, smileVal) {
  const w = [0, 0, 0, 0, 0];
  const o = fclamp(openVal, 0, 1);
  const s = fclamp(smileVal, 0, 1);
  if (o < 0.2) { w[2] = 1 - o / 0.2; w[1] = o / 0.2; }
  else if (o < 0.5) { const t = (o - 0.2) / 0.3; w[1] = 1 - t; w[0] = t; }
  else w[0] = 1;
  if (s > 0) {
    const squint = s * Math.max(0, 1 - o);
    for (let i = 0; i < 3; i++) w[i] *= 1 - s;
    w[3] += s;
    w[4] += squint * 0.3;
  }
  const total = w.reduce((a, b) => a + b, 0);
  return total > 0 ? w.map((x) => x / total) : [1, 0, 0, 0, 0];
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { fclamp, stackOpa, mouthBlend, eyeBlend };
}

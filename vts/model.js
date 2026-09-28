/* wardogs — математика модели в браузере (без Cubism Core).
 *
 * Этот файл — точный порт деформации из tools/build_model.py:
 *   head_warp / body_warp  — поворот и наклон головы, крен, дыхание;
 *   сеточные деформеры DHead (6x6) и DBody (5x5) — билинейная интерполяция
 *   смещений узлов, как это делает Cubism Core при обновлении модели.
 * Блендинг лица (mouthBlend / eyeBlend / stackOpa) берётся из
 * storyboard/blend.js — тот же файл, что проверяется юнит-тестом на
 * совпадение с Python-моделью.
 *
 * Используется и в браузере (vts/app.js), и в тестах (tests/test_vts.py).
 */
'use strict';

const BLEND = (typeof module !== 'undefined' && module.exports)
  ? require('../storyboard/blend.js')
  : { mouthBlend: mouthBlend, eyeBlend: eyeBlend, stackOpa: stackOpa };

/* Константы, которые в build_model.py записаны прямо в коде (не в CFG). */
const K = {
  BODY_T_Y0: 600.0,      // начало «следования» корпуса по Y
  BODY_T_SPAN: 423.0,
  BODY_X_SQUEEZE: 0.045, // лёгкое сжатие корпуса при повороте
  BODY_CENTER_X: 512.0,
  BODY_ROLL_FROM: 500.0, // зона крена корпуса (в зеркальных координатах)
  BODY_ROLL_TO: 900.0,
  BODY_HEAD_X: 0.30,     // корпус тянется за поворотом головы
  BREATH_BODY: 2.0,      // амплитуда дыхания корпуса
  BREATH_HEAD: 3.0,      // амплитуда дыхания головы (пиксели вверх)
  EYE_SMILE_FROM_PITCH: 0.3,  // естественный прищур при наклоне головы вверх
  /* Ограничение «широты» головы (порт V_LIMIT из build_model.py): держит
     cos(asin(v)) >= 0.25, чтобы масштаб по X при тангаже не взрывался.
     Ограничивается сам v — иначе нейтральный ключ не тождественен. */
  V_MAX: Math.sqrt(1 - 0.25 * 0.25),
};

let RIG = null;

/* Загрузить геометрию рига из манифеста пакета (vts/pack/manifest.json). */
function loadRig(manifest) {
  const d = manifest.deformers;
  RIG = {
    size: manifest.canvas.size,
    cutY: manifest.canvas.cut_y,
    head: { rect: d.head.rect.slice(), rows: d.head.rows, cols: d.head.cols },
    body: { rect: d.body.rect.slice(), rows: d.body.rows, cols: d.body.cols },
    headFade: manifest.rig.head_fade,
    headCenter: manifest.rig.head_center.slice(),
    neckPoint: manifest.rig.neck_point.slice(),
    rYaw: manifest.rig.r_yaw,
    rPitch: manifest.rig.r_pitch,
    yawScale: manifest.rig.yaw_scale,
    pitchScale: manifest.rig.pitch_scale,
    rollScale: manifest.rig.roll_scale,
    bodyShiftX: manifest.rig.body_shift_x,
    bodyShiftY: manifest.rig.body_shift_y,
    bodyRollDeg: manifest.rig.body_roll_deg,
    bodyHip: manifest.rig.body_hip.slice(),
    vMax: typeof manifest.rig.v_max === 'number' ? manifest.rig.v_max : K.V_MAX,
  };
  return RIG;
}

function clamp(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }

function smoothstep(edge0, edge1, x) {
  const t = clamp((x - edge0) / (edge1 - edge0), 0, 1);
  return t * t * (3 - 2 * t);
}

function rot(p, c, ang) {
  const s = Math.sin(ang), co = Math.cos(ang);
  const dx = p[0] - c[0], dy = p[1] - c[1];
  return [c[0] + dx * co - dy * s, c[1] + dx * s + dy * co];
}

/* Поворот головы: px,py — пиксели исходника, ax/ay/az — углы ParamAngle*.
   Возвращает АБСОЛЮТНУЮ позицию (не смещение). Порт head_warp(). */
function headWarp(px, py, ax, ay, az, breath) {
  py = py - (breath || 0) * K.BREATH_HEAD;      // дыхание: голова чуть приподнимается
  const w = smoothstep(RIG.cutY, RIG.cutY - RIG.headFade, py);
  const hcx = RIG.headCenter[0];
  const th = (ax * Math.PI / 180) * RIG.yawScale;
  const u = clamp((px - hcx) / RIG.rYaw, -1, 1);
  const phi = Math.asin(u);
  const x1 = px + RIG.rYaw * (Math.sin(phi + th) - Math.sin(phi));
  const y1 = py;
  const psi = -(ay * Math.PI / 180) * RIG.pitchScale;
  const v = clamp((y1 - RIG.headCenter[1]) / RIG.rPitch, -RIG.vMax, RIG.vMax);
  const a = Math.asin(v);
  const ca = Math.cos(a);
  const y2 = y1 + RIG.rPitch * (Math.sin(a + psi) - Math.sin(a));
  const x2 = x1 + (x1 - hcx) * (Math.cos(a + psi) / ca - 1);
  const p3 = rot([x2, y2], RIG.neckPoint, (az * Math.PI / 180) * RIG.rollScale);
  return [px + (p3[0] - px) * w, py + (p3[1] - py) * w];
}

/* Корпус: работает в «зеркальных» координатах (pym = size - py), как и в
   build_model.body_pos(). Порт body_warp(). */
function bodyWarpMirrored(px, pym, bx, by, bz, breath) {
  const xn = bx / 10, yn = by / 10, zn = bz / 10;
  const t = clamp((pym - K.BODY_T_Y0) / K.BODY_T_SPAN, 0, 1);
  let x = px + RIG.bodyShiftX * xn * t;
  x = K.BODY_CENTER_X + (x - K.BODY_CENTER_X) * (1 - K.BODY_X_SQUEEZE * Math.abs(xn));
  let y = pym - RIG.bodyShiftY * yn * (1 - t);
  y -= (breath || 0) * K.BREATH_BODY * t;
  const w = smoothstep(K.BODY_ROLL_FROM, K.BODY_ROLL_TO, pym);
  return rot([x, y], RIG.bodyHip, -(RIG.bodyRollDeg * Math.PI / 180) * zn * w);
}

/* Корпус в обычных пиксельных координатах исходника. */
function bodyWarp(px, py, st) {
  const pym = RIG.size - py;
  const p = bodyWarpMirrored(px, pym, st.ParamBodyAngleX || 0,
    st.ParamBodyAngleY || 0, st.ParamBodyAngleZ || 0, st.ParamBreath || 0);
  return [p[0] + K.BODY_HEAD_X * (st.ParamAngleX || 0), py + (pym - p[1])];
}

/* Узлы сетки деформера (строка-major): (rows+1) x (cols+1) точек. */
function gridNodes(rect, rows, cols) {
  const out = new Float64Array((rows + 1) * (cols + 1) * 2);
  let k = 0;
  for (let j = 0; j <= rows; j++) {
    for (let i = 0; i <= cols; i++) {
      out[k++] = rect[0] + (rect[2] - rect[0]) * i / cols;
      out[k++] = rect[1] + (rect[3] - rect[1]) * j / rows;
    }
  }
  return out;
}

/* Смещения узлов головы: head_warp(узел) − узел. */
function headGrid(st) {
  const g = RIG.head;
  const nodes = gridNodes(g.rect, g.rows, g.cols);
  const out = new Float64Array(nodes.length);
  for (let n = 0; n < nodes.length; n += 2) {
    const p = headWarp(nodes[n], nodes[n + 1], st.ParamAngleX || 0,
      st.ParamAngleY || 0, st.ParamAngleZ || 0, st.ParamBreath || 0);
    out[n] = p[0] - nodes[n];
    out[n + 1] = p[1] - nodes[n + 1];
  }
  return out;
}

/* Смещения узлов корпуса: body_warp(узел) − узел. */
function bodyGrid(st) {
  const g = RIG.body;
  const nodes = gridNodes(g.rect, g.rows, g.cols);
  const out = new Float64Array(nodes.length);
  for (let n = 0; n < nodes.length; n += 2) {
    const p = bodyWarp(nodes[n], nodes[n + 1], st);
    out[n] = p[0] - nodes[n];
    out[n + 1] = p[1] - nodes[n + 1];
  }
  return out;
}

/* Билинейная выборка смещения в точке (px,py) внутри сетки деформера. */
function sampleGrid(disp, rect, rows, cols, px, py, out) {
  const w = rect[2] - rect[0], h = rect[3] - rect[1];
  let u = (px - rect[0]) / w * cols;
  let v = (py - rect[1]) / h * rows;
  u = clamp(u, 0, cols); v = clamp(v, 0, rows);
  let i = Math.floor(u), j = Math.floor(v);
  if (i >= cols) i = cols - 1;
  if (j >= rows) j = rows - 1;
  const fu = u - i, fv = v - j;
  const stride = cols + 1;
  const a = (j * stride + i) * 2, b = (j * stride + i + 1) * 2;
  const c = ((j + 1) * stride + i) * 2, d = ((j + 1) * stride + i + 1) * 2;
  out[0] = (disp[a] * (1 - fu) + disp[b] * fu) * (1 - fv)
         + (disp[c] * (1 - fu) + disp[d] * fu) * fv;
  out[1] = (disp[a + 1] * (1 - fu) + disp[b + 1] * fu) * (1 - fv)
         + (disp[c + 1] * (1 - fu) + disp[d + 1] * fu) * fv;
  return out;
}

/* Итоговая деформация точки: DHead(точка) → DBody(результат).
   Порядок такой же, как в .moc3: деформер головы вложен в деформер корпуса. */
function warpPoint(px, py, st, hg, bg, tmp) {
  const h = RIG.head, b = RIG.body;
  const d = tmp || [0, 0];
  sampleGrid(hg, h.rect, h.rows, h.cols, px, py, d);
  const qx = px + d[0], qy = py + d[1];
  sampleGrid(bg, b.rect, b.rows, b.cols, qx, qy, d);
  return [qx + d[0], qy + d[1]];
}

/* Регулярная сетка для warp-рендера: вершины в пикселях исходника + UV. */
function buildGrid(rect, step) {
  const x0 = rect[0], y0 = rect[1], x1 = rect[2], y1 = rect[3];
  const cols = Math.max(1, Math.round((x1 - x0) / step));
  const rows = Math.max(1, Math.round((y1 - y0) / step));
  const n = (rows + 1) * (cols + 1);
  const pos = new Float32Array(n * 2);
  const uv = new Float32Array(n * 2);
  let k = 0;
  for (let j = 0; j <= rows; j++) {
    for (let i = 0; i <= cols; i++) {
      const px = x0 + (x1 - x0) * i / cols;
      const py = y0 + (y1 - y0) * j / rows;
      pos[k] = px; pos[k + 1] = py;
      uv[k] = (px - x0) / (x1 - x0);
      uv[k + 1] = (py - y0) / (y1 - y0);
      k += 2;
    }
  }
  return { rows: rows, cols: cols, count: n, pos: pos, uv: uv };
}

/* Треугольники сетки, у которых есть арт (cover(vx,vy) → bool). */
function buildTriangles(grid, cover) {
  const stride = grid.cols + 1;
  const idx = [];
  for (let j = 0; j < grid.rows; j++) {
    for (let i = 0; i < grid.cols; i++) {
      const a = j * stride + i, b = a + 1, c = a + stride, d = c + 1;
      if (!(cover(a) || cover(b) || cover(c) || cover(d))) continue;
      idx.push(a, b, c, b, d, c);
    }
  }
  return new Uint16Array(idx);
}

/* Рёбра той же сетки — для показа «каркаса модели». */
function buildWire(grid, cover) {
  const stride = grid.cols + 1;
  const idx = [];
  for (let j = 0; j <= grid.rows; j++) {
    for (let i = 0; i <= grid.cols; i++) {
      const a = j * stride + i;
      if (i < grid.cols) {
        const b = a + 1;
        if (cover(a) || cover(b)) idx.push(a, b);
      }
      if (j < grid.rows) {
        const c = a + stride;
        if (cover(a) || cover(c)) idx.push(a, c);
      }
    }
  }
  return new Uint16Array(idx);
}

/* Расширить карту покрытия на `steps` вершин (арт тоньше шага сетки не теряется). */
function dilateCover(cover, grid, steps) {
  if (!steps) return cover;
  const stride = grid.cols + 1;
  let cur = cover.slice();
  for (let s = 0; s < steps; s++) {
    const next = cur.slice();
    for (let j = 0; j <= grid.rows; j++) {
      for (let i = 0; i <= grid.cols; i++) {
        const a = j * stride + i;
        if (cur[a]) continue;
        if ((i > 0 && cur[a - 1]) || (i < grid.cols && cur[a + 1]) ||
            (j > 0 && cur[a - stride]) || (j < grid.rows && cur[a + stride])) next[a] = 1;
      }
    }
    cur = next;
  }
  return cur;
}

/* Деформация всех вершин сетки: result — Float32Array(n*2) в пикселях. */
function warpGrid(grid, st, hg, bg, result) {
  const out = result || new Float32Array(grid.pos.length);
  const h = RIG.head, b = RIG.body;
  const d = [0, 0];
  for (let n = 0; n < grid.pos.length; n += 2) {
    const px = grid.pos[n], py = grid.pos[n + 1];
    sampleGrid(hg, h.rect, h.rows, h.cols, px, py, d);
    const qx = px + d[0], qy = py + d[1];
    sampleGrid(bg, b.rect, b.rows, b.cols, qx, qy, d);
    out[n] = qx + d[0];
    out[n + 1] = qy + d[1];
  }
  return out;
}

/* Естественный прищур от наклона головы — как в opa_fn глаз в build_model.py. */
function eyeSmileNatural(smile, angleY) {
  const s = smile + Math.max(0, angleY / 30) * K.EYE_SMILE_FROM_PITCH;
  return clamp(s, 0, 1);
}

const API = {
  K: K, BLEND: BLEND,
  loadRig: loadRig,
  rig: function () { return RIG; },
  clamp: clamp,
  smoothstep: smoothstep,
  rot: rot,
  headWarp: headWarp,
  bodyWarpMirrored: bodyWarpMirrored,
  bodyWarp: bodyWarp,
  gridNodes: gridNodes,
  headGrid: headGrid,
  bodyGrid: bodyGrid,
  sampleGrid: sampleGrid,
  warpPoint: warpPoint,
  buildGrid: buildGrid,
  buildTriangles: buildTriangles,
  buildWire: buildWire,
  dilateCover: dilateCover,
  warpGrid: warpGrid,
  eyeSmileNatural: eyeSmileNatural,
};

if (typeof module !== 'undefined' && module.exports) module.exports = API;
else if (typeof window !== 'undefined') window.Chibi = API;

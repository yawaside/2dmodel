/* ChibiVT — интерактивный предпросмотр «как в VTube Studio».
 *
 * Модель рисуется без Cubism Core:
 *   · лицо — альфа-стек тайлов (storyboard/blend.js = mouth_blend/eye_blend
 *     из tools/build_model.py, совпадение проверяется юнит-тестом);
 *   · геометрия — порт деформеров head 6x6 / body 5x5 (vts/model.js);
 *   · движение — те же Param*, что VTube Studio передаёт в .moc3.
 */
'use strict';

(() => {
const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.prototype.slice.call(document.querySelectorAll(s));
const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
const approach = (cur, target, dt, tau) =>
  cur + (target - cur) * (1 - Math.exp(-dt / Math.max(1e-4, tau)));

/* ------------------------------------------------------------------ */
/* состояние                                                           */
/* ------------------------------------------------------------------ */

let manifest = null;
let params = [];               // описания из манифеста
const byId = {};               // id -> описание
const base = {};               // значения ползунков / экспрессий
const state = {};              // то, что идёт в рендер
const rows = {};               // id -> {row, input, out, note}

const sources = { track: true, idle: false, blink: true, physics: true, mic: false, talk: false };
const opts = { hud: true, wire: false, smooth: 0.14, amp: 1.0, blinkMax: 6.0, gain: 3.0, gate: 0.06 };
const view = { size: 1.0, x: 0, y: 0, rot: 0, bg: 'transparent' };

const track = { tx: 0, ty: 0, cx: 0, cy: 0, inside: false };
const TRACK_RANGE = { x: 20, y: 12, z: 8 };
const mic = { ctx: null, analyser: null, buf: null, stream: null, level: 0, mouth: 0 };
const talk = { t0: 0, dur: 0, amp: 0, next: 0, value: 0 };
const blink = { next: 2.5, active: null, queue: 0 };
const phys = { breath: 0, bodyX: 0, bodyY: 0, smile: 0, form: 0 };
let gestures = [];
let clock = 0;

const EXPRESSIONS = {
  neutral: {},
  smile: { ParamMouthForm: 1, ParamEyeSmileL: 0.85, ParamEyeSmileR: 0.85,
           ParamMouthOpenY: 0.12, ParamEyeLOpen: 0.82, ParamEyeROpen: 0.82 },
  smirk: { ParamMouthForm: -1, ParamEyeSmileL: 0.4, ParamEyeSmileR: 0.4,
           ParamMouthOpenY: 0.08 },
  talk: { ParamMouthOpenY: 0.9, ParamMouthForm: 0 },
  surprise: { ParamMouthOpenY: 0.45, ParamMouthForm: 0,
              ParamEyeLOpen: 1, ParamEyeROpen: 1,
              ParamEyeSmileL: 0, ParamEyeSmileR: 0 },
  sleepy: { ParamEyeLOpen: 0.3, ParamEyeROpen: 0.3, ParamMouthOpenY: 0.05,
            ParamMouthForm: 0, ParamEyeSmileL: 0, ParamEyeSmileR: 0,
            ParamAngleZ: -5, ParamAngleY: -6, ParamBodyAngleY: -3 },
};
const GESTURES = {
  nod: { id: 'ParamAngleY', dur: 0.7, keys: [[0, 0], [0.35, 14], [0.7, 0]] },
  shake: { id: 'ParamAngleX', dur: 0.9, keys: [[0, 0], [0.25, -14], [0.6, 14], [0.9, 0]] },
};

/* ------------------------------------------------------------------ */
/* WebGL                                                               */
/* ------------------------------------------------------------------ */

const canvas = $('#cv');
const gl = canvas.getContext('webgl', {
  alpha: true, premultipliedAlpha: true, antialias: true,
  preserveDrawingBuffer: true,   // нужен для «Скачать PNG кадра»
});

const VS_MODEL = `
attribute vec2 a_pos;
attribute vec2 a_uv;
uniform vec2 u_half;
uniform float u_scale;
uniform vec2 u_off;
uniform vec2 u_rot;
varying vec2 v_uv;
void main() {
  vec2 p = (a_pos - 512.0) * u_scale;
  p = vec2(p.x * u_rot.x - p.y * u_rot.y, p.x * u_rot.y + p.y * u_rot.x);
  p += u_off;
  gl_Position = vec4(p.x / u_half.x, -p.y / u_half.y, 0.0, 1.0);
  v_uv = a_uv;
}`;

const FS_MODEL = `
precision mediump float;
uniform sampler2D u_tex;
uniform float u_alpha;
uniform vec4 u_tint;
uniform float u_tint_mix;
varying vec2 v_uv;
void main() {
  vec4 c = texture2D(u_tex, v_uv) * u_alpha;
  gl_FragColor = mix(c, u_tint, u_tint_mix);
}`;

const VS_BG = `
attribute vec2 a_pos;
varying vec2 v_uv;
void main() { v_uv = a_pos * 0.5 + 0.5; gl_Position = vec4(a_pos, 0.0, 1.0); }`;

const FS_BG = `
precision mediump float;
uniform int u_mode;
uniform vec2 u_res;
varying vec2 v_uv;
void main() {
  if (u_mode == 1) {                       // тёмная сцена
    float d = distance(v_uv, vec2(0.5, 0.42));
    vec3 c = mix(vec3(0.145, 0.155, 0.19), vec3(0.045, 0.05, 0.065), smoothstep(0.05, 0.95, d));
    gl_FragColor = vec4(c, 1.0);
  } else if (u_mode == 2) {                // хромакей OBS
    gl_FragColor = vec4(0.0, 1.0, 0.0, 1.0);
  } else {                                 // стрим-фон
    float g = clamp(v_uv.y, 0.0, 1.0);
    vec3 c = mix(vec3(0.24, 0.13, 0.32), vec3(0.06, 0.08, 0.18), g);
    float d = distance(v_uv, vec2(0.5, 0.5));
    c *= 1.0 - 0.55 * smoothstep(0.35, 0.95, d);
    gl_FragColor = vec4(c, 1.0);
  }
}`;

function compile(type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}
function program(vs, fs) {
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl.VERTEX_SHADER, vs));
  gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  return p;
}

let progModel = null, progBg = null;
let A_POS = 0, A_UV = 0, A_BG = 0;
const U = {};

function initGL() {
  progModel = program(VS_MODEL, FS_MODEL);
  progBg = program(VS_BG, FS_BG);
  A_POS = gl.getAttribLocation(progModel, 'a_pos');
  A_UV = gl.getAttribLocation(progModel, 'a_uv');
  A_BG = gl.getAttribLocation(progBg, 'a_pos');
  ['u_half', 'u_scale', 'u_off', 'u_rot', 'u_tex', 'u_alpha', 'u_tint', 'u_tint_mix']
    .forEach((n) => { U[n] = gl.getUniformLocation(progModel, n); });
  U.bg_mode = gl.getUniformLocation(progBg, 'u_mode');
  U.bg_res = gl.getUniformLocation(progBg, 'u_res');

  gl.enable(gl.BLEND);
  gl.blendFuncSeparate(gl.ONE, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

  const quad = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, quad);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
  progBg.quad = quad;
}

function makeTexture(source) {
  const t = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  return t;
}

/* ------------------------------------------------------------------ */
/* ассеты                                                              */
/* ------------------------------------------------------------------ */

const assets = { head: null, body: null, mouth: [], eyes: [[], []] };
let headMesh = null, bodyMesh = null;
let faceCanvas = null, fctx = null;
let faceDirty = true, faceSig = '';
let headGridDef = null, bodyGridDef = null;

function loadImage(src) {
  return new Promise((res, rej) => {
    const im = new Image();
    im.onload = () => res(im);
    im.onerror = () => rej(new Error('не загрузился ' + src));
    im.src = src;
  });
}

async function loadAssets() {
  manifest = await (await fetch('pack/manifest.json')).json();
  Chibi.loadRig(manifest);
  params = manifest.parameters;
  params.forEach((p) => { byId[p.id] = p; base[p.id] = p.default; state[p.id] = p.default; });

  const f = manifest.files;
  const url = (rel) => 'pack/' + rel;          // манифест живёт в pack/
  assets.head = await loadImage(url(f.head));
  assets.body = await loadImage(url(f.body));
  assets.mouth = await Promise.all(f.mouth_tiles.map(url).map(loadImage));
  assets.eyes = [
    await Promise.all(f.eye_tiles.L.map(url).map(loadImage)),
    await Promise.all(f.eye_tiles.R.map(url).map(loadImage)),
  ];

  headGridDef = Chibi.buildGrid(manifest.canvas.head_rect, manifest.browser_mesh.step);
  bodyGridDef = Chibi.buildGrid(manifest.canvas.body_rect, manifest.browser_mesh.step);

  faceCanvas = document.createElement('canvas');
  faceCanvas.width = manifest.canvas.head_rect[2] - manifest.canvas.head_rect[0];
  faceCanvas.height = manifest.canvas.head_rect[3] - manifest.canvas.head_rect[1];
  fctx = faceCanvas.getContext('2d');
}

function coverage(img, rect, grid) {
  const c = document.createElement('canvas');
  c.width = img.width; c.height = img.height;
  const cx = c.getContext('2d');
  cx.drawImage(img, 0, 0);
  const data = cx.getImageData(0, 0, c.width, c.height).data;
  const cov = new Uint8Array(grid.count);
  for (let i = 0; i < grid.count; i++) {
    const px = Math.round(grid.pos[2 * i] - rect[0]);
    const py = Math.round(grid.pos[2 * i + 1] - rect[1]);
    if (px < 0 || py < 0 || px >= c.width || py >= c.height) continue;
    cov[i] = data[(py * c.width + px) * 4 + 3] > 12 ? 1 : 0;
  }
  return Chibi.dilateCover(cov, grid, manifest.browser_mesh.dilate_steps);
}

function buildMesh(grid, img, rect, tex) {
  const cov = coverage(img, rect, grid);
  const keep = (i) => !!cov[i];
  const posBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
  gl.bufferData(gl.ARRAY_BUFFER, grid.pos.length * 4, gl.DYNAMIC_DRAW);
  gl.bufferSubData(gl.ARRAY_BUFFER, 0, grid.pos);
  const uvBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, uvBuf);
  gl.bufferData(gl.ARRAY_BUFFER, grid.uv, gl.STATIC_DRAW);
  const triIdx = Chibi.buildTriangles(grid, keep);
  const idxBuf = gl.createBuffer();
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idxBuf);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, triIdx, gl.STATIC_DRAW);
  const wireIdx = Chibi.buildWire(grid, keep);
  const wireBuf = gl.createBuffer();
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, wireBuf);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, wireIdx, gl.STATIC_DRAW);
  return {
    grid: grid, posBuf: posBuf, uvBuf: uvBuf, idxBuf: idxBuf, wireBuf: wireBuf,
    n: triIdx.length, wireN: wireIdx.length,
    tex: tex, warp: new Float32Array(grid.pos.length),
  };
}

/* ------------------------------------------------------------------ */
/* композитинг лица (тот же альфа-стек, что в .moc3)                    */
/* ------------------------------------------------------------------ */

function drawStack(tiles, weights, box) {
  const ox = manifest.canvas.head_rect[0], oy = manifest.canvas.head_rect[1];
  for (let i = 0; i < tiles.length; i++) {
    const opa = Chibi.BLEND.stackOpa(weights, i);
    if (opa <= 0.001) continue;
    fctx.globalAlpha = opa;
    fctx.drawImage(tiles[i], box[0] - ox, box[1] - oy);
  }
  fctx.globalAlpha = 1;
}

function compositeFace() {
  const boxes = manifest.boxes;
  const open = state.ParamMouthOpenY, form = state.ParamMouthForm;
  const ay = state.ParamAngleY;
  const sig = [open, form, state.ParamEyeLOpen, state.ParamEyeROpen,
    state.ParamEyeSmileL, state.ParamEyeSmileR, ay]
    .map((v) => Math.round(v * 1000)).join(',');
  if (sig === faceSig) return;
  faceSig = sig;

  fctx.clearRect(0, 0, faceCanvas.width, faceCanvas.height);
  fctx.globalAlpha = 1;
  fctx.drawImage(assets.head, 0, 0);
  drawStack(assets.mouth, Chibi.BLEND.mouthBlend(open, form), boxes.mouth_box);
  for (let side = 0; side < 2; side++) {
    const openEye = side === 0 ? state.ParamEyeLOpen : state.ParamEyeROpen;
    const smile = side === 0 ? state.ParamEyeSmileL : state.ParamEyeSmileR;
    drawStack(assets.eyes[side],
      Chibi.BLEND.eyeBlend(openEye, Chibi.eyeSmileNatural(smile, ay)),
      boxes.eye_boxes[side]);
  }
  faceDirty = true;
}

/* ------------------------------------------------------------------ */
/* кадр                                                                */
/* ------------------------------------------------------------------ */

let lastT = 0, fpsAcc = 0, fpsFrames = 0, fpsShow = 0;

function resize() {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const w = Math.max(1, Math.round(canvas.clientWidth * dpr));
  const h = Math.max(1, Math.round(canvas.clientHeight * dpr));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w; canvas.height = h;
  }
}

const BG_MODE = { transparent: 0, dark: 1, green: 2, stream: 3 };

/* Атрибуты — общие для всех программ, поэтому включаем только нужные. */
function useAttribs(list) {
  [A_POS, A_UV, A_BG].forEach((loc) => {
    if (loc < 0) return;
    const on = list.indexOf(loc) >= 0;
    if (on) gl.enableVertexAttribArray(loc);
    else gl.disableVertexAttribArray(loc);
  });
}

function drawBackground() {
  const mode = BG_MODE[view.bg] || 0;
  if (!mode) return;                       // прозрачный: шахматы рисует CSS
  gl.disable(gl.BLEND);
  gl.useProgram(progBg);
  gl.bindBuffer(gl.ARRAY_BUFFER, progBg.quad);
  useAttribs([A_BG]);
  gl.vertexAttribPointer(A_BG, 2, gl.FLOAT, false, 0, 0);
  gl.uniform1i(U.bg_mode, mode);
  gl.uniform2f(U.bg_res, canvas.width, canvas.height);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  gl.enable(gl.BLEND);
}

function drawMesh(m) {
  gl.bindBuffer(gl.ARRAY_BUFFER, m.posBuf);
  useAttribs([A_POS, A_UV]);
  gl.vertexAttribPointer(A_POS, 2, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, m.uvBuf);
  gl.vertexAttribPointer(A_UV, 2, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, m.idxBuf);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, m.tex);
  gl.uniform1i(U.u_tex, 0);
  gl.drawElements(gl.TRIANGLES, m.n, gl.UNSIGNED_SHORT, 0);
}

function drawWire(m) {
  gl.bindBuffer(gl.ARRAY_BUFFER, m.posBuf);
  useAttribs([A_POS]);
  gl.vertexAttribPointer(A_POS, 2, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, m.wireBuf);
  gl.drawElements(gl.LINES, m.wireN, gl.UNSIGNED_SHORT, 0);
}

function draw() {
  resize();
  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.clearColor(0, 0, 0, 0);
  gl.clear(gl.COLOR_BUFFER_BIT);
  drawBackground();

  const hg = Chibi.headGrid(state);
  const bg = Chibi.bodyGrid(state);

  const half = [canvas.width / 2, canvas.height / 2];
  const fit = Math.min(canvas.width, canvas.height) / manifest.canvas.size * 0.98;
  const scale = fit * view.size;
  const rad = view.rot * Math.PI / 180;

  gl.useProgram(progModel);
  gl.uniform2f(U.u_half, half[0], half[1]);
  gl.uniform1f(U.u_scale, scale);
  gl.uniform2f(U.u_off, view.x / 100 * canvas.width, view.y / 100 * canvas.height);
  gl.uniform2f(U.u_rot, Math.cos(rad), Math.sin(rad));
  gl.uniform1f(U.u_alpha, 1);
  gl.uniform1f(U.u_tint_mix, 0);
  gl.uniform4f(U.u_tint, 0, 0, 0, 0);

  for (const m of [bodyMesh, headMesh]) {
    Chibi.warpGrid(m.grid, state, hg, bg, m.warp);
    gl.bindBuffer(gl.ARRAY_BUFFER, m.posBuf);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, m.warp);
  }
  if (faceDirty) {
    gl.bindTexture(gl.TEXTURE_2D, headMesh.tex);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, faceCanvas);
    faceDirty = false;
  }

  drawMesh(bodyMesh);
  drawMesh(headMesh);

  if (opts.wire) {
    const a = 0.35;
    gl.uniform1f(U.u_tint_mix, 1);
    gl.uniform4f(U.u_tint, 0.35 * a, 0.85 * a, 1.0 * a, a);
    drawWire(bodyMesh);
    drawWire(headMesh);
    gl.uniform1f(U.u_tint_mix, 0);
  }
}

/* ------------------------------------------------------------------ */
/* источники значений: трекинг, idle, микрофон, речь, моргание, физика  */
/* ------------------------------------------------------------------ */

function updateTracking(dt) {
  const k = 1 - Math.exp(-dt / Math.max(0.01, opts.smooth));
  const tx = track.inside ? track.tx : 0;
  const ty = track.inside ? track.ty : 0;
  track.cx += (tx - track.cx) * k;
  track.cy += (ty - track.cy) * k;
  if (sources.track) {
    // Курсор -> углы: как «OUT»-значения трекинга в VTS (сам параметр живёт в ±30).
    state.ParamAngleX = track.cx * TRACK_RANGE.x;
    state.ParamAngleY = -track.cy * TRACK_RANGE.y;
    state.ParamAngleZ = track.cx * TRACK_RANGE.z;
  } else if (sources.idle) {
    const t = clock, a = opts.amp;
    state.ParamAngleX = Math.sin(t * 0.7) * 22 * a;
    state.ParamAngleY = Math.sin(t * 0.53 + 1) * 14 * a;
    state.ParamAngleZ = Math.sin(t * 0.41) * 8 * a;
  } else {
    state.ParamAngleX = base.ParamAngleX;
    state.ParamAngleY = base.ParamAngleY;
    state.ParamAngleZ = base.ParamAngleZ;
  }
}

function updateMouth(dt) {
  let open = base.ParamMouthOpenY;
  if (sources.mic && mic.analyser) {
    mic.analyser.getByteTimeDomainData(mic.buf);
    let sum = 0;
    for (let i = 0; i < mic.buf.length; i++) {
      const v = (mic.buf[i] - 128) / 128;
      sum += v * v;
    }
    const rms = Math.sqrt(sum / mic.buf.length);
    mic.level = clamp(rms * opts.gain, 0, 1);
    const target = clamp((mic.level - opts.gate) / Math.max(0.05, 1 - opts.gate), 0, 1);
    mic.mouth = approach(mic.mouth, target, dt, target > mic.mouth ? 0.02 : 0.09);
    open = mic.mouth;
    $('#mic-level').style.width = (mic.level * 100).toFixed(1) + '%';
  } else if (sources.talk) {
    if (clock >= talk.next) {
      talk.t0 = clock;
      talk.dur = 0.08 + Math.random() * 0.10;
      talk.amp = 0.35 + Math.random() * 0.65;
      talk.next = clock + talk.dur + 0.02 + Math.random() * 0.10;
      if (Math.random() < 0.14) talk.next += 0.2 + Math.random() * 0.45;   // пауза между фразами
    }
    const p = (clock - talk.t0) / talk.dur;
    const env = (p >= 0 && p <= 1) ? Math.pow(Math.sin(Math.PI * p), 0.75) : 0;
    talk.value = env * talk.amp;
    open = talk.value;
    $('#mic-level').style.width = (talk.value * 100).toFixed(1) + '%';
  } else {
    mic.level = 0; mic.mouth = 0; talk.value = 0;
    $('#mic-level').style.width = '0%';
  }
  state.ParamMouthOpenY = open;
}

const EASE = (p) => p * p * (3 - 2 * p);

function startBlink() {
  const single = 0.09, hold = 0.03, back = 0.13;
  blink.active = { t: 0, close: single, hold: hold, open: back };
}

function updateBlink(dt) {
  if (!sources.blink && !blink.active) {
    state.ParamEyeLOpen = base.ParamEyeLOpen;
    state.ParamEyeROpen = base.ParamEyeROpen;
    return;
  }
  if (blink.active === null && sources.blink) {
    blink.next -= dt;
    if (blink.next <= 0) {
      startBlink();
      blink.queue = Math.random() < 0.2 ? 1 : 0;      // иногда двойное моргание
    }
  }
  if (blink.active) {
    const a = blink.active;
    a.t += dt;
    let v = 1;
    if (a.t < a.close) v = 1 - EASE(a.t / a.close);
    else if (a.t < a.close + a.hold) v = 0;
    else if (a.t < a.close + a.hold + a.open) v = EASE((a.t - a.close - a.hold) / a.open);
    else {
      blink.active = null;
      if (blink.queue > 0) { blink.queue--; startBlink(); blink.next = 0.12; return updateBlink(0); }
      blink.next = 1.6 + Math.random() * Math.max(0.4, opts.blinkMax - 1.6);
      v = 1;
    }
    state.ParamEyeLOpen = v;
    state.ParamEyeROpen = v;
  } else {
    state.ParamEyeLOpen = base.ParamEyeLOpen;
    state.ParamEyeROpen = base.ParamEyeROpen;
  }
}

function updatePhysics(dt) {
  if (!sources.physics) {
    state.ParamBreath = base.ParamBreath;
    state.ParamBodyAngleX = base.ParamBodyAngleX;
    state.ParamBodyAngleY = base.ParamBodyAngleY;
    state.ParamBodyAngleZ = base.ParamBodyAngleZ;
    state.ParamEyeSmileL = base.ParamEyeSmileL;
    state.ParamEyeSmileR = base.ParamEyeSmileR;
    state.ParamMouthForm = base.ParamMouthForm;
    phys.bodyX = state.ParamBodyAngleX; phys.bodyY = state.ParamBodyAngleY;
    phys.smile = 0; phys.form = 0;
    return;
  }
  // дыхание (≈ 4.2 с на вдох-выдох, как в VTS по умолчанию)
  phys.breath = 0.5 - 0.5 * Math.cos(2 * Math.PI * clock / 4.2);
  state.ParamBreath = phys.breath;

  // корпус догоняет голову: PhysicsSetting1/2 (scale 0.33 / 0.25, weight 100)
  phys.bodyX = approach(phys.bodyX, 0.22 * state.ParamAngleX, dt, 0.25);
  phys.bodyY = approach(phys.bodyY, 0.17 * state.ParamAngleY, dt, 0.30);
  state.ParamBodyAngleX = clamp(phys.bodyX, -10, 10);
  state.ParamBodyAngleY = clamp(phys.bodyY, -10, 10);
  state.ParamBodyAngleZ = base.ParamBodyAngleZ;

  // глаза: лёгкий прищур от наклона головы и улыбки рта (PhysicsSetting3)
  const smileT = clamp(0.11 * clamp(state.ParamAngleY / 30, 0, 1) +
                       0.033 * clamp(state.ParamMouthForm, 0, 1), 0, 1);
  phys.smile = approach(phys.smile, smileT, dt, 0.2);
  state.ParamEyeSmileL = clamp(base.ParamEyeSmileL + phys.smile, 0, 1);
  state.ParamEyeSmileR = clamp(base.ParamEyeSmileR + phys.smile, 0, 1);

  // лёгкий довод формы рта от открытия (PhysicsSetting4)
  phys.form = approach(phys.form, 0.042 * state.ParamMouthOpenY, dt, 0.1);
  state.ParamMouthForm = clamp(base.ParamMouthForm + phys.form, -1, 1);
}

function addGesture(name) {
  const g = GESTURES[name];
  if (!g) return;
  gestures = gestures.filter((x) => x.id !== g.id);
  gestures.push({ id: g.id, dur: g.dur, keys: g.keys, t0: clock, base: state[g.id] });
}

function updateGestures() {
  if (!gestures.length) return;
  const alive = [];
  for (const g of gestures) {
    const p = (clock - g.t0) / g.dur;
    if (p > 1) continue;
    let v = g.keys[g.keys.length - 1][1];
    for (let i = 0; i < g.keys.length - 1; i++) {
      const a = g.keys[i], b = g.keys[i + 1];
      if (p >= a[0] && p <= b[0]) {
        v = a[1] + (b[1] - a[1]) * ((p - a[0]) / Math.max(1e-6, b[0] - a[0]));
        break;
      }
    }
    state[g.id] = clamp(g.base + v, byId[g.id].min, byId[g.id].max);
    alive.push(g);
  }
  gestures = alive;
}

function update(dt) {
  clock += dt;
  // база = значения ползунков/экспрессий
  for (const p of params) state[p.id] = base[p.id];
  updateTracking(dt);
  updateMouth(dt);
  updateBlink(dt);
  updatePhysics(dt);
  updateGestures();
  for (const p of params) state[p.id] = clamp(state[p.id], p.min, p.max);
}

/* ------------------------------------------------------------------ */
/* UI                                                                  */
/* ------------------------------------------------------------------ */

function ownerOf(id) {
  if ((id === 'ParamAngleX' || id === 'ParamAngleY' || id === 'ParamAngleZ')) {
    if (sources.track) return 'трекер';
    if (sources.idle) return 'idle';
    return null;
  }
  if (id === 'ParamEyeLOpen' || id === 'ParamEyeROpen') return sources.blink ? 'моргание' : null;
  if (id === 'ParamMouthOpenY') return sources.mic ? 'микрофон' : (sources.talk ? 'речь' : null);
  if (id === 'ParamBodyAngleX' || id === 'ParamBodyAngleY') return sources.physics ? 'физика' : null;
  if (id === 'ParamBreath') return sources.physics ? 'физика' : null;
  return null;
}

function buildParams() {
  const host = $('#params');
  const groups = [];
  params.forEach((p) => { if (groups.indexOf(p.group) < 0) groups.push(p.group); });
  groups.forEach((g) => {
    const box = document.createElement('div');
    box.className = 'param-group';
    const h = document.createElement('h3');
    h.textContent = g;
    box.appendChild(h);
    params.filter((p) => p.group === g).forEach((p) => {
      const row = document.createElement('div');
      row.className = 'param';
      row.innerHTML =
        '<div class="param-top"><label for="p-' + p.id + '"><b>' + p.id + '</b>' +
        '<span>' + p.label + '</span></label><output id="o-' + p.id + '"></output></div>' +
        '<input type="range" id="p-' + p.id + '" min="' + p.min + '" max="' + p.max +
        '" step="' + ((p.max - p.min) / 200).toFixed(4) + '" value="' + p.default + '">' +
        '<div class="param-note">' + (p.vts || '') + '</div>';
      box.appendChild(row);
      const input = row.querySelector('input');
      input.addEventListener('input', () => {
        base[p.id] = parseFloat(input.value);
        $('#o-' + p.id).textContent = base[p.id].toFixed(2);
      });
      rows[p.id] = { row: row, input: input, out: row.querySelector('output'),
                     note: row.querySelector('.param-note') };
    });
    host.appendChild(box);
  });
}

let uiAcc = 0;
function syncUI(dt) {
  uiAcc += dt;
  if (uiAcc < 1 / 24) return;
  uiAcc = 0;
  for (const p of params) {
    const r = rows[p.id], v = state[p.id];
    if (document.activeElement !== r.input) r.input.value = v;
    r.out.textContent = v.toFixed(2);
    const owner = ownerOf(p.id);
    r.row.classList.toggle('auto', !!owner);
    const note = p.vts + (owner ? ' · источник: <span class="src-tag">' + owner + '</span>' : '');
    if (r.note.dataset.txt !== note) { r.note.innerHTML = note; r.note.dataset.txt = note; }
  }
  if (opts.hud) {
    const lines = params.map((p) => {
      const short = p.id.replace('Param', '');
      const owner = ownerOf(p.id);
      return '<b>' + short + '</b> ' + state[p.id].toFixed(2) +
        (owner ? ' <i>← ' + owner + '</i>' : '');
    });
    $('#hud').innerHTML = lines.join('<br>');
  }
}

function applyExpression(name) {
  const expr = EXPRESSIONS[name] || {};
  params.forEach((p) => { base[p.id] = p.default; });
  Object.keys(expr).forEach((id) => {
    if (byId[id]) base[id] = clamp(expr[id], byId[id].min, byId[id].max);
  });
  $$('#expr-row button').forEach((b) => b.classList.toggle('on', b.dataset.expr === name));
}

function resetPose() {
  applyExpression('neutral');
  view.size = 1; view.x = 0; view.y = 0; view.rot = 0;
  $('#v-size').value = 100; $('#v-x').value = 0; $('#v-y').value = 0; $('#v-rot').value = 0;
  syncViewOutputs();
}

function syncViewOutputs() {
  $('#o-size').textContent = Math.round(view.size * 100) + '%';
  $('#o-x').textContent = Math.round(view.x) + '%';
  $('#o-y').textContent = Math.round(view.y) + '%';
  $('#o-rot').textContent = Math.round(view.rot) + '°';
}

function toggle(btn, key, onChange) {
  btn.addEventListener('click', () => {
    const next = !sources[key];
    if (onChange) { const ok = onChange(next); if (ok === false) return; }
    sources[key] = next;
    btn.classList.toggle('on', next);
    btn.setAttribute('aria-pressed', next ? 'true' : 'false');
  });
}

function wireUI() {
  // фон
  $$('.bg-chips button').forEach((b) => b.addEventListener('click', () => {
    view.bg = b.dataset.bg;
    $$('.bg-chips button').forEach((x) => x.classList.toggle('on', x === b));
  }));

  // источники
  toggle($('#t-track'), 'track');
  toggle($('#t-idle'), 'idle');
  toggle($('#t-blink'), 'blink');
  toggle($('#t-physics'), 'physics');
  toggle($('#t-talk'), 'talk');
  $('#t-mic').addEventListener('click', () => {
    if (sources.mic) { stopMic(); return; }
    startMic();
  });

  // оверлеи
  $('#t-hud').addEventListener('click', () => {
    opts.hud = !opts.hud;
    $('#hud').hidden = !opts.hud;
    $('#t-hud').classList.toggle('on', opts.hud);
    $('#t-hud').setAttribute('aria-pressed', opts.hud ? 'true' : 'false');
  });
  $('#t-wire').addEventListener('click', () => {
    opts.wire = !opts.wire;
    $('#t-wire').classList.toggle('on', opts.wire);
    $('#t-wire').setAttribute('aria-pressed', opts.wire ? 'true' : 'false');
  });

  // экспрессии и жесты
  $$('#expr-row button').forEach((b) => b.addEventListener('click', () => applyExpression(b.dataset.expr)));
  $$('#gest-row button[data-gesture]').forEach((b) => b.addEventListener('click', () => {
    if (b.dataset.gesture === 'blink') startBlink();
    else addGesture(b.dataset.gesture);
  }));
  $('#btn-reset').addEventListener('click', resetPose);

  // позиция и размер
  $('#v-size').addEventListener('input', (e) => { view.size = +e.target.value / 100; syncViewOutputs(); });
  $('#v-x').addEventListener('input', (e) => { view.x = +e.target.value; syncViewOutputs(); });
  $('#v-y').addEventListener('input', (e) => { view.y = +e.target.value; syncViewOutputs(); });
  $('#v-rot').addEventListener('input', (e) => { view.rot = +e.target.value; syncViewOutputs(); });

  // настройки
  $('#opt-smooth').addEventListener('input', (e) => {
    opts.smooth = +e.target.value;
    $('#oo-smooth').textContent = opts.smooth.toFixed(2) + ' с';
  });
  $('#opt-amp').addEventListener('input', (e) => {
    opts.amp = +e.target.value / 100;
    $('#oo-amp').textContent = e.target.value + '%';
  });
  $('#opt-blink').addEventListener('input', (e) => {
    opts.blinkMax = +e.target.value;
    $('#oo-blink').textContent = '1.6–' + opts.blinkMax.toFixed(1) + ' с';
  });
  $('#opt-gain').addEventListener('input', (e) => {
    opts.gain = +e.target.value;
    $('#oo-gain').textContent = '×' + opts.gain.toFixed(1);
  });
  $('#opt-gate').addEventListener('input', (e) => {
    opts.gate = +e.target.value;
    $('#oo-gate').textContent = opts.gate.toFixed(2);
  });

  // кадр
  $('#btn-shot').addEventListener('click', () => {
    canvas.toBlob((blob) => {
      if (!blob) return;
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = 'ChibiVT_vts_frame.png';
      document.body.appendChild(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(a.href), 4000);
    }, 'image/png');
  });
  $('#btn-copy').addEventListener('click', async () => {
    const pose = {};
    params.forEach((p) => { pose[p.id] = Math.round(state[p.id] * 1000) / 1000; });
    const text = JSON.stringify(pose, null, 1);
    try {
      await navigator.clipboard.writeText(text);
      $('#btn-copy').textContent = 'Скопировано ✓';
    } catch (err) {
      $('#btn-copy').textContent = 'См. консоль';
      console.log(text);
    }
    setTimeout(() => { $('#btn-copy').textContent = 'Скопировать JSON позы'; }, 1600);
  });

  // сцена: трекинг курсором + перетаскивание модели
  const stage = $('#stage');
  let dragging = false, dragFrom = null;
  stage.addEventListener('pointermove', (e) => {
    const r = stage.getBoundingClientRect();
    const nx = ((e.clientX - r.left) / r.width) * 2 - 1;
    const ny = ((e.clientY - r.top) / r.height) * 2 - 1;
    if (dragging && dragFrom) {
      view.x = clamp(dragFrom.x + (e.clientX - dragFrom.px) / r.width * 100, -50, 50);
      view.y = clamp(dragFrom.y + (e.clientY - dragFrom.py) / r.height * 100, -50, 50);
      $('#v-x').value = Math.round(view.x); $('#v-y').value = Math.round(view.y);
      syncViewOutputs();
    }
    track.tx = clamp(nx, -1, 1);
    track.ty = clamp(ny, -1, 1);
    track.inside = true;
  });
  stage.addEventListener('pointerdown', (e) => {
    dragging = true;
    stage.classList.add('dragging');
    dragFrom = { px: e.clientX, py: e.clientY, x: view.x, y: view.y };
    stage.setPointerCapture(e.pointerId);
  });
  stage.addEventListener('pointerup', (e) => {
    dragging = false;
    stage.classList.remove('dragging');
    try { stage.releasePointerCapture(e.pointerId); } catch (err) { /* не страшно */ }
  });
  stage.addEventListener('pointerleave', () => { track.inside = false; });

  // сайдбар: переход к разделам
  $$('[data-jump]').forEach((b) => b.addEventListener('click', () => {
    const el = document.getElementById(b.dataset.jump);
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    $$('[data-jump]').forEach((x) => x.classList.toggle('active', x === b));
  }));

  // горячие клавиши
  window.addEventListener('keydown', (e) => {
    if (e.target && /input|textarea|select/i.test(e.target.tagName)) return;
    const names = Object.keys(EXPRESSIONS);
    if (e.key >= '1' && e.key <= '6') applyExpression(names[+e.key - 1]);
    else if (e.key === 'b' || e.key === 'B' || e.key === 'и' || e.key === 'И') startBlink();
    else if (e.key === 'r' || e.key === 'R' || e.key === 'к' || e.key === 'К') resetPose();
  });

  $('#hud').hidden = !opts.hud;
  syncViewOutputs();
  $('#oo-blink').textContent = '1.6–' + opts.blinkMax.toFixed(1) + ' с';
  $('#oo-smooth').textContent = opts.smooth.toFixed(2) + ' с';
}

function fillGroupsTable() {
  const table = $('#groups-table');
  const head = '<thead><tr><th>Группа VTube Studio</th><th>Параметры</th></tr></thead>';
  const body = (manifest.vts_groups || []).map((g) =>
    '<tr><td><b>' + g.name + '</b></td><td>' +
    g.ids.map((i) => '<code>' + i + '</code>').join(' ') + '</td></tr>').join('');
  table.innerHTML = head + '<tbody>' + body + '</tbody>';
}

/* ------------------------------------------------------------------ */
/* микрофон                                                            */
/* ------------------------------------------------------------------ */

async function startMic() {
  try {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error('браузер не даёт доступ к микрофону');
    }
    mic.stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    });
    const Ctx = window.AudioContext || window.webkitAudioContext;
    mic.ctx = new Ctx();
    mic.analyser = mic.ctx.createAnalyser();
    mic.analyser.fftSize = 1024;
    mic.analyser.smoothingTimeConstant = 0.2;
    mic.ctx.createMediaStreamSource(mic.stream).connect(mic.analyser);
    mic.buf = new Uint8Array(mic.analyser.fftSize);
    sources.mic = true;
    sources.talk = false;
    $('#t-talk').classList.remove('on');
    $('#t-talk').setAttribute('aria-pressed', 'false');
    $('#t-mic').classList.add('on');
    $('#t-mic').setAttribute('aria-pressed', 'true');
    $('#mic-error').hidden = true;
  } catch (err) {
    const box = $('#mic-error');
    box.hidden = false;
    box.textContent = 'Микрофон недоступен: ' + err.message +
      ' — можно включить «Речь (симуляция)», липсинк работает по тому же параметру ParamMouthOpenY.';
    stopMic();
  }
}

function stopMic() {
  sources.mic = false;
  $('#t-mic').classList.remove('on');
  $('#t-mic').setAttribute('aria-pressed', 'false');
  if (mic.stream) mic.stream.getTracks().forEach((t) => t.stop());
  if (mic.ctx) mic.ctx.close().catch(() => {});
  mic.stream = null; mic.analyser = null; mic.ctx = null; mic.level = 0; mic.mouth = 0;
}

/* ------------------------------------------------------------------ */
/* старт                                                               */
/* ------------------------------------------------------------------ */

function fail(msg) {
  const box = $('#error');
  box.hidden = false;
  box.textContent = msg;
}

function frame(now) {
  const dt = lastT ? Math.min(0.1, (now - lastT) / 1000) : 0.016;
  lastT = now;
  update(dt);
  compositeFace();
  draw();
  syncUI(dt);
  fpsAcc += dt; fpsFrames++;
  if (fpsAcc >= 0.5) {
    fpsShow = Math.round(fpsFrames / fpsAcc);
    fpsAcc = 0; fpsFrames = 0;
    $('#fps').textContent = fpsShow + ' fps';
  }
  requestAnimationFrame(frame);
}

(async function init() {
  if (!gl) { fail('WebGL недоступен — предпросмотр не может отрисовать модель.'); return; }
  try {
    initGL();
    await loadAssets();
  } catch (err) {
    fail('Не удалось загрузить пакет предпросмотра: ' + err.message +
      ' — соберите его: python3 tools/build_vts_demo.py (и запустите сайт: python3 tools/serve.py . 8000)');
    return;
  }
  headMesh = buildMesh(headGridDef, assets.head, manifest.canvas.head_rect, makeTexture(faceCanvas));
  bodyMesh = buildMesh(bodyGridDef, assets.body, manifest.canvas.body_rect, makeTexture(assets.body));
  buildParams();
  fillGroupsTable();
  wireUI();
  applyExpression('neutral');
  requestAnimationFrame(frame);
})();

})();

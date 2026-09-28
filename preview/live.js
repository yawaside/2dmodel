/* Cubism-Core-free live preview.
   Re-implements (in JS) the exact same continuous math tools/build_model.py
   bakes into ChibiVT.moc3 keyframes: head_warp()/body_warp()/mouth_blend()/
   eye_blend()/shading ramps. Loads dist/ChibiVT/rig_web.json (mesh topology
   + UVs, exported by build_model.py) and dist/ChibiVT/textures/texture_00.png
   (same atlas the real model uses, shading tiles included). */
(() => {
const RIG_URL = '../dist/ChibiVT/rig_web.json?v=' + Date.now();
const TEX_BASE = '../dist/ChibiVT/textures/';

const err = document.getElementById('err');
const status = document.getElementById('status');
const canvas = document.getElementById('cv');
const gl = canvas.getContext('webgl', { premultipliedAlpha: true, alpha: true });
if (!gl) { err.textContent = 'WebGL недоступен в этом браузере'; return; }

function fail(e) { err.textContent = String(e && e.stack || e); console.error(e); }

// ---------------------------------------------------------------- shaders -- //
const VS = `
attribute vec2 a_pos; attribute vec2 a_uv; varying vec2 v_uv;
void main(){ v_uv = a_uv; gl_Position = vec4(a_pos, 0.0, 1.0); }`;
const FS = `
precision mediump float; varying vec2 v_uv; uniform sampler2D u_tex; uniform float u_alpha;
void main(){ vec4 c = texture2D(u_tex, v_uv); gl_FragColor = vec4(c.rgb * c.a, c.a) * u_alpha; }`;

function shader(type, src) {
  const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}
const prog = gl.createProgram();
gl.attachShader(prog, shader(gl.VERTEX_SHADER, VS));
gl.attachShader(prog, shader(gl.FRAGMENT_SHADER, FS));
gl.linkProgram(prog); gl.useProgram(prog);
const A_POS = gl.getAttribLocation(prog, 'a_pos'), A_UV = gl.getAttribLocation(prog, 'a_uv');
const U_TEX = gl.getUniformLocation(prog, 'u_tex'), U_ALPHA = gl.getUniformLocation(prog, 'u_alpha');
gl.enable(gl.BLEND);
gl.blendFuncSeparate(gl.ONE, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

// ------------------------------------------------------------------ math -- //
const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
function smoothstep(e0, e1, x) { const t = clamp((x - e0) / (e1 - e0), 0, 1); return t * t * (3 - 2 * t); }
function rot(px, py, cx, cy, ang) {
  const s = Math.sin(ang), c = Math.cos(ang), dx = px - cx, dy = py - cy;
  return [cx + dx * c - dy * s, cy + dx * s + dy * c];
}
const D2R = Math.PI / 180;

function headWarp(px, py, ax, ay, az, C) {
  const w = smoothstep(C.cut_y, C.cut_y - C.head_fade, py);
  const [hcx, hcy] = C.head_center;
  const th = ax * D2R * C.yaw_scale;
  const u = clamp((px - hcx) / C.r_yaw, -1, 1);
  const phi = Math.asin(u);
  const x1 = px + C.r_yaw * (Math.sin(phi + th) - Math.sin(phi));
  const y1 = py;
  const psi = -(ay * D2R) * C.pitch_scale;
  const v = clamp((y1 - hcy) / C.r_pitch, -1, 1);
  const a = Math.asin(v);
  const ca = Math.max(Math.cos(a), 0.25);
  const y2 = y1 + C.r_pitch * (Math.sin(a + psi) - Math.sin(a));
  const x2 = x1 + (x1 - hcx) * (Math.cos(a + psi) / ca - 1.0);
  const [x3, y3] = rot(x2, y2, C.neck_point[0], C.neck_point[1], az * D2R * C.roll_scale);
  return [px + (x3 - px) * w, py + (y3 - py) * w];
}

function bodyWarp(px, py, bx, by, bz, breath, C) {
  const xn = bx / 10, yn = by / 10, zn = bz / 10;
  const t = clamp((py - 600) / 423, 0, 1);
  let x = px + C.body_shift_x * xn * t;
  x = 512 + (x - 512) * (1 - 0.045 * Math.abs(xn));
  let y = py - C.body_shift_y * yn * (1 - t);
  y -= breath * 2.0 * t;
  const w = smoothstep(500, 900, py);
  return rot(x, y, C.body_hip[0], C.body_hip[1], -(C.body_roll_deg * D2R) * zn * w);
}

function bodyXY(px, py, ax, bx, by, bz, breath, W, C) {
  const ym = W - py;
  let [wx, wym] = bodyWarp(px, ym, bx, by, bz, breath, C);
  wx += 0.30 * ax;
  return [px + (wx - px), py - (wym - ym)];
}

function headXY(px, py, ax, ay, az, bx, by, bz, breath, W, C) {
  const pyBob = py - breath * 3.0;
  const [wx, wy] = headWarp(px, pyBob, ax, ay, az, C);
  return bodyXY(wx, wy, ax, bx, by, bz, breath, W, C);
}

// mouth_blend / eye_blend / stack_opa ported 1:1 from tools/build_model.py
function mouthBlend(openY, form) {
  const w = [0, 0, 0, 0, 0, 0];
  const o = clamp(openY, 0, 1), f = clamp(form, -1, 1);
  if (o <= 0.12) w[0] = 1.0;
  else if (o <= 0.20) { const t = (o - 0.12) / 0.08; w[0] = 1 - t; w[1] = t; }
  else if (o < 0.38) w[1] = 1.0;
  else if (o <= 0.47) { const t = (o - 0.38) / 0.09; w[1] = 1 - t; w[2] = t; }
  else if (o < 0.66) w[2] = 1.0;
  else if (o <= 0.76) { const t = (o - 0.66) / 0.10; w[2] = 1 - t; w[3] = t; }
  else w[3] = 1.0;
  if (f > 0) { for (let i = 0; i < 4; i++) w[i] *= (1 - f); w[4] = f; }
  else if (f < 0) { const k = -f; for (let i = 0; i < 4; i++) w[i] *= (1 - k); w[5] = k; }
  const total = w.reduce((a, b) => a + b, 0);
  if (total > 0) for (let i = 0; i < 6; i++) w[i] /= total; else w[0] = 1;
  return w;
}
function eyeBlend(openVal, smileVal) {
  const w = [0, 0, 0, 0, 0];
  const o = clamp(openVal, 0, 1), s = clamp(smileVal, 0, 1);
  if (o < 0.2) { w[2] = 1 - o / 0.2; w[1] = o / 0.2; }
  else if (o < 0.5) { const t = (o - 0.2) / 0.3; w[1] = 1 - t; w[0] = t; }
  else w[0] = 1.0;
  if (s > 0) {
    const squint = s * Math.max(0, 1 - o);
    for (let i = 0; i < 3; i++) w[i] *= (1 - s);
    w[3] += s; w[4] += squint * 0.3;
  }
  const total = w.reduce((a, b) => a + b, 0);
  if (total > 0) for (let i = 0; i < 5; i++) w[i] /= total; else w[0] = 1;
  return w;
}
function stackOpa(weights, k) {
  let cum = 0; for (let j = 0; j <= k; j++) cum += weights[j];
  if (cum <= 1e-6) return 0;
  return clamp(weights[k] / cum, 0, 1);
}

// --------------------------------------------------------------- meshes -- //
function makeGroup(rest, tris, uvSets, W) {
  const n = rest.length;
  const posBuf = gl.createBuffer();
  const idxBuf = gl.createBuffer();
  const idx = new Uint16Array(tris.length * 3);
  tris.forEach((t, i) => { idx[i * 3] = t[0]; idx[i * 3 + 1] = t[1]; idx[i * 3 + 2] = t[2]; });
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idxBuf);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idx, gl.STATIC_DRAW);

  const uvBufs = {};
  for (const name in uvSets) {
    const arr = new Float32Array(n * 2);
    uvSets[name].forEach((uv, i) => { arr[i * 2] = uv[0]; arr[i * 2 + 1] = uv[1]; });
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, arr, gl.STATIC_DRAW);
    uvBufs[name] = buf;
  }

  const posArray = new Float32Array(n * 2);
  return {
    n, idxBuf, idxCount: idx.length, uvBufs, posBuf, posArray, rest,
    updatePositions(warpFn) {
      for (let i = 0; i < n; i++) {
        const [px, py] = rest[i];
        const [x, y] = warpFn(px, py);
        posArray[i * 2] = (x / W) * 2 - 1;
        posArray[i * 2 + 1] = 1 - (y / W) * 2;
      }
      gl.bindBuffer(gl.ARRAY_BUFFER, this.posBuf);
      gl.bufferData(gl.ARRAY_BUFFER, this.posArray, gl.DYNAMIC_DRAW);
    },
    draw(uvName, opacity) {
      if (opacity <= 0.002) return;
      gl.bindBuffer(gl.ARRAY_BUFFER, this.posBuf);
      gl.vertexAttribPointer(A_POS, 2, gl.FLOAT, false, 0, 0);
      gl.enableVertexAttribArray(A_POS);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.uvBufs[uvName]);
      gl.vertexAttribPointer(A_UV, 2, gl.FLOAT, false, 0, 0);
      gl.enableVertexAttribArray(A_UV);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.idxBuf);
      gl.uniform1f(U_ALPHA, opacity);
      gl.drawElements(gl.TRIANGLES, this.idxCount, gl.UNSIGNED_SHORT, 0);
    },
  };
}

function loadTexture(url) {
  return new Promise((res, rej) => {
    const img = new Image();
    img.onload = () => {
      const t = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      res(t);
    };
    img.onerror = () => rej(new Error('texture failed: ' + url));
    img.src = url;
  });
}

// -------------------------------------------------------------- UI wiring -- //
const ui = {};
['ax', 'ay', 'az', 'mo', 'mf', 'el', 'er', 'es'].forEach(id => {
  ui[id] = document.getElementById(id);
  const out = document.getElementById(id + 'V');
  const upd = () => { out.textContent = (+ui[id].value).toFixed(2); };
  ui[id].addEventListener('input', upd); upd();
});
const state = {
  idle: true, mouse: true, talk: false, mic: false,
  blinkT: 0, nextBlink: 2 + Math.random() * 3, blinkPhase: 0,
  mouseX: 0, mouseY: 0, micLevel: 0, micGain: 2.2,
};

function toggleBtn(id, key) {
  const b = document.getElementById(id);
  b.addEventListener('click', () => {
    state[key] = !state[key];
    b.classList.toggle('on', state[key]);
    b.textContent = b.textContent.replace(/вкл|выкл/, state[key] ? 'вкл' : 'выкл');
  });
}
document.getElementById('idleBtn').addEventListener('click', function () {
  state.idle = !state.idle; this.classList.toggle('on', state.idle);
  this.textContent = 'Авто-движение: ' + (state.idle ? 'вкл' : 'выкл');
});
document.getElementById('mouseBtn').addEventListener('click', function () {
  state.mouse = !state.mouse; this.classList.toggle('on', state.mouse);
  this.textContent = 'Следить за курсором: ' + (state.mouse ? 'вкл' : 'выкл');
});
document.getElementById('talkBtn').addEventListener('click', function () {
  state.talk = !state.talk; this.classList.toggle('on', state.talk);
  this.textContent = state.talk ? 'Демо-речь: выкл' : 'Демо-речь (без мика)';
});
document.getElementById('blinkBtn').addEventListener('click', () => { state.blinkPhase = 0.001; });

document.getElementById('micGain').addEventListener('input', function () {
  state.micGain = +this.value; document.getElementById('micGainV').textContent = this.value;
});
document.getElementById('micGainV').textContent = document.getElementById('micGain').value;

let audioCtx = null, analyser = null, micData = null;
document.getElementById('micBtn').addEventListener('click', async function () {
  if (state.mic) { state.mic = false; this.classList.remove('on'); this.textContent = '🎤 Включить микрофон'; return; }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const src = audioCtx.createMediaStreamSource(stream);
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 1024;
    micData = new Uint8Array(analyser.fftSize);
    src.connect(analyser);
    state.mic = true; this.classList.add('on'); this.textContent = '🎤 Микрофон: слушаю…';
  } catch (e) { fail(e); }
});

canvas.addEventListener('mousemove', (e) => {
  const r = canvas.getBoundingClientRect();
  state.mouseX = clamp((e.clientX - r.left) / r.width * 2 - 1, -1, 1);
  state.mouseY = clamp((e.clientY - r.top) / r.height * 2 - 1, -1, 1);
});

// ------------------------------------------------------------------ main -- //
async function init() {
  status.textContent = 'загрузка rig_web.json…';
  const rig = await fetch(RIG_URL).then(r => { if (!r.ok) throw new Error('rig_web.json: HTTP ' + r.status); return r.json(); });
  status.textContent = 'загрузка текстуры…';
  const tex = await loadTexture(TEX_BASE + rig.texture + '?v=' + Date.now());

  const W = rig.canvas_px;
  const C = rig.cfg;

  const head = makeGroup(rig.head.verts, rig.head.tris, {
    base: rig.head.uv,
    yaw_pos: rig.shade.yaw_pos_uv, yaw_neg: rig.shade.yaw_neg_uv,
    pitch_pos: rig.shade.pitch_pos_uv, pitch_neg: rig.shade.pitch_neg_uv,
  }, W);
  const body = makeGroup(rig.body.verts, rig.body.tris, { base: rig.body.uv }, W);
  const mouthUv = {}; rig.mouth.shapes.forEach((s, i) => { mouthUv['m' + i] = rig.mouth.uvs[i]; });
  const mouth = makeGroup(rig.mouth.verts, rig.mouth.tris, mouthUv, W);
  const eyes = rig.eyes.map(e => {
    const uvSets = {}; e.variants.forEach((v, i) => { uvSets[v] = e.uvs[i]; });
    return makeGroup(e.verts, e.tris, uvSets, W);
  });

  status.textContent = rig.head.verts.length + rig.body.verts.length + ' вершин · ' +
    (rig.head.tris.length + rig.body.tris.length) + ' треугольников';

  let last = performance.now(), fpsAcc = 0, fpsN = 0;
  const fpsEl = document.getElementById('fps');
  let bx = 0, by = 0, bz = 0; // smoothed body-follow state
  let curAx = 0, curAy = 0, curAz = 0;
  let smileAuto = 0;

  function frame(now) {
    const dt = Math.min(0.05, (now - last) / 1000); last = now;
    fpsAcc += dt; fpsN++;
    if (fpsAcc > 0.5) { fpsEl.textContent = (fpsN / fpsAcc).toFixed(0) + ' fps'; fpsAcc = 0; fpsN = 0; }

    const t = now / 1000;
    const breath = 0.5 + 0.5 * Math.sin(t * 1.1);

    // ---- target head angles ----
    let targetAx = +ui.ax.value, targetAy = +ui.ay.value, targetAz = +ui.az.value;
    if (state.mouse) {
      targetAx = state.mouseX * 26; targetAy = -state.mouseY * 20; targetAz = state.mouseX * -6;
    } else if (state.idle) {
      targetAx = Math.sin(t * 0.35) * 14 + Math.sin(t * 0.9) * 4;
      targetAy = Math.sin(t * 0.5 + 1.3) * 8;
      targetAz = Math.sin(t * 0.27 + 0.6) * 5;
    }
    // smooth head motion a bit so idle/mouse doesn't snap
    curAx += (targetAx - curAx) * Math.min(1, dt * 6);
    curAy += (targetAy - curAy) * Math.min(1, dt * 6);
    curAz += (targetAz - curAz) * Math.min(1, dt * 6);
    if (!state.mouse && !state.idle) { curAx = targetAx; curAy = targetAy; curAz = targetAz; }

    // ---- body follows head with lag (mirrors physics3.json intent) ----
    bx += ((curAx * 0.33) - bx) * Math.min(1, dt * 3.2);
    by += ((curAy * 0.25) - by) * Math.min(1, dt * 3.2);
    bz += ((curAz * 0.40) - bz) * Math.min(1, dt * 3.2);

    // ---- mouth: mic > demo-talk > slider ----
    let openY = +ui.mo.value, form = +ui.mf.value;
    if (state.mic && analyser) {
      analyser.getByteTimeDomainData(micData);
      let sum = 0;
      for (let i = 0; i < micData.length; i++) { const v = (micData[i] - 128) / 128; sum += v * v; }
      const rms = Math.sqrt(sum / micData.length);
      const target = clamp(rms * state.micGain * 3.2, 0, 1);
      state.micLevel += (target - state.micLevel) * Math.min(1, dt * 18);
      document.getElementById('micMeter').style.width = (state.micLevel * 100).toFixed(0) + '%';
      openY = state.micLevel;
    } else if (state.talk) {
      openY = clamp(Math.max(0, Math.sin(t * 7.0)) * 0.8 + Math.max(0, Math.sin(t * 13.0)) * 0.3, 0, 1);
      form = Math.sin(t * 0.6) * 0.3;
    }

    // ---- eyes: blink timer + sliders ----
    state.blinkT += dt;
    let blink = 0;
    if (state.blinkPhase > 0) { state.blinkPhase += dt; }
    if (state.idle || state.mouse) {
      if (state.blinkT > state.nextBlink) { state.blinkT = 0; state.nextBlink = 2.5 + Math.random() * 4; state.blinkPhase = 0.0001; }
    }
    if (state.blinkPhase > 0) {
      const bt = state.blinkPhase;
      blink = bt < 0.09 ? bt / 0.09 : (bt < 0.16 ? 1 - (bt - 0.09) / 0.07 : 0);
      if (bt > 0.2) state.blinkPhase = 0;
    }
    const elOpen = clamp(+ui.el.value * (1 - blink), 0, 1);
    const erOpen = clamp(+ui.er.value * (1 - blink), 0, 1);
    const smile = +ui.es.value;

    // ---- warp + draw ----
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.uniform1i(U_TEX, 0);

    const bodyWarpFn = (px, py) => bodyXY(px, py, curAx, bx, by, bz, breath, W, C);
    const headWarpFn = (px, py) => headXY(px, py, curAx, curAy, curAz, bx, by, bz, breath, W, C);

    body.updatePositions(bodyWarpFn);
    body.draw('base', 1);

    head.updatePositions(headWarpFn);
    head.draw('base', 1);
    const yawMax = C.shade_yaw_max_deg, pitchMax = C.shade_pitch_max_deg;
    head.draw('yaw_pos', clamp(curAx, 0, yawMax) / yawMax);
    head.draw('yaw_neg', clamp(-curAx, 0, yawMax) / yawMax);
    head.draw('pitch_pos', clamp(curAy, 0, pitchMax) / pitchMax);
    head.draw('pitch_neg', clamp(-curAy, 0, pitchMax) / pitchMax);

    mouth.updatePositions(headWarpFn);
    const mw = mouthBlend(openY, form);
    for (let i = 0; i < mw.length; i++) mouth.draw('m' + i, stackOpa(mw, i));

    const ayNat = clamp(smile + Math.max(0, curAy / 30) * 0.3, 0, 1);
    const ewL = eyeBlend(elOpen, ayNat), ewR = eyeBlend(erOpen, ayNat);
    const variants = rig.eyes[0].variants;
    eyes[0].updatePositions(headWarpFn);
    eyes[1].updatePositions(headWarpFn);
    for (let i = 0; i < variants.length; i++) eyes[0].draw(variants[i], stackOpa(ewL, i));
    for (let i = 0; i < variants.length; i++) eyes[1].draw(variants[i], stackOpa(ewR, i));

    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

init().catch(fail);
})();

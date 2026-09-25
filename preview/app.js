/* Minimal Cubism-4 preview renderer (WebGL) - dev helper only.
   Needs Live2D Cubism Core (lib/live2dcubismcore.min.js), which is NOT part of
   this repository (Live2D proprietary licence). */
(() => {
// The kit is rebuilt constantly while rigging.  Without a cache-buster the
// browser serves a fresh .moc3 against the PREVIOUS texture - the model then
// looks completely broken (e.g. an eye where the mouth should be).
const BUST = '?v=' + Date.now();
const MODEL_URL = '../dist/ChibiVT/ChibiVT.model3.json' + BUST;
const BASE = MODEL_URL.replace(/[^/]*$/, '').replace(/\?.*$/, '');
const cfg = { scale: 1.35, flipV: false };

const err = document.getElementById('err');
const status = document.getElementById('status');
const canvas = document.getElementById('cv');
// The shader premultiplies, so the drawing buffer has to be premultiplied too
// (otherwise every feathered edge and every cross-fade goes dark).
const gl = canvas.getContext('webgl', { premultipliedAlpha: true, alpha: true });
if (!gl) { err.textContent = 'WebGL недоступен'; return; }

const VS = `
attribute vec2 a_pos; attribute vec2 a_uv; varying vec2 v_uv;
uniform vec2 u_scale; uniform vec2 u_off;
void main(){ v_uv = a_uv; gl_Position = vec4(a_pos*u_scale + u_off, 0.0, 1.0); }`;
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
const U_SCALE = gl.getUniformLocation(prog, 'u_scale'), U_OFF = gl.getUniformLocation(prog, 'u_off');
const U_TEX = gl.getUniformLocation(prog, 'u_tex'), U_ALPHA = gl.getUniformLocation(prog, 'u_alpha');
gl.enable(gl.BLEND);
// premultiplied source: dst = src + dst * (1 - src.a)
gl.blendFuncSeparate(gl.ONE, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

let model = null, textures = [], drawables = [];

function loadTexture(url) {
  return new Promise((res, rej) => {
    const img = new Image();
    img.onload = () => {
      const t = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, t);
      // Cubism Core hands out GL-style UVs (v=0 = bottom row), so flip on upload
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
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

async function init() {
  status.textContent = 'загрузка model3.json…';
  const meta = await fetch(MODEL_URL).then(r => r.json());
  status.textContent = 'загрузка moc3…';
  const mocBuf = await fetch(BASE + meta.FileReferences.Moc + BUST).then(r => r.arrayBuffer());
  const moc = Live2DCubismCore.Moc.fromArrayBuffer(mocBuf);
  if (!moc) throw new Error('Core не смог прочитать moc3');
  model = Live2DCubismCore.Model.fromMoc(moc);
  if (!model) throw new Error('Core не смог создать модель');

  status.textContent = 'загрузка текстур…';
  textures = await Promise.all(meta.FileReferences.Textures.map(t => loadTexture(BASE + t + BUST)));

  const d = model.drawables;
  drawables = [];
  for (let i = 0; i < d.count; i++) {
    const uvs = d.vertexUvs[i];
    const uv = Float32Array.from(uvs);
    if (cfg.flipV) for (let k = 1; k < uv.length; k += 2) uv[k] = 1 - uv[k];
    const uvBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, uvBuf);
    gl.bufferData(gl.ARRAY_BUFFER, uv, gl.STATIC_DRAW);
    const idx = Uint16Array.from(d.indices[i]);
    const idxBuf = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idxBuf);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idx, gl.STATIC_DRAW);
    drawables.push({ id: d.ids[i], posBuf: gl.createBuffer(), uvBuf, idxBuf, n: idx.length, tex: d.textureIndices[i] });
  }
  model.update();
  buildUI();
  status.textContent = d.count + ' мешей, ' + model.parameters.count + ' параметров';
  requestAnimationFrame(frame);
}

function frame() {
  const d = model.drawables;
  model.update();
  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.clearColor(0, 0, 0, 0);
  gl.clear(gl.COLOR_BUFFER_BIT);
  const order = Array.from(d.renderOrders);
  const list = drawables.map((o, i) => ({ o, i, ro: order[i], op: d.opacities[i], vis: d.dynamicFlags[i] & 1 }));
  list.sort((a, b) => a.ro - b.ro);
  for (const it of list) {
    if (!it.vis || it.op <= 0.001) continue;
    const vp = d.vertexPositions[it.i];
    gl.bindBuffer(gl.ARRAY_BUFFER, it.o.posBuf);
    gl.bufferData(gl.ARRAY_BUFFER, vp, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(A_POS);
    gl.vertexAttribPointer(A_POS, 2, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, it.o.uvBuf);
    gl.enableVertexAttribArray(A_UV);
    gl.vertexAttribPointer(A_UV, 2, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, it.o.idxBuf);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, textures[it.o.tex] || textures[0]);
    gl.uniform1i(U_TEX, 0);
    gl.uniform2f(U_SCALE, cfg.scale, cfg.scale);
    gl.uniform2f(U_OFF, 0, 0);
    gl.uniform1f(U_ALPHA, it.op);
    gl.drawElements(gl.TRIANGLES, it.o.n, gl.UNSIGNED_SHORT, 0);
  }
  requestAnimationFrame(frame);
}

/* ------------------------------- UI ---------------------------------- */
const PARAM_LABELS = {
  ParamAngleX: 'поворот головы X',
  ParamAngleY: 'наклон головы Y',
  ParamAngleZ: 'крен головы Z',
  ParamBodyAngleX: 'корпус X',
  ParamBodyAngleY: 'корпус Y',
  ParamBodyAngleZ: 'корпус Z',
  ParamEyeLOpen: 'левый глаз',
  ParamEyeROpen: 'правый глаз',
  ParamMouthOpenY: 'открытие рта',
};
function buildUI() {
  const box = document.getElementById('sliders');
  box.innerHTML = '';
  model.parameters.ids.forEach((id, i) => {
    const row = document.createElement('div');
    row.className = 'row';
    const lo = model.parameters.minimumValues[i], hi = model.parameters.maximumValues[i];
    row.innerHTML = '<label><span>' + (PARAM_LABELS[id] || id) + '</span><span class="val"></span></label>' +
      '<input type="range" min="' + lo + '" max="' + hi + '" step="' + ((hi - lo) / 200) + '">';
    const inp = row.querySelector('input'), val = row.querySelector('.val');
    inp.value = model.parameters.defaultValues[i];
    val.textContent = (+inp.value).toFixed(1);
    inp.addEventListener('input', () => {
      val.textContent = (+inp.value).toFixed(1);
      manual[id] = true;
    });
    sliders[id] = inp; vals[id] = val;
    box.appendChild(row);
  });
}
const sliders = {}, vals = {}, manual = {};

function setParam(id, v) {
  const i = model.parameters.ids.indexOf(id);
  if (i < 0) return;
  model.parameters.values[i] = v;
  if (sliders[id]) { sliders[id].value = v; vals[id].textContent = v.toFixed(1); }
}

let auto = false, mouse = true;
document.getElementById('auto').onclick = (e) => {
  auto = !auto; e.target.textContent = 'Авто-движение: ' + (auto ? 'вкл' : 'выкл');
  if (auto) Object.keys(manual).forEach(k => delete manual[k]);
};
document.getElementById('mouse').onclick = (e) => {
  mouse = !mouse; e.target.textContent = 'Следовать за мышью: ' + (mouse ? 'вкл' : 'выкл');
};
const stage = document.getElementById('stage');
stage.addEventListener('mousemove', (e) => {
  if (!mouse) return;
  const r = stage.getBoundingClientRect();
  const nx = ((e.clientX - r.left) / r.width) * 2 - 1;
  const ny = ((e.clientY - r.top) / r.height) * 2 - 1;
  setParam('ParamAngleX', nx * 30);
  setParam('ParamAngleY', -ny * 30);
  setParam('ParamAngleZ', nx * 8);
  setParam('ParamBodyAngleX', nx * 6);
  setParam('ParamBodyAngleZ', nx * 3);
});

const poses = {
  reset: {},
  blink: { ParamEyeLOpen: 0, ParamEyeROpen: 0 },
  talk: { ParamMouthOpenY: 1 },
};
document.querySelectorAll('[data-pose]').forEach(b => b.onclick = () => {
  const p = poses[b.dataset.pose];
  model.parameters.ids.forEach(id => {
    if (id in p) setParam(id, p[id]);
    else if (id.startsWith('Param')) setParam(id, model.parameters.defaultValues[model.parameters.ids.indexOf(id)]);
  });
});

// idle animation loop
let t0 = performance.now();
setInterval(() => {
  if (!model) return;
  const t = (performance.now() - t0) / 1000;
  if (!auto) return;
  const bx = Math.max(0, 1 - Math.abs(((t % 3.2) - 0.25) * 4));
  if (!mouse) {
    setParam('ParamAngleX', Math.sin(t * 0.7) * 22);
    setParam('ParamAngleY', Math.sin(t * 0.53 + 1) * 14);
    setParam('ParamAngleZ', Math.sin(t * 0.41) * 8);
    setParam('ParamBodyAngleX', Math.sin(t * 0.7 - 0.5) * 5);
  }
  setParam('ParamEyeLOpen', 1 - bx);
  setParam('ParamEyeROpen', 1 - bx);
  setParam('ParamMouthOpenY', Math.max(0, Math.sin(t * 3.1) * 0.8));
}, 40);

init().catch(e => { err.textContent = e.stack || String(e); status.textContent = 'ошибка'; });
})();

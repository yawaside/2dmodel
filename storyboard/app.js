/* Chibi Studio · интерактивная раскадровка глаз и рта.
   Блендинг — общий файл blend.js, точный порт математики .moc3
   (mouth_blend / eye_blend / stack_opa из tools/build_model.py). */
'use strict';

const $ = (sel) => document.querySelector(sel);
const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
/* stackOpa / mouthBlend / eyeBlend приходят из blend.js */

/* ---------- загрузка ассетов ---------- */

const MOUTH_LABELS = ['closed', 'slight', 'half', 'A_open', 'smile', 'smirk'];
const EYE_LABELS = ['neutral', 'half', 'blink', 'happy', 'squint'];

function loadImage(src) {
  return new Promise((res, rej) => {
    const im = new Image();
    im.onload = () => res(im);
    im.onerror = () => rej(new Error('не загрузился ' + src));
    im.src = src;
  });
}

let assets = null;

async function loadAssets() {
  const manifest = await (await fetch('pack/manifest.json')).json();
  const boxes = manifest.interactive.boxes;
  const mouthTiles = await Promise.all(
    manifest.interactive.mouth_tiles.map(loadImage));
  const eyesL = await Promise.all(manifest.interactive.eye_tiles.L.map(loadImage));
  const eyesR = await Promise.all(manifest.interactive.eye_tiles.R.map(loadImage));
  const head = await loadImage(manifest.interactive.head);
  return {
    head,
    mouth: mouthTiles.map((img, i) => ({ img, box: boxes.mouth_box })),
    eyes: [eyesL.map((img) => ({ img, box: boxes.eye_boxes[0] })),
           eyesR.map((img) => ({ img, box: boxes.eye_boxes[1] }))],
    headRect: boxes.head_rect,
    manifest,
  };
}

/* ---------- рендер (тот же альфа-стек, что в .moc3) ---------- */

function renderFace(ctx, st) {
  const A = assets;
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.drawImage(A.head, 0, 0);
  const drawStack = (tiles, weights) => {
    tiles.forEach((t, i) => {
      const opa = stackOpa(weights, i);
      if (opa <= 0.001) return;
      ctx.globalAlpha = opa;
      ctx.drawImage(t.img, t.box[0] - A.headRect[0], t.box[1] - A.headRect[1]);
    });
    ctx.globalAlpha = 1;
  };
  if (st.eyeOpen !== undefined) {
    drawStack(A.eyes[0], eyeBlend(st.eyeOpen, st.smile));
    drawStack(A.eyes[1], eyeBlend(st.eyeOpen2 ?? st.eyeOpen, st.smile));
  }
  if (st.mouthOpen !== undefined) {
    drawStack(A.mouth, mouthBlend(st.mouthOpen, st.form));
  }
}

/* ---------- вкладки ---------- */

document.querySelectorAll('.nav-item').forEach((btn) => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.nav-item').forEach((b) => b.classList.remove('active'));
    btn.classList.add('active');
    document.querySelectorAll('.tab-page').forEach((p) => (p.hidden = true));
    $('#tab-' + btn.dataset.tab).hidden = false;
  });
});

/* ---------- ползунки рта ---------- */

const mouthState = { open: 0, form: 0 };

function renderMouth() {
  const ctx = $('#mouth-canvas').getContext('2d');
  renderFace(ctx, {
    mouthOpen: mouthState.open,
    form: mouthState.form,
    eyeOpen: 1,
    smile: Math.max(0, mouthState.form), // физика модели: улыбка рта -> глаза
  });
  $('#out-open').textContent = mouthState.open.toFixed(2);
  $('#out-form').textContent = mouthState.form.toFixed(2);
  // весовые бары
  const w = mouthBlend(mouthState.open, mouthState.form);
  const bars = $('#mouth-weights').children;
  for (let i = 0; i < bars.length; i++) bars[i].style.height = Math.max(2, w[i] * 34) + 'px';
}

$('#sl-open').addEventListener('input', (e) => { mouthState.open = +e.target.value; renderMouth(); });
$('#sl-form').addEventListener('input', (e) => { mouthState.form = +e.target.value; renderMouth(); });
$('#mouth-presets').addEventListener('click', (e) => {
  const b = e.target.closest('button');
  if (!b) return;
  mouthState.open = +b.dataset.open;
  mouthState.form = +b.dataset.form;
  $('#sl-open').value = mouthState.open;
  $('#sl-form').value = mouthState.form;
  renderMouth();
});

/* ---------- ползунки глаз + авточ blink ---------- */

const eyeState = { openL: 1, openR: 1, smile: 0, blink: null };

function renderEyes() {
  const ctx = $('#eye-canvas').getContext('2d');
  renderFace(ctx, {
    eyeOpen: eyeState.openL,
    eyeOpen2: eyeState.openR,
    smile: eyeState.smile,
  });
  $('#out-el').textContent = eyeState.openL.toFixed(2);
  $('#out-er').textContent = eyeState.openR.toFixed(2);
  $('#out-sm').textContent = eyeState.smile.toFixed(2);
}

$('#sl-el').addEventListener('input', (e) => { eyeState.openL = +e.target.value; renderEyes(); });
$('#sl-er').addEventListener('input', (e) => { eyeState.openR = +e.target.value; renderEyes(); });
$('#sl-sm').addEventListener('input', (e) => { eyeState.smile = +e.target.value; renderEyes(); });
$('#btn-smile').addEventListener('click', () => {
  eyeState.smile = 1; $('#sl-sm').value = 1; renderEyes();
});
$('#btn-reset-eyes').addEventListener('click', () => {
  eyeState.openL = eyeState.openR = 1; eyeState.smile = 0;
  $('#sl-el').value = $('#sl-er').value = $('#sl-sm').value = 0; // 0 для smile
  $('#sl-el').value = $('#sl-er').value = 1;
  renderEyes();
});

let blinkEnabled = false;
let nextBlinkAt = 0;
$('#btn-blink').addEventListener('click', (e) => {
  blinkEnabled = !blinkEnabled;
  e.currentTarget.classList.toggle('on', blinkEnabled);
  e.currentTarget.textContent = blinkEnabled ? '■ Стоп моргание' : '▲ Авто-моргание';
  if (blinkEnabled) nextBlinkAt = performance.now() + 400;
});

function blinkLoop(now) {
  if (blinkEnabled) {
    if (eyeState.blink === null && now >= nextBlinkAt) eyeState.blink = { t0: now };
    if (eyeState.blink) {
      const D = 300; // мс на цикл 1->0->1, как моргание в VTS
      const t = (now - eyeState.blink.t0) / D;
      if (t >= 1) {
        eyeState.blink = null;
        nextBlinkAt = now + 1800 + Math.random() * 2600;
        eyeState.openL = eyeState.openR = 1;
      } else {
        const v = 1 - Math.sin(Math.PI * t); // гладкий вниз-вверх
        eyeState.openL = eyeState.openR = clamp(v, 0, 1);
      }
      $('#sl-el').value = eyeState.openL;
      $('#sl-er').value = eyeState.openR;
      renderEyes();
    }
  }
  requestAnimationFrame(blinkLoop);
}

/* ---------- миниатюры кадров ---------- */

function fillThumbs(container, files, labels, ranges) {
  files.forEach((f, i) => {
    const fig = document.createElement('figure');
    const img = new Image();
    img.src = f; img.loading = 'lazy';
    fig.appendChild(img);
    const cap = document.createElement('figcaption');
    cap.innerHTML = '<b>' + labels[i] + '</b><br>' + ranges[i];
    fig.appendChild(cap);
    container.appendChild(fig);
  });
}

/* ---------- шкалы времени (вкладка «Данные») ---------- */

function fillTimeline(el, timeline, title) {
  const t = document.createElement('div');
  t.className = 'tl-title';
  t.textContent = title;
  el.appendChild(t);
  const bar = document.createElement('div');
  bar.className = 'tl-bar';
  timeline.forEach((seg) => {
    const s = document.createElement('div');
    s.className = 'tl-seg ' + (seg.kind === 'hold' ? 'hold' : 'trans');
    s.style.width = ((seg.to - seg.from) * 100).toFixed(2) + '%';
    s.textContent = seg.frame_b ? seg.frame_a + '→' + seg.frame_b : seg.frame_a;
    s.title = seg.from.toFixed(2) + ' – ' + seg.to.toFixed(2);
    bar.appendChild(s);
  });
  el.appendChild(bar);
  const scale = document.createElement('div');
  scale.className = 'tl-scale';
  const marks = [0, 0.2, 0.4, 0.6, 0.8, 1];
  scale.innerHTML = marks.map((m) => '<span>' + m.toFixed(1) + '</span>').join('');
  el.appendChild(scale);
}

/* ---------- CSV-таблица ---------- */

async function fillCsvTable() {
  const csv = await (await fetch('pack/storyboard.csv')).text();
  const rows = csv.trim().split(/\r?\n/).map((r) => r.split(','));
  const table = $('#csv-table');
  rows.forEach((cells, ri) => {
    const tr = document.createElement('tr');
    cells.forEach((c, ci) => {
      const cell = document.createElement(ri === 0 ? 'th' : 'td');
      if (ri > 0 && ci === 2) {
        const im = new Image();
        im.src = 'pack/' + c;
        im.loading = 'lazy';
        cell.appendChild(im);
        cell.appendChild(document.createTextNode(' ' + c.split('/').pop()));
      } else {
        cell.textContent = c;
      }
      tr.appendChild(cell);
    });
    table.appendChild(tr);
  });
}

/* ---------- старт ---------- */

(async function init() {
  try {
    assets = await loadAssets();
  } catch (err) {
    const box = $('#error');
    box.hidden = false;
    box.textContent = 'Не удалось загрузить пакет раскадровки: ' + err.message +
      ' — соберите его: python3 tools/build_face_storyboard.py';
    return;
  }
  // весовые бары рта
  const ww = $('#mouth-weights');
  MOUTH_LABELS.forEach((l) => {
    const b = document.createElement('div');
    b.className = 'wbar';
    b.style.height = '2px';
    b.innerHTML = '<span>' + l + '</span>';
    ww.appendChild(b);
  });
  // миниатюры
  const mf = assets.manifest.mouth_frames;
  fillThumbs($('#mouth-thumbs'),
    mf.map((f) => 'pack/' + f.file),
    mf.map((f) => f.index + ' · ' + f.label),
    mf.map((f) => f.range));
  const ef = assets.manifest.eye_frames;
  fillThumbs($('#eye-thumbs'),
    ef.map((f) => 'pack/' + f.file),
    ef.map((f) => f.index + ' · ' + f.label),
    ef.map((f) => f.range));
  // шкалы
  fillTimeline($('#timeline-mouth'), assets.manifest.mouth_timeline,
    'Цепочка открытия рта: зелёный = hold (один кадр), жёлтый = короткий crossfade (макс. 2 кадра)');
  fillTimeline($('#timeline-eyes'), assets.manifest.eye_timeline,
    'Цепочка открытия глаз: моргание = быстрый проход 1 → 0 → 1');
  fillCsvTable();
  renderMouth();
  renderEyes();
  requestAnimationFrame(blinkLoop);
})();

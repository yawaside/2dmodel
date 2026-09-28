/* Smoke-тест страницы предпросмотра (vts/) без браузера.
 *
 *   node tests/vts_smoke.cjs
 *
 * Грубая эмуляция DOM и WebGL: проверяет, что vts/app.js поднимается,
 * догружает пакет, строит меши и рисует кадры — и что все обработчики UI
 * (переключатели, экспрессии, жесты, фон, экспорт кадра) не падают.
 * Рендер при этом, разумеется, не проверяется: для этого есть
 * tests/browser_turnaround.cjs (playwright) и глаза.
 *
 * Зависимостей нет: только node. Файлы читаются с диска, сеть не нужна.
 */
'use strict';

const fs = require('node:fs');
const path = require('node:path');

const ROOT = path.resolve(__dirname, '..');
const PAGE = path.join(ROOT, 'vts');
const calls = { drawElements: 0, drawArrays: 0, texImage2D: 0, bufferSubData: 0, drawImage: 0 };

/* ------------------------------------------------------------------ */
/* заглушки DOM                                                        */
/* ------------------------------------------------------------------ */

function stubEl(tag = 'div') {
  return {
    tagName: String(tag).toUpperCase(),
    style: {}, dataset: {}, handlers: {}, children: [],
    classList: { toggle: () => {}, add: () => {}, remove: () => {}, contains: () => false },
    hidden: false, value: '0', width: 0, height: 0,
    clientWidth: 900, clientHeight: 600, attrs: {},
    setAttribute(k, v) { this.attrs[k] = v; },
    getAttribute(k) { return this.attrs[k]; },
    addEventListener(type, fn) { (this.handlers[type] ||= []).push(fn); },
    removeEventListener() {},
    dispatch(type, ev = {}) {
      (this.handlers[type] || []).forEach((fn) => fn(Object.assign({
        target: this, currentTarget: this, clientX: 120, clientY: 80,
        pointerId: 1, preventDefault() {},
      }, ev)));
    },
    appendChild(c) { this.children.push(c); return c; },
    remove() {}, click() {},
    querySelector: () => stubEl('input'),
    querySelectorAll: () => [],
    getBoundingClientRect: () => ({ left: 0, top: 0, width: 900, height: 600 }),
    scrollIntoView() {}, setPointerCapture() {}, releasePointerCapture() {},
    getContext(kind) { return kind === '2d' ? ctx2d() : fakeGL(); },
    toBlob(cb) { cb({ size: 1 }); },
  };
}

function ctx2d() {
  return {
    globalAlpha: 1,
    clearRect() {},
    drawImage() { calls.drawImage++; },
    getImageData(x, y, w, h) {
      // «арт» — прямоугольник в центре: покрытие сетки получается непустым
      const data = new Uint8ClampedArray(w * h * 4);
      for (let j = 0; j < h; j++) {
        for (let i = 0; i < w; i++) {
          const inside = i > w * 0.15 && i < w * 0.85 && j > h * 0.1 && j < h * 0.9;
          data[(j * w + i) * 4 + 3] = inside ? 255 : 0;
        }
      }
      return { data, width: w, height: h };
    },
  };
}

const GL_ENUM = {};
let enumSeq = 0;
function fakeGL() {
  const noop = () => {};
  const base = {
    createShader: () => ({}), getShaderParameter: () => true, getShaderInfoLog: () => '',
    createProgram: () => ({}), getProgramParameter: () => true, getProgramInfoLog: () => '',
    getAttribLocation: () => (enumSeq++ % 3), getUniformLocation: () => ({}),
    createBuffer: () => ({}), createTexture: () => ({}),
    bufferSubData: () => { calls.bufferSubData++; },
    texImage2D: () => { calls.texImage2D++; },
    drawElements: () => { calls.drawElements++; },
    drawArrays: () => { calls.drawArrays++; },
  };
  return new Proxy(base, {
    get(target, prop) {
      if (prop in target) return target[prop];
      if (typeof prop === 'string' && /^[A-Z0-9_]+$/.test(prop)) {
        return (GL_ENUM[prop] ??= ++enumSeq);
      }
      return noop;
    },
  });
}

const elements = new Map();
const el = (sel) => {
  if (!elements.has(sel)) elements.set(sel, stubEl(sel.includes('canvas') ? 'canvas' : 'div'));
  return elements.get(sel);
};

const LISTS = {
  '#expr-row button': ['neutral', 'smile', 'smirk', 'talk', 'surprise', 'sleepy'],
  '#gest-row button[data-gesture]': ['blink', 'nod', 'shake'],
  '.bg-chips button': ['transparent', 'dark', 'green', 'stream'],
};
const lists = new Map();
function list(sel) {
  if (!lists.has(sel)) {
    const keys = LISTS[sel] || [undefined, undefined, undefined];
    lists.set(sel, keys.map((key) => {
      const e = stubEl('button');
      if (sel.includes('expr-row')) e.dataset.expr = key;
      if (sel.includes('gest-row')) e.dataset.gesture = key;
      if (sel.includes('bg-chips')) e.dataset.bg = key;
      if (sel.includes('data-jump')) e.dataset.jump = 'stage-sec';
      return e;
    }));
  }
  return lists.get(sel);
}

const keyHandlers = [];
global.window = {
  devicePixelRatio: 1,
  addEventListener: (type, fn) => { if (type === 'keydown') keyHandlers.push(fn); },
  AudioContext: function () {
    return {
      createAnalyser: () => ({
        fftSize: 1024, smoothingTimeConstant: 0,
        getByteTimeDomainData: () => {},
      }),
      createMediaStreamSource: () => ({ connect: () => {} }),
      close: () => Promise.resolve(),
    };
  },
};
global.document = {
  querySelector: (s) => el(s),
  querySelectorAll: list,
  createElement: (t) => stubEl(t),
  getElementById: (id) => el('#' + id),
  activeElement: null,
  body: stubEl('body'),
};
global.Image = class {
  set src(v) {
    this._src = v;
    const p = path.join(PAGE, v);
    setTimeout(() => {
      if (fs.existsSync(p)) { this.width = 725; this.height = 640; this.onload && this.onload(); }
      else this.onerror && this.onerror(new Error('нет файла ' + p));
    }, 1);
  }
  get src() { return this._src; }
};
global.fetch = async (url) => {
  const p = path.join(PAGE, url);
  if (!fs.existsSync(p)) throw new Error('нет файла ' + url);
  return { json: async () => JSON.parse(fs.readFileSync(p, 'utf8')) };
};
let rafCount = 0;
global.requestAnimationFrame = (fn) => {
  if (rafCount++ < 4) setTimeout(() => fn(rafCount * 16), 1);
  return rafCount;
};
// node >= 21 держит navigator доступным только для чтения — подменяем через defineProperty
Object.defineProperty(global, 'navigator', {
  value: { clipboard: { writeText: async () => {} } },
  configurable: true, writable: true,
});
global.URL = { createObjectURL: () => 'blob:stub', revokeObjectURL: () => {} };

/* ------------------------------------------------------------------ */
/* прогон                                                              */
/* ------------------------------------------------------------------ */

el('#error').hidden = true;
global.Chibi = require(path.join(PAGE, 'model.js'));
require(path.join(PAGE, 'app.js'));

const problems = [];
function check(cond, msg) { if (!cond) problems.push(msg); }

setTimeout(() => {
  // весь UI: переключатели, экспрессии, жесты, фон, экспорт
  const ids = ['#t-track', '#t-idle', '#t-blink', '#t-physics', '#t-mic', '#t-talk',
    '#t-hud', '#t-wire', '#btn-reset', '#btn-shot', '#btn-copy',
    '#v-size', '#v-x', '#v-y', '#v-rot',
    '#opt-smooth', '#opt-amp', '#opt-blink', '#opt-gain', '#opt-gate'];
  for (const id of ids) {
    const e = el(id);
    e.value = id === '#v-size' ? '150' : id === '#v-rot' ? '20' : '1';
    e.dispatch('click');
    e.dispatch('input', { target: e });
  }
  for (const sel of [...Object.keys(LISTS), '[data-jump]']) {
    list(sel).forEach((e) => e.dispatch('click'));
  }
  const stage = el('#stage');
  stage.dispatch('pointerdown');
  stage.dispatch('pointermove');
  stage.dispatch('pointerup');
  stage.dispatch('pointerleave');
  keyHandlers.forEach((fn) => fn({ key: '3', target: { tagName: 'BODY' } }));

  setTimeout(() => {
    check(rafCount >= 2, 'цикл кадров не запустился');
    check(calls.drawElements >= 2, 'меши не рисуются');
    check(calls.drawImage >= 3, 'лицо не композитится');
    check(calls.texImage2D >= 2, 'текстуры не загружаются');
    check(el('#error').hidden, 'страница показала ошибку: ' + el('#error').textContent);

    if (problems.length) {
      console.error('FAIL');
      problems.forEach((p) => console.error(' - ' + p));
      process.exit(1);
    }
    console.log('ok: кадров %d, drawElements %d, drawImage %d, текстур %d',
      rafCount, calls.drawElements, calls.drawImage, calls.texImage2D);
    console.log('SMOKE OK — vts/app.js поднимается и рисует');
    process.exit(0);
  }, 250);
}, 400);

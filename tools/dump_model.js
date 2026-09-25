#!/usr/bin/env node
/* Dump a moc3 model state to JSON so it can be rasterised offline.
   usage: node dump_model.js <model.moc3> [ '{"ParamAngleX":30}' ] */
const fs = require('fs'), path = require('path');
const COREDIR = process.env.CUBISM_CORE_DIR ||
                path.join(__dirname, '..', 'preview', 'lib');
globalThis.__dirname = COREDIR;
globalThis.__filename = path.join(COREDIR, 'live2dcubismcore.min.js');
globalThis.require = require;
(0, eval)(fs.readFileSync(path.join(COREDIR, 'live2dcubismcore.min.js'), 'utf8'));
const Core = globalThis.Live2DCubismCore;

const file = process.argv[2];
const states = process.argv[3] ? JSON.parse(process.argv[3]) : [{}];
const buf = fs.readFileSync(file);
const moc = Core.Moc.fromArrayBuffer(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
if (!moc) { console.error('MOC LOAD FAILED'); process.exit(2); }
const model = Core.Model.fromMoc(moc);
if (!model) { console.error('MODEL INIT FAILED'); process.exit(3); }
const d = model.drawables;

const out = [];
for (const st of states) {
  for (let i = 0; i < model.parameters.count; i++)
    model.parameters.values[i] = model.parameters.defaultValues[i];
  for (const k in st) {
    const i = model.parameters.ids.indexOf(k);
    if (i < 0) throw new Error('unknown parameter ' + k);
    model.parameters.values[i] = st[k];
  }
  model.update();
  const ds = [];
  for (let i = 0; i < d.count; i++) {
    ds.push({
      id: d.ids[i],
      ro: d.renderOrders[i],
      op: d.opacities[i],
      tex: d.textureIndices[i],
      visible: !!(d.dynamicFlags[i] & 1),
      pos: Array.from(d.vertexPositions[i]),
      uv: Array.from(d.vertexUvs[i]),
      idx: Array.from(d.indices[i]),
    });
  }
  out.push({ state: st, canvas: {
    w: model.canvasinfo.CanvasWidth, h: model.canvasinfo.CanvasHeight,
    ox: model.canvasinfo.CanvasOriginX, oy: model.canvasinfo.CanvasOriginY,
    ppu: model.canvasinfo.PixelsPerUnit }, drawables: ds });
}
console.log('===JSON===');
console.log(JSON.stringify(out));

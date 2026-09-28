'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const icons = {
    cube: '<path d="m12 3 9 5v8l-9 5-9-5V8zM3 8l9 5 9-5M12 13v8M7 5.8l9 5"/>',
    grid: '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 10h18M10 3v18M3 16h18M16 3v18"/>',
    layers: '<path d="m12 3 10 5-10 5L2 8zM2 12l10 5 10-5M2 16l10 5 10-5"/>',
    book: '<path d="M12 5v15M3 4c4-1 6 0 9 1 3-1 5-2 9-1v15c-4-1-6 0-9 1-3-1-5-2-9-1z"/>',
    download: '<path d="M12 3v12m-5-5 5 5 5-5M4 15v5h16v-5"/>',
    contrast: '<circle cx="12" cy="12" r="9"/><path d="M12 3v18"/><path d="M12 3a9 9 0 0 1 0 18z" fill="currentColor" stroke="none"/>',
    prev: '<path d="M5 5v14m14-14L8 12l11 7z"/>',
    next: '<path d="M19 5v14M5 5l11 7-11 7z"/>',
    play: '<path d="m7 4 13 8-13 8z" fill="currentColor" stroke="none"/>',
    pause: '<path d="M7 4v16M17 4v16" stroke-width="4"/>',
    info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v6m0-10v.2"/>',
  };
  const svg = name => `<svg viewBox="0 0 24 24" aria-hidden="true">${icons[name] || ''}</svg>`;
  document.querySelectorAll('[data-icon]').forEach(el => { el.innerHTML = svg(el.dataset.icon); });
  const titles = {
    views: ['Персонаж со всех сторон', 'Единый набор ракурсов, чтобы следующий шаг в Live2D стал проще.'],
    storyboard: ['Каждый ракурс — на своём месте', 'Контактные листы и опорные кадры для художника и риггера.'],
    prepare: ['Подготовка к Live2D', 'Отдельные части, чистые перекрытия и понятный план риггинга.'],
  };
  let activeTab = 'views';
  function setTab(tab, updateHash = true) {
    if (!titles[tab]) tab = 'views';
    activeTab = tab;
    stop();
    document.querySelectorAll('[data-tab]').forEach(button => {
      const active = button.dataset.tab === tab;
      button.classList.toggle('active', active);
      if (active) button.setAttribute('aria-current', 'page');
      else button.removeAttribute('aria-current');
    });
    document.querySelectorAll('.tab-section').forEach(section => { section.hidden = section.id !== tab; });
    $('page-title').replaceChildren(document.createTextNode(titles[tab][0]));
    const dot = document.createElement('span'); dot.textContent = '.'; $('page-title').append(dot);
    $('page-subtitle').textContent = titles[tab][1];
    if (updateHash) history.replaceState(null, '', '#' + tab);
  }
  document.querySelectorAll('[data-tab]').forEach(button => button.addEventListener('click', () => setTab(button.dataset.tab)));
  window.addEventListener('hashchange', () => setTab(location.hash.slice(1), false));

  let manifest, mode = 'body', index = 0, timer = null, loaded = false;
  const positions = {body: 0, head: 4};
  const imageCache = new Map();
  const frames = () => mode === 'body' ? manifest.frames : manifest.head_frames;
  const source = frame => 'pack/' + frame.file;
  function error(message) { $('error').textContent = message; $('error').hidden = false; }
  function stop() {
    if (timer !== null) clearInterval(timer);
    timer = null;
    $('play').innerHTML = svg('play');
    $('play').setAttribute('aria-label', 'Воспроизвести');
    $('play').setAttribute('aria-pressed', 'false');
  }
  function play() {
    if (!loaded) return;
    if (timer !== null) { stop(); return; }
    $('play').innerHTML = svg('pause');
    $('play').setAttribute('aria-label', 'Пауза');
    $('play').setAttribute('aria-pressed', 'true');
    timer = setInterval(() => show(index + 1), Number($('speed').value));
  }
  function show(next, manual = false) {
    if (!loaded) return;
    if (manual) stop();
    const list = frames();
    index = ((next % list.length) + list.length) % list.length;
    positions[mode] = index;
    const frame = list[index], previous = list[(index - 1 + list.length) % list.length];
    $('main-image').src = source(frame);
    $('main-image').alt = frame.label + (frame.mirrored ? ' — зеркальный черновик' : ' — AI-референс');
    $('main-image').hidden = false;
    $('onion-image').src = source(previous);
    $('onion-image').hidden = !$('onion-toggle').checked;
    $('detail-name').textContent = mode === 'body' ? frame.label : 'Поворот головы';
    $('stage-label').textContent = frame.label;
    $('detail-angle').classList.toggle('head-value', mode === 'head');
    if (mode === 'body') {
      $('detail-angle').textContent = frame.angle;
      const degree = document.createElement('span'); degree.textContent = '°'; $('detail-angle').append(degree);
    } else $('detail-angle').textContent = frame.label;
    const status = frame.mirrored ? 'Зеркальный черновик' : (mode === 'head' ? 'Целевой ключ' : 'AI-референс');
    $('status-badge').textContent = status;
    $('status-badge').classList.toggle('draft', !!frame.mirrored);
    $('stage-note').textContent = status;
    $('format-badge').textContent = mode === 'body' ? 'PNG · RGBA' : 'PNG · RGB';
    $('detail-canvas').textContent = mode === 'body' ? '768 × 1024 px' : '512 × 512 px';
    $('canvas-dim').textContent = mode === 'body' ? '768 × 1024' : '512 × 512';
    $('detail-pivot-label').textContent = mode === 'body' ? 'Опорная точка' : 'Углы';
    $('detail-pivot').textContent = mode === 'body' ? frame.pivot.join(', ') : 'Приблизительные';
    const number = String(index + 1).padStart(2, '0');
    const total = String(list.length).padStart(2, '0');
    $('detail-index').textContent = `${number} / ${total}`;
    $('frame-count').textContent = number + ' ';
    const count = document.createElement('span'); count.textContent = '/ ' + total; $('frame-count').append(count);
    $('frame-range').value = index;
    $('frame-range').setAttribute('aria-valuetext', `${index + 1}: ${frame.label}`);
    $('download-frame').href = source(frame);
    $('download-frame').download = frame.file.split('/').pop();
    $('detail-note').textContent = mode === 'head'
      ? 'X/Y — целевые ключи, не точные замеры. Скорректируйте перспективу век, козырька и микрофона перед риггингом.'
      : (frame.mirrored ? 'Отражённый черновик. Исправьте сторону микрофона, узел платка и подсумки перед нарезкой.'
        : 'Спина и нижняя часть — предложенный дизайн. Согласуйте пропорции и экипировку всех ракурсов перед нарезкой.');
    $('frame-strip').querySelectorAll('button').forEach((button, i) => {
      button.classList.toggle('selected', i === index);
      button.setAttribute('aria-pressed', String(i === index));
    });
  }
  function buildStrip() {
    const box = $('frame-strip'); box.replaceChildren();
    box.classList.toggle('head-strip', mode === 'head');
    $('image-plane').classList.toggle('head', mode === 'head');
    $('frame-range').max = frames().length - 1;
    $('strip-title').textContent = mode === 'body' ? 'Ключевые ракурсы ' : 'Повороты головы ';
    const badge = document.createElement('span'); badge.textContent = mode === 'body' ? '8 кадров' : '9 кадров'; $('strip-title').append(badge);
    $('strip-hint').textContent = mode === 'body' ? 'Полный оборот · шаг 45°' : 'Целевые ключи · X / Y';
    frames().forEach((frame, i) => {
      const button = document.createElement('button'); button.className = 'frame-card';
      button.setAttribute('aria-label', `${i + 1}. ${frame.label}${frame.mirrored ? ', зеркальный черновик' : ''}`);
      button.title = button.getAttribute('aria-label');
      const num = document.createElement('span'); num.className = 'frame-number'; num.textContent = String(i + 1).padStart(2, '0');
      const tick = document.createElement('span'); tick.className = 'frame-selected'; tick.textContent = '●'; tick.setAttribute('aria-hidden', 'true');
      const image = document.createElement('img'); image.src = source(frame); image.alt = '';
      const angle = document.createElement('div'); angle.className = 'frame-angle'; angle.textContent = mode === 'body' ? frame.angle + '°' : frame.label;
      const label = document.createElement('div'); label.className = 'frame-name'; label.textContent = mode === 'body' ? frame.label.split(' · ')[0] : 'Целевой ключ';
      const mirror = document.createElement('div'); mirror.className = 'mirror-mark'; mirror.textContent = frame.mirrored ? 'Зерк. черновик' : '';
      button.append(num, tick, image, angle, label, mirror);
      button.addEventListener('click', () => show(i, true));
      box.append(button);
    });
  }
  document.querySelectorAll('[data-mode]').forEach(button => button.addEventListener('click', () => {
    if (!loaded) return;
    stop(); mode = button.dataset.mode;
    document.querySelectorAll('[data-mode]').forEach(b => { b.classList.toggle('selected', b === button); b.setAttribute('aria-pressed', String(b === button)); });
    buildStrip(); show(positions[mode]);
  }));
  $('prev').addEventListener('click', () => show(index - 1, true));
  $('next').addEventListener('click', () => show(index + 1, true));
  $('play').addEventListener('click', play);
  $('frame-range').addEventListener('input', e => show(Number(e.target.value), true));
  $('speed').addEventListener('change', () => { if (timer !== null) { stop(); play(); } });
  $('onion-toggle').addEventListener('change', () => { if (loaded) show(index); });
  $('grid-toggle').addEventListener('click', () => {
    const active = $('stage').classList.toggle('show-grid');
    $('grid-toggle').setAttribute('aria-pressed', String(active));
  });
  const backgrounds = ['green', 'white', 'dark']; let background = 0;
  $('bg-toggle').addEventListener('click', () => { background = (background + 1) % backgrounds.length; $('stage').dataset.bg = backgrounds[background]; });
  document.addEventListener('visibilitychange', () => { if (document.hidden) stop(); });
  document.addEventListener('keydown', e => {
    if (!loaded || activeTab !== 'views' || e.altKey || e.ctrlKey || e.metaKey || e.target.closest('input, select, textarea, button, a, [contenteditable]')) return;
    if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') { e.preventDefault(); show(index + (e.key === 'ArrowRight' ? 1 : -1), true); }
    if (e.code === 'Space') { e.preventDefault(); play(); }
  });
  function preload(frame) {
    return new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => { imageCache.set(frame.file, image); resolve(); };
      image.onerror = () => reject(new Error(`Не удалось загрузить ${frame.file}`));
      image.src = source(frame);
    });
  }
  async function init() {
    const controls = document.querySelectorAll('[data-mode], #prev, #play, #next, #frame-range');
    controls.forEach(control => { control.disabled = true; });
    try {
      const response = await fetch('pack/manifest.json');
      if (!response.ok) throw new Error(`manifest.json: HTTP ${response.status}`);
      manifest = await response.json();
      if (manifest.schema_version !== 1 || manifest.frames?.length !== 8 || manifest.head_frames?.length !== 9) throw new Error('Неизвестный формат пакета. Пересоберите его.');
      await Promise.all([...manifest.frames, ...manifest.head_frames].map(preload));
      loaded = true; buildStrip(); show(0);
      controls.forEach(control => { control.disabled = false; });
    } catch (e) {
      $('stage-label').textContent = 'Референсы недоступны';
      error(`Не удалось открыть пакет: ${e.message}. Запустите python3 tools/build_turnaround.py --psd, затем откройте страницу через tools/serve.py (не file://).`);
    }
  }

  // A local planning aid, not a claim that the rigging parts already exist.
  const groups = [
    ['design', '01 / Утверждение дизайна', ['Согласовать пропорции всех ракурсов', 'Проверить микрофон и наушники', 'Исправить зеркальные черновики', 'Утвердить спину и нижнюю часть']],
    ['head', '02 / Голова и лицо', ['Основа лица, дорисовка перекрытий', 'Уши и передние / задние волосы', 'Белки, радужки и блики отдельно', 'Веки, брови и формы моргания', 'Полость рта, губы, зубы и язык']],
    ['kit', '03 / Головной убор и детали', ['Тулья кепки и козырёк', 'Звезда на отдельном слое', 'Левый / правый наушник и дужка', 'Штанга и наконечник микрофона']],
    ['body', '04 / Корпус и одежда', ['Основа шеи и туловища', 'Передняя / задняя часть жилета', 'Ремни и отдельные подсумки', 'Платок, узел и два хвоста']],
    ['limbs', '05 / Руки и ноги', ['Плечо, предплечье, кисть: слева', 'Плечо, предплечье, кисть: справа', 'Левая нога и ботинок', 'Правая нога и ботинок']],
    ['rig', '06 / Проверка в Cubism', ['RGB, 8-bit, прозрачный PSD', 'Меши и иерархия деформеров', 'Ключи X/Y и четыре диагонали', 'Моргание, лип-синк и физика', 'Проверка крайних углов и стыков']],
  ];
  const storageKey = 'chibi-turnaround-checklist-v1';
  let checked = {};
  try { const saved = JSON.parse(localStorage.getItem(storageKey) || '{}'); if (saved && typeof saved === 'object' && !Array.isArray(saved)) checked = saved; }
  catch { $('storage-status').textContent = 'Локальное сохранение недоступно — используйте экспорт'; }
  function updateProgress() {
    const boxes = [...$('checklist').querySelectorAll('input')];
    const count = boxes.filter(box => box.checked).length;
    $('progress-number').textContent = Math.round(count / boxes.length * 100) + '%';
  }
  groups.forEach(([id, title, items]) => {
    const section = document.createElement('section'); section.className = 'checklist-group';
    const heading = document.createElement('h3'); heading.textContent = title; section.append(heading);
    items.forEach((text, i) => {
      const key = `${id}_${i}`;
      const label = document.createElement('label');
      const input = document.createElement('input'); input.type = 'checkbox'; input.checked = checked[key] === true;
      input.dataset.key = key; input.dataset.label = text;
      input.addEventListener('change', () => {
        checked[key] = input.checked;
        try { localStorage.setItem(storageKey, JSON.stringify(checked)); }
        catch { $('storage-status').textContent = 'Не удалось сохранить в браузере — используйте экспорт'; }
        updateProgress();
      });
      label.append(input, document.createTextNode(text)); section.append(label);
    });
    $('checklist').append(section);
  });
  $('export-checklist').addEventListener('click', () => {
    const items = [...$('checklist').querySelectorAll('input')].map(input => ({id: input.dataset.key, label: input.dataset.label, done: input.checked}));
    const blob = new Blob([JSON.stringify({version: 1, purpose: 'manual_preparation_checklist_not_generated_layers', items}, null, 2)], {type: 'application/json'});
    const link = document.createElement('a'); const url = URL.createObjectURL(blob);
    link.href = url; link.download = 'ChibiVT_preparation_checklist.json'; document.body.append(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  updateProgress();
  setTab(location.hash.slice(1), false);
  init();
})();

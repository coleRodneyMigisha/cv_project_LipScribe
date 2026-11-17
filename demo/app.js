/*
 * LipScribe demo page (F-06) — owner: Wakisa Isabelle, reviewer: Cole Rodney Migisha
 *
 * Talks to the backend over one WebSocket. Message format: demo/README.md.
 *   - No ?backend=... in the address  -> the pretend backend in mock-backend.js
 *   - ?backend=wss://host/ws/live      -> the real backend on the Alienware
 * You can also set DEFAULT_BACKEND below once Cole's address is fixed.
 */
(function () {
  'use strict';

  var DEFAULT_BACKEND = '';   // e.g. 'wss://lipscribe-api.example.com/ws/live'
  var CHUNK_MS = 1000;        // how often a piece of audio+video is sent
  var CLIP_SECONDS = 3;       // Clip mode stops by itself after this
  var FPS = 25;               // gate readings per second
  var WINDOW_FRAMES = 75;     // gate plot shows 3 s
  var MAX_CROPS = 8;
  var MAX_WORDS = 24;         // words kept on screen per caption in Live mode
  var CONNECT_TIMEOUT_MS = 8000;
  var REPLY_TIMEOUT_MS = 20000;

  var ERRORS = {
    no_face: ['We couldn’t see your face', 'Face the camera in good light, keep your mouth in view, and try again.'],
    no_audio: ['We couldn’t hear you', 'Check that the browser can use your microphone, then speak a little louder.'],
    clip_too_short: ['That clip was too short', 'Keep recording until the timer reaches 3 seconds and say a whole sentence.'],
    server_busy: ['The Alienware is busy', 'Another recording is being captioned. Wait a few seconds and try again.'],
    server_unreachable: ['We can’t reach the captioning server', 'The backend may be switched off. Check the address, or open the page without ?backend= to use sample captions.'],
    connection_lost: ['The connection dropped', 'The link to the captioning server was lost. Try again.'],
    no_reply: ['The server didn’t answer', 'Nothing came back in time. Try again in a moment.'],
    camera_denied: ['Camera or microphone blocked', 'Allow camera and microphone for this site in your browser settings, then reload the page.'],
    camera_missing: ['No camera or microphone found', 'Plug in or turn on a camera and microphone, then reload the page.'],
    camera_busy: ['Your camera is in use', 'Close other apps or tabs using the camera (Zoom, Teams, Meet), then try again.'],
    needs_https: ['This page needs a secure address', 'Browsers only share the camera on https:// pages or on localhost.'],
    unsupported: ['This browser can’t record here', 'Try the latest Chrome, Firefox, Edge or Safari.'],
    unknown: ['Something went wrong', 'Try again. If it keeps happening, tell the team what you did.']
  };

  // ---------- Elements ----------
  var $ = function (id) { return document.getElementById(id); };
  var el = {
    root: document.documentElement, left: $('leftCard'), video: $('video'), mouthBox: $('mouthBox'),
    camEmpty: $('camEmpty'), camEmptyText: $('camEmptyText'), camBtn: $('camBtn'),
    timerPill: $('timerPill'), recLabel: $('recLabel'), timer: $('timer'), facePill: $('facePill'),
    overlay: $('overlayCaption'), recBtn: $('recBtn'), recTitle: $('recTitle'), recHelp: $('recHelp'),
    noiseDb: $('noiseDb'), noiseWord: $('noiseWord'), gauge: $('gauge'), gaugeKnob: $('gaugeKnob'),
    badge: $('badge'), badgeText: $('badgeText'),
    errorBox: $('errorBox'), errorTitle: $('errorTitle'), errorText: $('errorText'), errorCode: $('errorCode'), errorRetry: $('errorRetry'),
    loading: $('loading'), steps: $('steps'), cards: $('cards'),
    earsText: $('earsText'), lipsText: $('lipsText'), earsModel: $('earsModel'), lipsModel: $('lipsModel'),
    gateSub: $('gateSub'), bars: $('bars'), soundPct: $('soundPct'), lipsPct: $('lipsPct'),
    ax: [$('ax0'), $('ax1'), $('ax2'), $('ax3')], crops: $('crops'),
    statTime: $('statTime'), statFace: $('statFace'), conn: $('conn'), connText: $('connText'),
    themeBtn: $('themeBtn')
  };

  // ---------- State ----------
  var params = new URLSearchParams(location.search);
  var backendUrl = toWs(params.get('backend') || DEFAULT_BACKEND);
  var useMock = !backendUrl || backendUrl === 'mock';
  var S = {
    run: 'live', mode: 'grid', stream: null, recorder: null, socket: null,
    phase: 'nocam',       // nocam | ready | recording | waiting
    startedAt: 0, raf: 0, gate: [], crops: [], connectTimer: 0, replyTimer: 0, gotFinal: false
  };

  // ---------- Helpers ----------
  function toWs(u) {
    if (!u) return '';
    if (u === 'mock') return 'mock';
    if (u.indexOf('https://') === 0) return 'wss://' + u.slice(8);
    if (u.indexOf('http://') === 0) return 'ws://' + u.slice(7);
    return u;
  }
  function clamp(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }
  function words(s) { return (s || '').trim().split(/\s+/).filter(Boolean); }
  function fmtTime(sec) {
    var m = Math.floor(sec / 60), s = sec - m * 60;
    return String(m).padStart(2, '0') + ':' + s.toFixed(1).padStart(4, '0');
  }
  function minus(n) { return String(n).replace('-', '−'); }
  function store(key, val) { try { if (val === undefined) return localStorage.getItem(key); localStorage.setItem(key, val); } catch (e) { return null; } }

  // Which ears-only words differ from ears+lips (longest common subsequence).
  function diffIndexes(a, b) {
    var n = a.length, m = b.length, i, j, dp = [];
    for (i = 0; i <= n; i++) { dp.push(new Array(m + 1).fill(0)); }
    for (i = n - 1; i >= 0; i--) for (j = m - 1; j >= 0; j--)
      dp[i][j] = a[i] === b[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
    var keep = new Set(); i = 0; j = 0;
    while (i < n && j < m) {
      if (a[i] === b[j]) { keep.add(i); i++; j++; }
      else if (dp[i + 1][j] >= dp[i][j + 1]) i++; else j++;
    }
    var out = new Set();
    for (i = 0; i < n; i++) if (!keep.has(i)) out.add(i);
    return out;
  }

  // ---------- Theme ----------
  function setTheme(t) {
    el.root.setAttribute('data-theme', t);
    el.themeBtn.setAttribute('aria-label', t === 'dark' ? 'Switch to light theme' : 'Switch to dark theme');
    el.themeBtn.innerHTML = t === 'dark'
      ? '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="4"></circle><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"></path></svg>'
      : '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"></path></svg>';
  }
  var savedTheme = params.get('theme') || store('lipscribe-theme');
  setTheme(savedTheme === 'dark' || savedTheme === 'light' ? savedTheme
    : (window.matchMedia && matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'));
  el.themeBtn.addEventListener('click', function () {
    var t = el.root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    setTheme(t); store('lipscribe-theme', t);
  });

  // ---------- Toggles ----------
  var runBtns = document.querySelectorAll('[data-run]');
  var modeBtns = document.querySelectorAll('[data-mode]');
  function pressOne(list, attr, val) {
    list.forEach(function (b) { b.setAttribute('aria-pressed', String(b.getAttribute(attr) === val)); });
  }
  runBtns.forEach(function (b) {
    b.addEventListener('click', function () { S.run = b.getAttribute('data-run'); pressOne(runBtns, 'data-run', S.run); resetResults(); render(); });
  });
  modeBtns.forEach(function (b) {
    b.addEventListener('click', function () { S.mode = b.getAttribute('data-mode'); pressOne(modeBtns, 'data-mode', S.mode); resetResults(); render(); });
  });
  if (params.get('run') === 'clip') { S.run = 'clip'; pressOne(runBtns, 'data-run', 'clip'); }
  if (params.get('mode') === 'any') { S.mode = 'any'; pressOne(modeBtns, 'data-mode', 'any'); }

  // ---------- Gate plot ----------
  var barEls = [];
  for (var b = 0; b < WINDOW_FRAMES; b++) {
    var bar = document.createElement('div');
    bar.className = 'bar empty';
    bar.innerHTML = '<i class="s"></i><i class="l"></i>';
    el.bars.appendChild(bar);
    barEls.push(bar);
  }
  function renderGate() {
    var vals = S.gate, shown;
    if (S.run === 'live') {
      shown = vals.slice(-WINDOW_FRAMES);
      while (shown.length < WINDOW_FRAMES) shown.unshift(null);
    } else {
      shown = [];
      for (var i = 0; i < WINDOW_FRAMES; i++) {
        if (!vals.length) { shown.push(null); continue; }
        var lo = Math.floor(i * vals.length / WINDOW_FRAMES), hi = Math.max(lo + 1, Math.floor((i + 1) * vals.length / WINDOW_FRAMES));
        var sum = 0; for (var k = lo; k < hi; k++) sum += vals[k];
        shown.push(sum / (hi - lo));
      }
    }
    var total = 0, count = 0;
    shown.forEach(function (v, i) {
      var bar = barEls[i], s = bar.firstChild, l = bar.lastChild;
      if (v === null || v === undefined) { bar.className = 'bar empty'; s.style.flexGrow = 1; l.style.flexGrow = 0.0001; return; }
      bar.className = 'bar';
      var lips = clamp(v, 0, 1);
      s.style.flexGrow = Math.max(1 - lips, 0.001); l.style.flexGrow = Math.max(lips, 0.001);
      total += lips; count++;
    });
    if (count) {
      var lp = Math.round(total / count * 100);
      el.lipsPct.textContent = lp + '%'; el.soundPct.textContent = (100 - lp) + '%';
      el.bars.setAttribute('aria-label', 'Trust per frame. On average the model trusted the sound ' + (100 - lp) + '% and the lips ' + lp + '%.');
    } else {
      el.lipsPct.textContent = '--'; el.soundPct.textContent = '--';
    }
  }

  // ---------- Captions ----------
  function renderCaption(target, text, pending, diffs, caret, placeholder) {
    var w = words(text), p = words(pending);
    target.textContent = '';
    if (!w.length && !p.length) {
      var ph = document.createElement('span'); ph.className = 'placeholder'; ph.textContent = placeholder;
      target.appendChild(ph);
      if (caret) target.appendChild(caretEl());
      return;
    }
    var start = Math.max(0, w.length - MAX_WORDS);
    if (start > 0) target.appendChild(document.createTextNode('… '));
    for (var i = start; i < w.length; i++) {
      var span = document.createElement('span');
      span.textContent = w[i];
      if (diffs && diffs.has(i)) { span.className = 'diff'; span.title = 'Different from Ears + lips'; }
      target.appendChild(span);
      target.appendChild(document.createTextNode(' '));
    }
    if (p.length) {
      var pend = document.createElement('span'); pend.className = 'pending'; pend.textContent = p.join(' ');
      target.appendChild(pend);
    }
    if (caret) target.appendChild(caretEl());
  }
  function caretEl() { var c = document.createElement('span'); c.className = 'caret'; c.setAttribute('aria-hidden', 'true'); return c; }

  var last = { ears: { text: '', pending: '' }, lips: { text: '', pending: '' } };
  function renderCaptions() {
    var live = S.phase === 'recording' && S.run === 'live';
    var diffs = diffIndexes(words(last.ears.text), words(last.lips.text));
    renderCaption(el.earsText, last.ears.text, last.ears.pending, diffs, live, S.run === 'live' ? 'Captions from sound alone appear here as you talk.' : 'Captions from sound alone appear here.');
    renderCaption(el.lipsText, last.lips.text, last.lips.pending, null, live, S.run === 'live' ? 'Captions from sound and lips appear here as you talk.' : 'Captions from sound and lips appear here.');
    var tail = words(last.lips.text + ' ' + last.lips.pending).slice(-10).join(' ');
    el.overlay.textContent = tail;
    el.overlay.classList.toggle('show', live && !!tail);
  }

  // ---------- Noise ----------
  function renderNoise(db) {
    if (db === null || db === undefined || isNaN(db)) return;
    var pct = clamp((20 - db) / 30, 0, 1) * 100;
    var word = db >= 15 ? 'Quiet room' : db >= 5 ? 'Some noise' : db >= -2 ? 'Noisy room' : 'Very noisy';
    var r = Math.round(db);
    el.noiseDb.textContent = '≈' + minus(r) + 'dB';
    el.noiseWord.textContent = word;
    el.gaugeKnob.style.left = pct + '%';
    el.gauge.setAttribute('aria-valuenow', String(Math.round(pct)));
    el.gauge.setAttribute('aria-valuetext', word + ', about ' + r + ' decibels signal to noise');
  }

  // ---------- Mouth box + crops ----------
  function renderMouthBox(box) {
    if (!box || box.length !== 4) { el.mouthBox.style.display = 'none'; el.facePill.hidden = true; return; }
    el.mouthBox.style.display = 'block';
    el.mouthBox.style.left = box[0] * 100 + '%'; el.mouthBox.style.top = box[1] * 100 + '%';
    el.mouthBox.style.width = box[2] * 100 + '%'; el.mouthBox.style.height = box[3] * 100 + '%';
    el.facePill.hidden = false;
  }
  function renderCrops() {
    el.crops.textContent = '';
    if (!S.crops.length) {
      for (var i = 0; i < MAX_CROPS; i++) {
        var f = document.createElement('figure');
        f.innerHTML = '<div class="crop-empty" aria-hidden="true"></div><figcaption>--</figcaption>';
        el.crops.appendChild(f);
      }
      return;
    }
    S.crops.slice(-MAX_CROPS).forEach(function (c) {
      var fig = document.createElement('figure');
      var img = document.createElement('img');
      var t = typeof c.t === 'number' ? c.t.toFixed(1) + 's' : '';
      img.src = c.src; img.alt = 'Mouth crop' + (t ? ' at ' + t : ''); img.width = 84; img.height = 84;
      var cap = document.createElement('figcaption'); cap.textContent = t;
      fig.appendChild(img); fig.appendChild(cap);
      el.crops.appendChild(fig);
    });
  }

  // ---------- Errors + loading ----------
  function showError(code, detail) {
    var e = ERRORS[code] || ERRORS.unknown;
    el.errorTitle.textContent = e[0];
    el.errorText.textContent = e[1];
    el.errorCode.textContent = 'code: ' + (code || 'unknown');
    if (detail) console.warn('[LipScribe]', code, detail);
    el.errorBox.classList.add('show');
    el.loading.style.display = 'none';
    el.cards.style.display = '';
  }
  function hideError() { el.errorBox.classList.remove('show'); }
  el.errorRetry.addEventListener('click', function () {
    hideError();
    if (!S.stream) startCamera(); else if (S.phase === 'ready') startRun();
  });

  var STEP_ORDER = ['sent', 'face', 'listen', 'write'];
  function setStep(step) {
    var idx = STEP_ORDER.indexOf(step);
    el.steps.querySelectorAll('li').forEach(function (li, i) {
      li.className = i < idx ? 'done' : i === idx ? 'now' : '';
    });
  }

  // ---------- Main render ----------
  function render() {
    var live = S.run === 'live', rec = S.phase === 'recording', wait = S.phase === 'waiting';
    el.left.classList.toggle('recording', rec);
    el.recBtn.disabled = S.phase === 'nocam' || wait;
    el.recBtn.setAttribute('aria-label', rec ? (live ? 'Stop live captions' : 'Stop recording') : (live ? 'Start live captions' : 'Record a clip'));
    el.recTitle.textContent = rec ? (live ? 'Live captions are on' : 'Recording…') : wait ? 'Working on it…' : (live ? 'Start live captions' : 'Record a clip');
    el.recHelp.textContent = live
      ? 'Talk normally. Words appear as you speak; tap again to stop.'
      : 'Face the camera and say one sentence. We stop at ' + CLIP_SECONDS + ' seconds.';
    el.recLabel.textContent = live ? 'LIVE' : 'REC';
    el.timerPill.hidden = !rec;
    runBtns.forEach(function (x) { x.disabled = rec || wait; });
    modeBtns.forEach(function (x) { x.disabled = rec || wait; });

    el.badge.classList.toggle('on', rec);
    el.badgeText.textContent = rec ? (live ? 'Listening' : 'Recording') : wait ? 'Reading your lips' : S.gotFinal ? 'Done' : 'Ready';

    el.gateSub.textContent = live ? 'The last 3 seconds, scrolling as you speak' : FPS + ' readings a second from the fusion gate';
    var axes = live ? ['−3s', '−2s', '−1s', 'now'] : ['0.0s', '1.0s', '2.0s', (S.gate.length / FPS || CLIP_SECONDS).toFixed(1) + 's'];
    axes.forEach(function (t, i) { el.ax[i].textContent = t; });

    var showLoading = wait && S.run === 'clip';
    el.loading.style.display = showLoading ? 'flex' : 'none';
    el.cards.style.display = showLoading ? 'none' : '';

    el.conn.className = 'conn ' + (useMock ? 'mock' : S.socket && S.socket.readyState === 1 ? 'ok' : '');
    el.connText.textContent = useMock ? 'Pretend backend (sample captions)' : S.socket && S.socket.readyState === 1 ? 'Connected to ' + safeHost(backendUrl) : 'Backend: ' + safeHost(backendUrl);
    renderCaptions();
  }
  function safeHost(u) { try { return new URL(u).host; } catch (e) { return u; } }

  function resetResults() {
    last = { ears: { text: '', pending: '' }, lips: { text: '', pending: '' } };
    S.gate = []; S.crops = []; S.gotFinal = false;
    renderGate(); renderCrops();
    el.statTime.innerHTML = 'Captioned in <strong>--</strong>';
    el.statFace.innerHTML = 'Face found in <strong>--</strong> of frames';
    hideError();
  }

  // ---------- Camera ----------
  function startCamera() {
    if (!window.isSecureContext) { showError('needs_https'); return; }
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) { showError('unsupported'); return; }
    el.camBtn.disabled = true;
    el.camEmptyText.textContent = 'Asking for your camera and microphone…';
    navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: FPS } },
      audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false }
    }).then(function (stream) {
      S.stream = stream;
      el.video.srcObject = stream;
      var p = el.video.play(); if (p && p.catch) p.catch(function () {});
      el.camEmpty.style.display = 'none';
      if (window.LipScribeMockSocket) window.LipScribeMockSocket.video = el.video;
      S.phase = 'ready';
      hideError();
      render();
    }).catch(function (err) {
      el.camBtn.disabled = false;
      el.camEmptyText.textContent = 'LipScribe needs your camera and microphone to read your lips and hear you.';
      var name = err && err.name;
      showError(name === 'NotAllowedError' || name === 'SecurityError' ? 'camera_denied'
        : name === 'NotFoundError' || name === 'OverconstrainedError' ? 'camera_missing'
        : name === 'NotReadableError' ? 'camera_busy' : 'unknown', err);
    });
  }
  el.camBtn.addEventListener('click', startCamera);
  // If permission was already given, open the camera straight away.
  if (navigator.permissions && navigator.permissions.query) {
    navigator.permissions.query({ name: 'camera' }).then(function (r) { if (r.state === 'granted') startCamera(); }).catch(function () {});
  }

  function pickMime() {
    if (!window.MediaRecorder) return '';
    var options = ['video/webm;codecs=vp8,opus', 'video/webm;codecs=vp9,opus', 'video/webm', 'video/mp4;codecs=avc1,mp4a', 'video/mp4'];
    for (var i = 0; i < options.length; i++) if (MediaRecorder.isTypeSupported(options[i])) return options[i];
    return '';
  }

  // ---------- Recording + socket ----------
  el.recBtn.addEventListener('click', function () {
    if (S.phase === 'ready') startRun();
    else if (S.phase === 'recording') stopRun();
  });
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && S.phase === 'recording') stopRun();
  });

  function startRun() {
    if (!window.MediaRecorder && !useMock) { showError('unsupported'); return; }
    resetResults();
    var sock;
    try { sock = useMock ? new window.LipScribeMockSocket('mock://pretend-backend') : new WebSocket(backendUrl); }
    catch (e) { showError('server_unreachable', e); return; }
    S.socket = sock;
    S.phase = 'waiting'; // until the socket opens
    render();
    S.connectTimer = setTimeout(function () { if (sock.readyState !== 1) fail('server_unreachable'); }, CONNECT_TIMEOUT_MS);

    sock.onopen = function () {
      clearTimeout(S.connectTimer);
      if (S.socket !== sock) return;
      var mime = pickMime();
      sock.send(JSON.stringify({ type: 'start', run: S.run, mode: S.mode, mime: mime, chunk_ms: CHUNK_MS, fps: FPS, clip_seconds: S.run === 'clip' ? CLIP_SECONDS : null }));
      if (window.MediaRecorder) {
        try {
          S.recorder = mime ? new MediaRecorder(S.stream, { mimeType: mime }) : new MediaRecorder(S.stream);
        } catch (e) { fail('unsupported', e); return; }
        S.recorder.ondataavailable = function (ev) {
          if (ev.data && ev.data.size && sock.readyState === 1) sock.send(ev.data);
        };
        S.recorder.onstop = function () { if (sock.readyState === 1) sock.send(JSON.stringify({ type: 'stop' })); };
        S.recorder.start(CHUNK_MS);
      }
      S.phase = 'recording';
      S.startedAt = performance.now();
      tick();
      render();
    };
    sock.onmessage = function (ev) {
      if (S.socket !== sock) return;
      var msg; try { msg = JSON.parse(ev.data); } catch (e) { return; }
      handle(msg);
    };
    sock.onerror = function (e) { if (S.socket === sock && !S.gotFinal) fail(sock.readyState === 1 ? 'connection_lost' : 'server_unreachable', e); };
    sock.onclose = function () {
      if (S.socket !== sock) return;
      if (!S.gotFinal && (S.phase === 'recording' || S.phase === 'waiting')) fail('connection_lost');
      render();
    };
  }

  function tick() {
    if (S.phase !== 'recording') return;
    var sec = (performance.now() - S.startedAt) / 1000;
    if (S.run === 'clip' && sec >= CLIP_SECONDS) { el.timer.textContent = fmtTime(CLIP_SECONDS); stopRun(); return; }
    el.timer.textContent = fmtTime(sec);
    S.raf = requestAnimationFrame(tick);
  }

  function stopRun() {
    if (S.phase !== 'recording') return;
    cancelAnimationFrame(S.raf);
    S.phase = 'waiting';
    if (S.recorder && S.recorder.state !== 'inactive') S.recorder.stop();   // sends {"type":"stop"} after the last chunk
    else if (S.socket && S.socket.readyState === 1) S.socket.send(JSON.stringify({ type: 'stop' }));
    if (S.run === 'clip') setStep('sent');
    S.replyTimer = setTimeout(function () { if (!S.gotFinal) fail('no_reply'); }, REPLY_TIMEOUT_MS);
    render();
  }

  function handle(msg) {
    switch (msg.type) {
      case 'ready':
        break;
      case 'progress':
        if (msg.step) setStep(msg.step);
        break;
      case 'partial':
      case 'final':
        if (msg.ears) { last.ears = { text: msg.ears.text || '', pending: msg.ears.pending || '' }; if (msg.ears.model) el.earsModel.textContent = msg.ears.model; }
        if (msg.lips) { last.lips = { text: msg.lips.text || '', pending: msg.lips.pending || '' }; if (msg.lips.model) el.lipsModel.textContent = msg.lips.model; }
        if (Array.isArray(msg.gate) && msg.gate.length) {
          S.gate = msg.type === 'final' && S.run === 'clip' ? msg.gate.slice() : S.gate.concat(msg.gate);
          if (S.gate.length > 4 * WINDOW_FRAMES && S.run === 'live') S.gate = S.gate.slice(-WINDOW_FRAMES);
          renderGate();
        }
        if (typeof msg.noise_db === 'number') renderNoise(msg.noise_db);
        if ('mouth_box' in msg) renderMouthBox(msg.mouth_box);
        if (Array.isArray(msg.crops) && msg.crops.length) {
          S.crops = msg.type === 'final' && S.run === 'clip' ? msg.crops.slice() : S.crops.concat(msg.crops).slice(-MAX_CROPS);
          renderCrops();
        }
        if (msg.type === 'final') finish(msg); else renderCaptions();
        break;
      case 'error':
        fail(msg.code || 'unknown', msg.message);
        break;
    }
  }

  function finish(msg) {
    S.gotFinal = true;
    clearTimeout(S.replyTimer);
    if (typeof msg.processing_s === 'number') el.statTime.innerHTML = 'Captioned in <strong>' + msg.processing_s.toFixed(1) + ' s</strong> after you stopped';
    if (typeof msg.face_found === 'number') el.statFace.innerHTML = 'Face found in <strong>' + Math.round(msg.face_found * 100) + '%</strong> of frames';
    endRun();
  }

  function fail(code, detail) {
    showError(code, detail);
    endRun();
  }

  function endRun() {
    cancelAnimationFrame(S.raf);
    clearTimeout(S.connectTimer); clearTimeout(S.replyTimer);
    if (S.recorder && S.recorder.state !== 'inactive') { S.recorder.onstop = null; try { S.recorder.stop(); } catch (e) {} }
    S.recorder = null;
    var sock = S.socket; S.socket = null;
    if (sock && sock.readyState <= 1) { try { sock.close(); } catch (e) {} }
    renderMouthBox(null);
    S.phase = S.stream ? 'ready' : 'nocam';
    render();
  }

  // ---------- First paint ----------
  renderGate();
  renderCrops();
  render();
})();

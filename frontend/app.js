/*
 * LipScribe demo page (F-06) — owner: Wakisa Isabelle, reviewer: Cole Rodney Migisha
 *
 * Talks to the backend over one WebSocket. Message format: backend/README.md.
 *   - No ?backend=... in the address  -> the pretend backend in mock-backend.js
 *   - ?backend=wss://host/ws/live      -> the real backend on the Alienware
 *   - ?backend=self                    -> the backend that served this page (SSH tunnel, localhost:8800)
 *   - ?code=...                        -> the access code, passed on to the backend (remembered)
 * The default backend address is set in config.js (DEPLOYMENT.md).
 * Upload (in Clip mode): the file plays here from disk; the backend asks for the bytes it needs
 * ("need"), and subtitle cues come back in batches ("cues") while the rest uploads.
 */
(function () {
  'use strict';

  // Set in config.js (e.g. 'wss://lipscribe-api.ucu-dct.org/ws/live'). When the backend serves this page itself, it sets
  // window.LIPSCRIBE_BACKEND = 'self', so the page talks to wherever it was loaded from.
  var DEFAULT_BACKEND = window.LIPSCRIBE_BACKEND || '';
  var CHUNK_MS = 1000;        // how often a piece of audio+video is sent
  var CLIP_SECONDS = 3;       // Clip mode stops by itself after this
  var FPS = 25;               // gate readings per second
  var WINDOW_FRAMES = 75;     // gate plot shows 3 s
  var MAX_CROPS = 8;
  var MAX_WORDS = 24;         // words kept on screen per caption in Live mode
  var CONNECT_TIMEOUT_MS = 8000;
  var REPLY_TIMEOUT_MS = 20000;
  var BUFFER_AHEAD_S = 4;     // resume a paused upload once captions are this far ahead

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
    bad_code: ['Wrong access code', 'Open the link with ?code=… that the team gave you.'],
    bad_file: ['We couldn’t read that file', 'Try an .mp4, .mkv, .mov or .webm video with sound.'],
    file_too_large: ['That file is too big', 'Files up to 8 GB (about 3 hours) can be captioned.'],
    grid_too_long: ['GRID sentences takes short clips', 'Upload a clip of up to 10 seconds, or switch to Any speech for longer videos.'],
    no_meeting_audio: ['The shared screen has no sound', 'Share again and tick “Share tab audio” (or “Share system audio” on Windows), or pick a sound device under “Meeting sound from”.'],
    display_denied: ['Screen sharing was cancelled', 'Press the button again and choose the meeting’s tab, window or screen.'],
    bad_zoom_link: ['That isn’t a Zoom caption link', 'Paste the caption API token from Zoom (CC → Set up manual captioner → Copy the API token), or leave the box empty.'],
    needs_backend: ['Uploads need the real backend', 'Open the page with ?backend=… to caption a file; the pretend backend only does camera captions.'],
    model_unavailable: ['That model isn’t loaded', 'Switch model, or ask the team to start the server with both models.'],
    server_error: ['The server hit a problem', 'Try again. If it keeps happening, tell the team what you did.'],
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
    themeBtn: $('themeBtn'),
    camera: document.querySelector('.camera'), player: $('filePlayer'), snrSel: $('snrSel'),
    uploadBox: $('uploadBox'), upIntro: $('upIntro'), fileInput: $('fileInput'), fileBtn: $('fileBtn'), backBtn: $('backBtn'),
    upPanel: $('upPanel'), barUp: $('barUp'), barCap: $('barCap'), barHead: $('barHead'), upSent: $('upSent'), upCap: $('upCap'),
    srtBtn: $('srtBtn'), lipsChip: $('lipsChip'), lipsFoot: $('lipsFoot'), upNote: $('upNote'), noiseRow: $('noiseRow'),
    meetingBox: $('meetingBox'), meetingPanel: $('meetingPanel'), soundSel: $('soundSel'), zoomUrl: $('zoomUrl'), zoomStatus: $('zoomStatus')
  };

  // ---------- State ----------
  var params = new URLSearchParams(location.search);
  // A ?backend=wss://... address is remembered in this browser, so a Pages link works on later visits too.
  var givenBackend = params.get('backend');
  if (givenBackend && givenBackend !== 'self') store('lipscribe-backend', givenBackend === 'mock' ? '' : givenBackend);
  var backendUrl = toWs(givenBackend || (DEFAULT_BACKEND === 'self' ? 'self' : store('lipscribe-backend') || DEFAULT_BACKEND));
  var useMock = !backendUrl || backendUrl === 'mock';
  if (params.get('code')) store('lipscribe-code', params.get('code'));
  var accessCode = params.get('code') || store('lipscribe-code') || '';
  function socketUrl() {
    if (!accessCode) return backendUrl;
    return backendUrl + (backendUrl.indexOf('?') < 0 ? '?' : '&') + 'code=' + encodeURIComponent(accessCode);
  }
  function snrValue() { var v = el.snrSel ? el.snrSel.value : ''; return v === '' ? null : Number(v); }
  var S = {
    run: 'live', mode: 'grid', stream: null, recorder: null, socket: null,
    phase: 'nocam',       // nocam | ready | recording | waiting
    startedAt: 0, raf: 0, gate: [], crops: [], connectTimer: 0, replyTimer: 0, gotFinal: false,
    up: null,             // upload in progress or finished: see "Upload" below
    source: 'camera',     // Live mode: 'camera' or 'meeting' (a shared tab, window or screen)
    runStream: null       // the stream being recorded when it is not the camera
  };

  // ---------- Helpers ----------
  function toWs(u) {
    if (!u) return '';
    if (u === 'mock') return 'mock';
    if (u === 'self') return (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws/live';
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
    b.addEventListener('click', function () { if (S.up) stopUpload(); S.run = b.getAttribute('data-run'); pressOne(runBtns, 'data-run', S.run); resetResults(); render(); });
  });
  modeBtns.forEach(function (b) {
    b.addEventListener('click', function () { if (S.up) stopUpload(); S.mode = b.getAttribute('data-mode'); pressOne(modeBtns, 'data-mode', S.mode); resetResults(); render(); });
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
    renderCaption(el.earsText, last.ears.text, last.ears.pending, diffs, live, S.up ? 'Play the video: each line from sound alone appears here.' : S.run === 'live' ? 'Captions from sound alone appear here as you talk.' : 'Captions from sound alone appear here.');
    renderCaption(el.lipsText, last.lips.text, last.lips.pending, null, live, S.up ? 'Play the video: each line from sound and lips appears here.' : S.run === 'live' ? 'Captions from sound and lips appear here as you talk.' : 'Captions from sound and lips appear here.');
    if (S.up) return;                    // the upload code drives the overlay
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
    el.uploadBox.hidden = S.run !== 'clip' && !S.up;
    el.noiseRow.hidden = !!S.up;           // the room-noise gauge is for the microphone only
    el.camera.classList.toggle('file', !!S.up);
    if (S.up) { renderUploadState(); return; }
    el.fileBtn.classList.toggle('disabled', S.phase === 'recording' || S.phase === 'waiting');
    el.backBtn.hidden = true; el.upPanel.hidden = true; el.lipsChip.hidden = true;
    var live = S.run === 'live', rec = S.phase === 'recording', wait = S.phase === 'waiting';
    var meeting = live && S.source === 'meeting';
    el.meetingBox.hidden = !live;
    el.meetingPanel.hidden = !meeting;
    if (sourceBtns) sourceBtns.forEach(function (x) { x.disabled = rec || wait; });
    el.camera.classList.toggle('screen', meeting && !!S.runStream);
    el.left.classList.toggle('recording', rec);
    el.recBtn.disabled = (S.phase === 'nocam' && !meeting) || wait;
    el.recBtn.setAttribute('aria-label', rec ? (live ? 'Stop live captions' : 'Stop recording') : (live ? 'Start live captions' : 'Record a clip'));
    el.recTitle.textContent = rec ? (meeting ? 'Captioning the meeting' : live ? 'Live captions are on' : 'Recording…') : wait ? 'Working on it…' : (meeting ? 'Caption a meeting' : live ? 'Start live captions' : 'Record a clip');
    el.recHelp.textContent = meeting
      ? (rec ? 'Captions follow the meeting a few seconds behind. Tap to stop, or stop sharing.' : 'Choose the meeting’s tab, window or screen when the browser asks.')
      : live
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
    if (S.up) { if (!S.up.done && !S.up.failed) cancelUpload(); return; }
    if (S.run === 'live' && S.source === 'meeting' && (S.phase === 'ready' || S.phase === 'nocam')) startMeeting();
    else if (S.phase === 'ready') startRun();
    else if (S.phase === 'recording') stopRun();
  });
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && S.phase === 'recording') stopRun();
  });

  function startRun() {
    if (!window.MediaRecorder && !useMock) { showError('unsupported'); return; }
    resetResults();
    var sock;
    try { sock = useMock ? new window.LipScribeMockSocket('mock://pretend-backend') : new WebSocket(socketUrl()); }
    catch (e) { showError('server_unreachable', e); return; }
    S.socket = sock;
    S.phase = 'waiting'; // until the socket opens
    render();
    S.connectTimer = setTimeout(function () { if (sock.readyState !== 1) fail('server_unreachable'); }, CONNECT_TIMEOUT_MS);

    sock.onopen = function () {
      clearTimeout(S.connectTimer);
      if (S.socket !== sock) return;
      var mime = pickMime();
      var meetingRun = !!S.runStream;
      sock.send(JSON.stringify({ type: 'start', run: S.run, mode: S.mode, mime: mime, chunk_ms: CHUNK_MS, fps: FPS,
        clip_seconds: S.run === 'clip' ? CLIP_SECONDS : null, snr: snrValue(), source: meetingRun ? 'meeting' : 'camera',
        zoom_url: meetingRun && el.zoomUrl.value.trim() ? el.zoomUrl.value.trim() : null }));
      if (window.MediaRecorder) {
        try {
          var recStream = S.runStream || S.stream;
          S.recorder = mime ? new MediaRecorder(recStream, { mimeType: mime }) : new MediaRecorder(recStream);
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
    if (S.up && handleUpload(msg)) return;
    switch (msg.type) {
      case 'ready':
        break;
      case 'zoom':
        el.zoomStatus.className = 'zoom-status ' + (msg.ok ? 'ok' : 'bad');
        el.zoomStatus.textContent = msg.ok ? 'Sent ' + msg.sent + (msg.sent === 1 ? ' line' : ' lines') + ' to Zoom' : 'Zoom: ' + (msg.message || 'could not send');
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
    if (S.up && !S.up.done) S.up.failed = true;
    cancelAnimationFrame(S.raf);
    clearTimeout(S.connectTimer); clearTimeout(S.replyTimer);
    if (S.recorder && S.recorder.state !== 'inactive') { S.recorder.onstop = null; try { S.recorder.stop(); } catch (e) {} }
    S.recorder = null;
    var sock = S.socket; S.socket = null;
    if (sock && sock.readyState <= 1) { try { sock.close(); } catch (e) {} }
    renderMouthBox(null);
    if (S.runStream) {
      S.runStream.getTracks().forEach(function (t) { t.stop(); });
      S.runStream = null;
      el.video.srcObject = S.stream || null;
      if (!S.stream) el.camEmpty.style.display = '';
    }
    S.phase = S.stream ? 'ready' : 'nocam';
    render();
  }

  // ---------- Meeting captions (Live mode) ----------
  // The meeting's picture comes from screen sharing; its sound from the shared tab or screen, or
  // from an audio input (a virtual device such as BlackHole on a Mac, "Monitor of …" on Linux).
  var sourceBtns = document.querySelectorAll('[data-source]');
  sourceBtns.forEach(function (b) {
    b.addEventListener('click', function () {
      S.source = b.getAttribute('data-source');
      pressOne(sourceBtns, 'data-source', S.source);
      if (S.source === 'meeting') listSoundInputs();
      resetResults(); render();
    });
  });
  var savedZoom = store('lipscribe-zoom'); if (savedZoom) el.zoomUrl.value = savedZoom;
  el.zoomUrl.addEventListener('change', function () { store('lipscribe-zoom', el.zoomUrl.value.trim()); el.zoomStatus.textContent = ''; });

  function listSoundInputs() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
    navigator.mediaDevices.enumerateDevices().then(function (devs) {
      var keep = el.soundSel.value;
      while (el.soundSel.options.length > 1) el.soundSel.remove(1);
      devs.filter(function (d) { return d.kind === 'audioinput'; }).forEach(function (d, i) {
        var o = document.createElement('option');
        o.value = d.deviceId; o.textContent = d.label || ('Sound input ' + (i + 1));
        el.soundSel.appendChild(o);
      });
      el.soundSel.value = keep;
      if (el.soundSel.selectedIndex < 0) el.soundSel.selectedIndex = 0;
    }).catch(function () {});
  }
  el.soundSel.addEventListener('focus', listSoundInputs);

  function startMeeting() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getDisplayMedia) { showError('unsupported'); return; }
    if (!window.isSecureContext) { showError('needs_https'); return; }
    var sound = el.soundSel.value;
    navigator.mediaDevices.getDisplayMedia({
      video: { frameRate: { ideal: FPS }, width: { max: 1280 }, height: { max: 1280 } },
      audio: sound === 'shared' ? { echoCancellation: false, noiseSuppression: false, autoGainControl: false } : false,
      systemAudio: 'include', selfBrowserSurface: 'exclude', surfaceSwitching: 'include'
    }).then(function (screen) {
      if (sound === 'shared') {
        if (!screen.getAudioTracks().length) { screen.getTracks().forEach(function (t) { t.stop(); }); showError('no_meeting_audio'); return null; }
        return screen;
      }
      return navigator.mediaDevices.getUserMedia({ audio: { deviceId: { exact: sound }, echoCancellation: false, noiseSuppression: false, autoGainControl: false } })
        .then(function (mic) { return new MediaStream(screen.getVideoTracks().concat(mic.getAudioTracks())); })
        .catch(function (e) { screen.getTracks().forEach(function (t) { t.stop(); }); throw e; });
    }).then(function (stream) {
      if (!stream) return;
      S.runStream = stream;
      el.video.srcObject = stream;
      var p = el.video.play(); if (p && p.catch) p.catch(function () {});
      el.camEmpty.style.display = 'none';
      if (window.LipScribeMockSocket) window.LipScribeMockSocket.video = el.video;
      stream.getVideoTracks()[0].addEventListener('ended', function () { if (S.phase === 'recording') stopRun(); });
      el.zoomStatus.textContent = el.zoomUrl.value.trim() ? 'Waiting for the first line…' : '';
      el.zoomStatus.className = 'zoom-status muted';
      S.phase = 'ready';
      startRun();
    }).catch(function (err) {
      var name = err && err.name;
      showError(name === 'NotAllowedError' || name === 'AbortError' ? 'display_denied' : name === 'NotFoundError' ? 'camera_missing' : 'unknown', err);
    });
  }
  if (params.get('source') === 'meeting') { S.source = 'meeting'; pressOne(sourceBtns, 'data-source', 'meeting'); }

  // ---------- Upload (Clip mode) ----------
  // The video plays from the person's own disk; only captions come back. Cues arrive in order, a
  // batch at a time. If playback catches up with the captions it pauses ("buffering") and resumes
  // once they are BUFFER_AHEAD_S ahead again.
  var REASONS = {
    'camera cut': 'the shot changes', 'no face': 'no face on screen', 'faces=0': 'no face on screen',
    'faces=2': 'two faces on screen', '2nd face': 'two faces on screen', 'not frontal': 'the face is turned away',
    'face too small': 'the face is too small', 'face lost': 'the face is not visible throughout',
    'lips still': 'the lips are not moving', 'no video': 'there is no picture'
  };
  var trackBtns = document.querySelectorAll('[data-track]');

  function fmtClock(sec) {
    sec = Math.max(0, Math.floor(sec || 0));
    var h = Math.floor(sec / 3600), m = Math.floor(sec % 3600 / 60), s2 = sec % 60;
    return (h ? h + ':' + String(m).padStart(2, '0') : String(m)) + ':' + String(s2).padStart(2, '0');
  }
  function trackText(c, track) { return track === 'ears' ? c.ears : track === 'lips' ? c.lips : (c.lips_used ? c.lips : c.ears); }
  function cueAt(t) {
    var cues = S.up.cues, lo = 0, hi = cues.length - 1, best = -1;
    while (lo <= hi) { var mid = (lo + hi) >> 1; if (cues[mid].start <= t) { best = mid; lo = mid + 1; } else hi = mid - 1; }
    if (best < 0) return null;
    var c = cues[best], next = cues[best + 1];
    var hold = Math.min(c.end + 0.8, next ? next.start : Infinity);   // keep a line up briefly after it is said
    return t <= hold ? c : null;
  }

  function startUpload(file) {
    if (!file) return;
    if (useMock) { showError('needs_backend'); return; }
    if (S.phase === 'recording' || S.phase === 'waiting') return;
    var track = S.up ? S.up.track : 'routed';
    if (S.up) stopUpload();
    resetResults();
    var up = S.up = { file: file, url: URL.createObjectURL(file), cues: [], captioned: 0, duration: 0, uploaded: 0,
      size: file.size, done: false, failed: false, track: track, autoPaused: false, cur: undefined };
    el.upNote.hidden = true;
    el.player.src = up.url;
    el.player.load();
    var sock;
    try { sock = new WebSocket(socketUrl()); sock.binaryType = 'arraybuffer'; }
    catch (e) { showError('server_unreachable', e); return; }
    S.socket = sock;
    S.connectTimer = setTimeout(function () { if (sock.readyState !== 1) fail('server_unreachable'); }, CONNECT_TIMEOUT_MS);
    sock.onopen = function () {
      clearTimeout(S.connectTimer);
      if (S.socket !== sock) return;
      sock.send(JSON.stringify({ type: 'start', run: 'upload', mode: S.mode, name: file.name, size: file.size,
        mime: file.type || '', snr: snrValue() }));
      render();
    };
    sock.onmessage = function (ev) {
      if (S.socket !== sock) return;
      var msg; try { msg = JSON.parse(ev.data); } catch (e) { return; }
      handle(msg);
    };
    sock.onerror = function (e) { if (S.socket === sock && !up.done) fail(sock.readyState === 1 ? 'connection_lost' : 'server_unreachable', e); };
    sock.onclose = function () {
      if (S.socket !== sock) return;
      if (!up.done && !up.failed) fail('connection_lost');
      render();
    };
    render();
  }

  function handleUpload(msg) {
    var up = S.up;
    switch (msg.type) {
      case 'need': sendPiece(msg.offset, msg.length); return true;
      case 'ready':
        if (msg.ears_model) el.earsModel.textContent = msg.ears_model;
        if (msg.lips_model) el.lipsModel.textContent = msg.lips_model;
        return true;
      case 'progress':
        if (typeof msg.uploaded === 'number') up.uploaded = msg.uploaded;
        if (typeof msg.captioned_s === 'number') up.captioned = Math.max(up.captioned, msg.captioned_s);
        if (msg.duration_s) up.duration = msg.duration_s;
        renderUpBar();
        return true;
      case 'cues':
        (msg.cues || []).forEach(function (c) { up.cues.push(c); });
        up.cues.sort(function (a, b) { return a.start - b.start; });
        if (typeof msg.captioned_s === 'number') up.captioned = Math.max(up.captioned, msg.captioned_s);
        if (msg.duration_s) up.duration = msg.duration_s;
        el.srtBtn.disabled = !up.cues.length;
        maybeResume();
        showCue();
        return true;
      case 'final':
        up.done = true;
        up.captioned = up.duration || msg.duration_s || up.captioned;
        S.gotFinal = true;
        if (typeof msg.processing_s === 'number') el.statTime.innerHTML = 'Captioned in <strong>' + fmtClock(msg.processing_s) + '</strong> (' + up.cues.length + ' lines)';
        if (typeof msg.lips_share === 'number') el.statFace.innerHTML = 'Lips used for <strong>' + Math.round(msg.lips_share * 100) + '%</strong> of the speech';
        else if (typeof msg.face_found === 'number') el.statFace.innerHTML = 'Face found in <strong>' + Math.round(msg.face_found * 100) + '%</strong> of frames';
        var sock = S.socket; S.socket = null;
        if (sock && sock.readyState <= 1) { try { sock.close(); } catch (e) {} }
        maybeResume();
        showCue();
        render();
        return true;
      default:
        return false;              // errors go through the normal path
    }
  }

  function sendPiece(off, len) {
    var sock = S.socket, up = S.up;
    if (!sock || !up) return;
    up.file.slice(off, off + len).arrayBuffer().then(function (buf) {
      if (sock.readyState !== 1) return;
      var out = new Uint8Array(8 + buf.byteLength), dv = new DataView(out.buffer);
      dv.setUint32(0, Math.floor(off / 4294967296));      // 8-byte big-endian offset, then the bytes
      dv.setUint32(4, off >>> 0);
      out.set(new Uint8Array(buf), 8);
      sock.send(out.buffer);
    }).catch(function (e) { fail('bad_file', e); });
  }

  function renderUpBar() {
    var up = S.up; if (!up) return;
    var d = up.duration || el.player.duration || 0;
    var sent = up.size ? Math.min(1, up.uploaded / up.size) : 0;
    el.barUp.style.width = (sent * 100) + '%';
    el.barCap.style.width = (d ? Math.min(100, up.captioned / d * 100) : 0) + '%';
    el.barHead.style.left = (d && isFinite(el.player.currentTime) ? Math.min(100, el.player.currentTime / d * 100) : 0) + '%';
    el.upSent.textContent = 'Uploaded ' + Math.round(sent * 100) + '% of ' + (up.size / 1048576).toFixed(up.size > 1e9 ? 0 : 1) + ' MB';
    el.upCap.textContent = up.done ? 'Captioned all ' + fmtClock(d) + ' · ' + up.cues.length + ' lines'
      : 'Captioned to ' + fmtClock(up.captioned) + (d ? ' of ' + fmtClock(d) : '');
  }

  function maybeResume() {
    var up = S.up;
    if (up && up.autoPaused && (up.done || up.captioned >= el.player.currentTime + BUFFER_AHEAD_S)) {
      up.autoPaused = false;
      var p = el.player.play(); if (p && p.catch) p.catch(function () {});
    }
  }

  function showCue() {
    var up = S.up; if (!up) return;
    var t = el.player.currentTime || 0;
    var d = up.duration || el.player.duration || Infinity;
    if (!up.done && !up.failed && !el.player.paused && t > up.captioned - 0.3 && up.captioned < d - 0.5) {
      up.autoPaused = true;
      el.player.pause();
    }
    var c = cueAt(t);
    if (up.autoPaused) { el.overlay.textContent = 'Captioning ahead… the video goes on in a moment'; el.overlay.classList.add('show'); }
    else if (c) { el.overlay.textContent = trackText(c, up.track); el.overlay.classList.add('show'); }
    else el.overlay.classList.remove('show');
    if (c !== up.cur) {
      up.cur = c;
      last = { ears: { text: c ? c.ears : '', pending: '' }, lips: { text: c ? c.lips : '', pending: '' } };
      renderCaptions();
      S.gate = c ? (c.gate || []).slice() : []; renderGate();
      S.crops = c ? (c.crops || []).slice() : []; renderCrops();
      el.lipsChip.hidden = !c;
      if (c) {
        el.lipsChip.className = 'lips-chip' + (c.lips_used ? ' on' : '');
        el.lipsChip.textContent = c.lips_used ? 'Lips used for this line' : 'Ears only here: ' + (REASONS[c.reason] || c.reason);
      }
      var ax = c ? [c.start, c.start + (c.end - c.start) / 3, c.start + 2 * (c.end - c.start) / 3, c.end] : null;
      el.ax.forEach(function (a, i) { a.textContent = ax ? fmtClock(ax[i]) + '.' + Math.floor((ax[i] % 1) * 10) : '--'; });
    }
    renderUpBar();
  }

  function renderUploadState() {
    var up = S.up, busy = !up.done && !up.failed;
    el.left.classList.toggle('recording', busy);
    el.recBtn.disabled = !busy;
    el.recBtn.setAttribute('aria-label', busy ? 'Stop captioning the video' : 'Captioning finished');
    el.recTitle.textContent = busy ? 'Captioning your video…' : up.done ? 'Captions ready' : 'Captioning stopped';
    el.recHelp.textContent = busy ? 'Press play: captions arrive in batches, a little ahead of the picture. Tap to stop.'
      : up.done ? 'Switch the subtitle track, or download it as an .srt file.' : 'The lines captioned so far are kept; choose the video again to start over.';
    el.timerPill.hidden = true;
    runBtns.forEach(function (x) { x.disabled = busy; });
    modeBtns.forEach(function (x) { x.disabled = busy; });
    el.badge.classList.toggle('on', busy);
    el.badgeText.textContent = busy ? 'Captioning' : up.done ? 'Done' : 'Stopped';
    el.upIntro.textContent = up.file.name;
    el.upPanel.hidden = false;
    el.backBtn.hidden = busy;
    el.fileBtn.hidden = busy;
    el.loading.style.display = 'none';
    el.cards.style.display = '';
    el.gateSub.textContent = 'For the line playing now, ' + FPS + ' readings a second';
    el.lipsFoot.textContent = 'Listens and lip-reads together';
    pressOne(trackBtns, 'data-track', up.track);
    el.conn.className = 'conn ' + (S.socket && S.socket.readyState === 1 ? 'ok' : '');
    el.connText.textContent = S.socket && S.socket.readyState === 1 ? 'Connected to ' + safeHost(backendUrl) : 'Backend: ' + safeHost(backendUrl);
    renderUpBar();
    renderCaptions();
  }

  function cancelUpload() {
    var sock = S.socket;
    if (sock && sock.readyState === 1) { try { sock.send(JSON.stringify({ type: 'cancel' })); } catch (e) {} }
    S.up.failed = true;
    endRun();
  }

  function stopUpload() {
    var up = S.up; if (!up) return;
    if (!up.done && !up.failed) cancelUpload();
    S.up = null;
    el.player.pause();
    el.player.removeAttribute('src');
    el.player.load();
    URL.revokeObjectURL(up.url);
    el.overlay.classList.remove('show');
    el.fileBtn.hidden = false;
    resetResults();
    render();
  }

  function downloadSrt() {
    var up = S.up; if (!up || !up.cues.length) return;
    function ts(x) {
      var ms = Math.round(x * 1000), h = Math.floor(ms / 3600000), m = Math.floor(ms / 60000) % 60, s2 = Math.floor(ms / 1000) % 60;
      return String(h).padStart(2, '0') + ':' + String(m).padStart(2, '0') + ':' + String(s2).padStart(2, '0') + ',' + String(ms % 1000).padStart(3, '0');
    }
    var lines = [], n = 0;
    up.cues.forEach(function (c, i) {
      var text = trackText(c, up.track); if (!text) return;
      var next = up.cues[i + 1], end = Math.min(c.end + 0.5, next ? next.start : Infinity);
      lines.push(String(++n), ts(c.start) + ' --> ' + ts(end), text, '');
    });
    var blob = new Blob([lines.join('\r\n')], { type: 'application/x-subrip' });
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = up.file.name.replace(/\.[^.]+$/, '') + '.lipscribe-' + (up.track === 'routed' ? 'best' : up.track) + '.srt';
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(function () { URL.revokeObjectURL(a.href); }, 1000);
  }

  el.fileInput.addEventListener('change', function () { var f = el.fileInput.files && el.fileInput.files[0]; el.fileInput.value = ''; startUpload(f); });
  el.fileBtn.addEventListener('keydown', function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); el.fileInput.click(); } });
  el.backBtn.addEventListener('click', stopUpload);
  el.srtBtn.addEventListener('click', downloadSrt);
  trackBtns.forEach(function (b) {
    b.addEventListener('click', function () {
      if (!S.up) return;
      S.up.track = b.getAttribute('data-track');
      pressOne(trackBtns, 'data-track', S.up.track);
      S.up.cur = undefined;
      showCue();
    });
  });
  ['timeupdate', 'seeked', 'loadedmetadata'].forEach(function (ev) { el.player.addEventListener(ev, function () { if (S.up) showCue(); }); });
  el.player.addEventListener('error', function () { if (S.up) el.upNote.hidden = false; });
  el.player.addEventListener('play', function () { if (S.up && S.up.autoPaused) { S.up.autoPaused = false; } });

  // ---------- First paint ----------
  renderGate();
  renderCrops();
  render();
})();

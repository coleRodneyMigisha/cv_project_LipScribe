/*
 * LipScribe pretend backend (F-06)
 * --------------------------------
 * A fake WebSocket that speaks the same messages the real backend will send
 * (see "Message format" in demo/README.md). The page uses it whenever no
 * ?backend=wss://... address is given, so the frontend can be built and tested
 * without the Alienware.
 *
 * What is real and what is pretend:
 *   - Mouth crops are cut from YOUR camera in the browser (a fixed box, not a face finder).
 *   - Captions are sample sentences: GRID grammar sentences, or film subtitle lines.
 *   - The noise number drifts on its own; when it is noisy, "Ears only" makes mistakes
 *     and the gate leans towards the lips (about 20% in quiet, about 31% at the loudest,
 *     the pattern measured in Week 3). In a quiet room both captions match.
 *
 * Test the error messages with ?mock_error=no_face | no_audio | clip_too_short | server_busy
 * Test a quiet or noisy room with ?mock_noise=quiet | noisy
 */
(function () {
  'use strict';

  var GRID = {
    command: ['bin', 'lay', 'place', 'set'],
    colour: ['blue', 'green', 'red', 'white'],
    prep: ['at', 'by', 'in', 'with'],
    letter: 'abcdefghijklmnopqrstuvxyz'.split(''),
    digit: ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'],
    adverb: ['again', 'now', 'please', 'soon']
  };

  // Clean subtitle lines from the held-out films in the Flix v4 check, plus two from the deck.
  var FILM_LINES = [
    'i wish you all the best',
    'wait why didnt you just open it',
    'oh its one of mine actually',
    'i love you too',
    'like what',
    'she was walking her baby in the middle of the night',
    'with a toast to the bride and groom',
    'i used to be',
    'oh i thought you were a novelist not a screenwriter'
  ];

  // Words that sound alike, for the ears-only model's mistakes in noise.
  var SOUNDALIKE = {
    b: 'd', d: 'b', p: 't', t: 'p', m: 'n', n: 'm', j: 'g', g: 'j', v: 'b', e: 'b', c: 'z', z: 'c',
    five: 'nine', nine: 'five', now: 'snow', soon: 'spoon', blue: 'glue', red: 'bed', set: 'sit',
    lay: 'play', place: 'plays', green: 'cream', white: 'right', with: 'which', at: 'that',
    open: 'hoping', mine: 'mind', love: 'of', toast: 'toes', bride: 'bright', groom: 'room',
    night: 'knife', baby: 'maybe', best: 'vest', just: 'adjust', walking: 'working', used: 'youth'
  };

  function pick(list) { return list[Math.floor(Math.random() * list.length)]; }
  function clamp(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }

  function gridSentence() {
    return [pick(GRID.command), pick(GRID.colour), pick(GRID.prep), pick(GRID.letter), pick(GRID.digit), pick(GRID.adverb)];
  }

  // Ears-only version of a sentence: more mistakes when the room is noisier.
  function earsVersion(words, snr) {
    var out = words.slice();
    var errors = snr > 12 ? 0 : snr > 4 ? (Math.random() < 0.5 ? 1 : 0) : snr > -3 ? 1 : 2;
    for (var k = 0; k < errors; k++) {
      var i = Math.floor(Math.random() * out.length);
      var w = out[i];
      out[i] = SOUNDALIKE[w] || (w.length > 3 ? w.slice(0, -1) : w + 'e');
    }
    return out;
  }

  function lipsShare(snr) {
    // ~0.20 in a quiet room, ~0.31 at -10 dB, with frame-to-frame wobble.
    var base = 0.20 + 0.11 * clamp((20 - snr) / 30, 0, 1);
    return clamp(base + (Math.random() - 0.5) * 0.12, 0.04, 0.9);
  }

  var MOUTH_BOX = [0.36, 0.56, 0.28, 0.2]; // x, y, w, h as fractions of the camera frame

  function MockSocket(url) {
    this.url = url || 'mock://pretend-backend';
    this.readyState = 0;
    this.onopen = null; this.onmessage = null; this.onclose = null; this.onerror = null;
    this._timers = [];
    this._params = new URLSearchParams(location.search);
    var self = this;
    this._later(function () {
      self.readyState = 1;
      if (self.onopen) self.onopen({ type: 'open' });
      self._emit({ type: 'ready', backend: 'pretend', version: 'mock-1' });
    }, 250);
  }

  MockSocket.CONNECTING = 0; MockSocket.OPEN = 1; MockSocket.CLOSING = 2; MockSocket.CLOSED = 3;
  MockSocket.video = null; // app.js sets this to the <video> element so crops come from the real camera

  MockSocket.prototype._later = function (fn, ms) {
    var id = setTimeout(fn, ms);
    this._timers.push(id);
    return id;
  };

  MockSocket.prototype._emit = function (msg) {
    if (this.readyState !== 1 || !this.onmessage) return;
    this.onmessage({ data: JSON.stringify(msg) });
  };

  MockSocket.prototype.send = function (data) {
    if (typeof data !== 'string') { this._bytes = (this._bytes || 0) + (data.size || data.byteLength || 0); return; }
    var msg;
    try { msg = JSON.parse(data); } catch (e) { return; }
    if (msg.type === 'start') this._start(msg);
    if (msg.type === 'stop') this._stop();
  };

  MockSocket.prototype.close = function () {
    this._timers.forEach(clearTimeout);
    clearInterval(this._tick);
    if (this.readyState === 3) return;
    this.readyState = 3;
    if (this.onclose) this.onclose({ type: 'close', code: 1000 });
  };

  MockSocket.prototype._start = function (msg) {
    var self = this;
    this.run = msg.run === 'clip' ? 'clip' : 'live';
    this.mode = msg.mode === 'any' ? 'any' : 'grid';
    this.models = this.mode === 'grid' ? { ears: 'b1_ears', lips: 'p4' } : { ears: 'twin_v5', lips: 'ws_v5' };
    this.startedAt = performance.now();
    var noiseParam = this._params.get('mock_noise');
    this.snr = noiseParam === 'quiet' ? 22 : noiseParam === 'noisy' ? -6 : 8 + Math.random() * 10;
    this.lipsDone = []; this.earsDone = [];
    this.sentence = null; this.wordIndex = 0; this.pause = 0;
    this.gateAll = []; this.cropCount = 0; this.faceFrames = 0; this.frames = 0;

    var err = this._params.get('mock_error');
    if (err && err !== 'clip_too_short') {
      this._later(function () { self._error(err); }, this.run === 'live' ? 1500 : 3300);
      return;
    }
    if (this.run === 'live') {
      this._tick = setInterval(function () { self._liveStep(); }, 400);
    } else {
      this._tick = setInterval(function () { self._clipFrames(); }, 400);
    }
  };

  MockSocket.prototype._nextSentence = function () {
    var words = this.mode === 'grid' ? gridSentence() : pick(FILM_LINES).split(' ');
    this.sentence = { lips: words, ears: earsVersion(words, this.snr) };
    this.wordIndex = 0;
  };

  MockSocket.prototype._driftNoise = function () {
    if (this._params.get('mock_noise')) return;
    this.snr = clamp(this.snr + (Math.random() - 0.5) * 3.2, -10, 24);
  };

  MockSocket.prototype._gateFrames = function (n) {
    var g = [];
    for (var i = 0; i < n; i++) g.push(Math.round(lipsShare(this.snr) * 1000) / 1000);
    this.gateAll = this.gateAll.concat(g);
    this.frames += n; this.faceFrames += n;
    return g;
  };

  MockSocket.prototype._crop = function () {
    var v = MockSocket.video;
    if (!v || !v.videoWidth) return null;
    try {
      var c = document.createElement('canvas');
      c.width = 96; c.height = 96;
      var ctx = c.getContext('2d');
      var w = v.videoWidth, h = v.videoHeight;
      var bw = MOUTH_BOX[2] * w, bh = MOUTH_BOX[3] * h, side = Math.max(bw, bh);
      var cx = (MOUTH_BOX[0] + MOUTH_BOX[2] / 2) * w, cy = (MOUTH_BOX[1] + MOUTH_BOX[3] / 2) * h;
      ctx.filter = 'grayscale(1)';
      ctx.drawImage(v, cx - side / 2, cy - side / 2, side, side, 0, 0, 96, 96);
      return c.toDataURL('image/jpeg', 0.8);
    } catch (e) { return null; }
  };

  MockSocket.prototype._elapsed = function () { return (performance.now() - this.startedAt) / 1000; };

  MockSocket.prototype._liveStep = function () {
    this._driftNoise();
    if (!this.sentence) this._nextSentence();
    var finishedSentence = false;
    if (this.pause > 0) {
      this.pause--;
    } else {
      this.wordIndex++;
      if (this.wordIndex >= this.sentence.lips.length) finishedSentence = true;
    }
    var shownLips = this.lipsDone.concat(this.sentence.lips.slice(0, this.wordIndex));
    var shownEars = this.earsDone.concat(this.sentence.ears.slice(0, this.wordIndex));
    // The last word is still "pending" (it may change) until the sentence ends.
    var pendingLips = finishedSentence || this.pause > 0 ? '' : shownLips.pop() || '';
    var pendingEars = finishedSentence || this.pause > 0 ? '' : shownEars.pop() || '';
    var crop = this._crop();
    this._emit({
      type: 'partial',
      t: Math.round(this._elapsed() * 10) / 10,
      ears: { text: shownEars.join(' '), pending: pendingEars, model: this.models.ears },
      lips: { text: shownLips.join(' '), pending: pendingLips, model: this.models.lips },
      gate: this._gateFrames(10),
      noise_db: Math.round(this.snr * 10) / 10,
      mouth_box: MOUTH_BOX,
      crops: crop ? [{ t: Math.round(this._elapsed() * 10) / 10, src: crop }] : []
    });
    if (finishedSentence) {
      this.lipsDone = this.lipsDone.concat(this.sentence.lips);
      this.earsDone = this.earsDone.concat(this.sentence.ears);
      this.sentence = null;
      this.pause = 2;
      this._driftNoise();
    }
  };

  MockSocket.prototype._clipFrames = function () {
    // During a clip the server only collects; it shows the mouth box so you can line up.
    this._driftNoise();
    this._gateFrames(10);
    var crop = this._crop();
    if (crop) (this._clipCrops = this._clipCrops || []).push({ t: Math.round(this._elapsed() * 10) / 10, src: crop });
    this._emit({ type: 'partial', t: Math.round(this._elapsed() * 10) / 10, noise_db: Math.round(this.snr * 10) / 10, mouth_box: MOUTH_BOX });
  };

  MockSocket.prototype._stop = function () {
    var self = this;
    clearInterval(this._tick);
    var seconds = this._elapsed();
    if (this._params.get('mock_error') === 'clip_too_short' || (this.run === 'clip' && seconds < 1)) {
      this._later(function () { self._error('clip_too_short'); }, 300);
      return;
    }
    if (this.run === 'live') {
      var lips = this.lipsDone.concat(this.sentence ? this.sentence.lips.slice(0, this.wordIndex) : []);
      var ears = this.earsDone.concat(this.sentence ? this.sentence.ears.slice(0, this.wordIndex) : []);
      this._later(function () {
        self._emit(self._final(ears, lips, 0.4));
      }, 300);
      return;
    }
    // Clip mode: pretend to process for about 2.5 s, reporting each step.
    this._later(function () { self._emit({ type: 'progress', step: 'sent' }); }, 150);
    this._later(function () { self._emit({ type: 'progress', step: 'face' }); }, 700);
    this._later(function () { self._emit({ type: 'progress', step: 'listen' }); }, 1300);
    this._later(function () { self._emit({ type: 'progress', step: 'write' }); }, 2000);
    this._later(function () {
      self._nextSentence();
      self._emit(self._final(self.sentence.ears, self.sentence.lips, 2.6));
    }, 2600);
  };

  MockSocket.prototype._final = function (ears, lips, processing) {
    var crops = (this._clipCrops || []).filter(function (c, i, all) { return i % Math.max(1, Math.floor(all.length / 8)) === 0; }).slice(0, 8);
    return {
      type: 'final',
      t: Math.round(this._elapsed() * 10) / 10,
      ears: { text: ears.join(' '), pending: '', model: this.models.ears },
      lips: { text: lips.join(' '), pending: '', model: this.models.lips },
      gate: this.run === 'clip' ? this.gateAll : [],
      noise_db: Math.round(this.snr * 10) / 10,
      crops: this.run === 'clip' ? crops : [],
      processing_s: processing,
      face_found: this.frames ? this.faceFrames / this.frames : 1
    };
  };

  MockSocket.prototype._error = function (code) {
    clearInterval(this._tick);
    this._emit({ type: 'error', code: code, message: 'Pretend error for testing: ' + code });
  };

  window.LipScribeMockSocket = MockSocket;
})();

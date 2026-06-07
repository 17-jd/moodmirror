// MoodMirror dashboard — fully offline, vanilla JS, no frameworks/CDNs.
(function () {
  "use strict";

  // --- Config ---
  var STATE_INTERVAL_MS = 200; // ~5x/sec
  var FEED_INTERVAL_MS = 500;  // ~2x/sec
  var MAX_FEED_LINES = 300;
  var NEAR_BOTTOM_PX = 80;     // auto-scroll threshold
  var GESTURE_FLASH_MS = 750;
  var RESET_REENABLE_MS = 1000;
  var START_REENABLE_MS = 1000;
  var SOUND_FLASH_MS = 900;
  var SITUATION_FLASH_MS = 900;
  var FRESH_MS = 800;
  var SONG_FLASH_MS = 900;
  var AUDIO_HISTORY = 5; // polls to consider for "smooth vs climbing"

  // Map raw song_status -> friendly readout + whether a spinner/bar should run.
  var SONG_STEPS = {
    "":           { text: "Idle", working: false },
    describing:   { text: "📸 Reading your moment…", working: true },
    submitting:   { text: "🎼 Composing with Suno…", working: true },
    generating:   { text: "🎼 Composing with Suno…", working: true },
    downloading:  { text: "⬇ Fetching the track…", working: true },
    playing:      { text: "▶ Playing your song", working: false },
    done:         { text: "✓ Song ready", working: false }
  };

  var SOUND_SOURCES = { idle: 1, gemini: 1, emotion: 1, gesture: 1 };

  var EMOTION_COLORS = {
    happy: "#36d399",
    sad: "#5b8def",
    angry: "#f87272",
    surprise: "#fbbd23",
    fear: "#a78bfa",
    disgust: "#a3e635",
    neutral: "#94a3b8"
  };

  // Simplified 3-class mood -> headline color (the prominent emotion display).
  var MOOD3_COLORS = {
    happy: "#36d399",
    sad: "#5b8def",
    normal: "#94a3b8"
  };

  var FEED_KINDS = { change: 1, hold: 1, observe: 1, gemini: 1, gesture: 1 };

  var PHASES = {
    idle:      { cls: "phase-idle",      text: "Idle" },
    observing: { cls: "phase-observing", text: "Watching you…" },
    directing: { cls: "phase-directing", text: "Asking Gemini for the vibe…" },
    streaming: { cls: "phase-streaming", text: "Streaming • reacting live" }
  };

  // --- Element refs ---
  var el = {
    musicReadout: document.getElementById("musicReadout"),
    musicStyle: document.getElementById("musicStyle"),
    musicStatus: document.getElementById("musicStatus"),
    soundMode: document.getElementById("soundMode"),
    soundModeChip: document.getElementById("soundModeChip"),
    soundModeSource: document.getElementById("soundModeSource"),
    soundModeLabel: document.getElementById("soundModeLabel"),
    soundModeStyle: document.getElementById("soundModeStyle"),
    audioHealth: document.getElementById("audioHealth"),
    startGate: document.getElementById("startGate"),
    startBtn: document.getElementById("startBtn"),
    fpsCapture: document.getElementById("fpsCapture"),
    fpsDetect: document.getElementById("fpsDetect"),
    fpsEmotion: document.getElementById("fpsEmotion"),
    fpsGesture: document.getElementById("fpsGesture"),
    latencyHeadline: document.getElementById("latencyHeadline"),
    latencySub: document.getElementById("latencySub"),
    resetBtn: document.getElementById("resetBtn"),
    phaseBanner: document.getElementById("phaseBanner"),
    phaseIcon: document.getElementById("phaseIcon"),
    phaseText: document.getElementById("phaseText"),
    phaseDetail: document.getElementById("phaseDetail"),
    situationNow: document.getElementById("situationNow"),
    situationNowText: document.getElementById("situationNowText"),
    geminiLive: document.getElementById("geminiLive"),
    facePill: document.getElementById("facePill"),
    emotionDot: document.getElementById("emotionDot"),
    emotionLabel: document.getElementById("emotionLabel"),
    motionTag: document.getElementById("motionTag"),
    smileFill: document.getElementById("smileFill"),
    smileCaption: document.getElementById("smileCaption"),
    poseYaw: document.getElementById("poseYaw"),
    posePitch: document.getElementById("posePitch"),
    poseRoll: document.getElementById("poseRoll"),
    poseGaze: document.getElementById("poseGaze"),
    gestureChips: document.getElementById("gestureChips"),
    directive: document.getElementById("directive"),
    directiveSource: document.getElementById("directiveSource"),
    directivePhilosophy: document.getElementById("directivePhilosophy"),
    directiveObservation: document.getElementById("directiveObservation"),
    directiveStyle: document.getElementById("directiveStyle"),
    feed: document.getElementById("feed"),
    feedEngines: document.getElementById("feedEngines"),
    geminiIntervalSlider: document.getElementById("geminiIntervalSlider"),
    geminiIntervalValue: document.getElementById("geminiIntervalValue"),
    geminiModelSelect: document.getElementById("geminiModelSelect"),
    modeSwitch: document.getElementById("modeSwitch"),
    modeLocalBtn: document.getElementById("modeLocalBtn"),
    modeSunoBtn: document.getElementById("modeSunoBtn"),
    modeLocalName: document.getElementById("modeLocalName"),
    modeSunoNoKey: document.getElementById("modeSunoNoKey"),
    sunoPanel: document.getElementById("sunoPanel"),
    sunoStatus: document.getElementById("sunoStatus"),
    nowPlaying: document.getElementById("nowPlaying"),
    nowPlayingText: document.getElementById("nowPlayingText"),
    stopSongBtn: document.getElementById("stopSongBtn")
  };

  // Map gesture chips by their data-gesture key for fast flashing.
  var chipMap = {};
  (function () {
    var chips = el.gestureChips ? el.gestureChips.querySelectorAll(".gesture-chip") : [];
    for (var i = 0; i < chips.length; i++) {
      chipMap[chips[i].getAttribute("data-gesture")] = chips[i];
    }
  })();

  // --- Single mutable view state ---
  var state = {
    lastSeq: 0,
    lastStyle: null,
    resetting: false,
    starting: false,
    lastGesture: "",
    gestureFlashTimer: null,
    soundFlashTimer: null,
    musicFlashTimer: null,
    lastSoundSource: null,
    lastSoundLabel: null,
    lastSoundStyle: null,
    underrunHistory: [],   // recent audio_underruns readings
    lastSituation: null,
    situationFlashTimer: null,
    intervalDragging: false,   // true while the user is dragging the slider
    intervalInitialized: false, // seed the slider from state exactly once
    modelListKey: null,        // signature of the last-rendered <option> list
    modelPostInFlight: false,  // guard double POST of gemini-model
    mode: null,                // last-rendered mode ("local" | "suno")
    pendingMode: null,         // optimistic mode while a POST is in flight
    switchingMode: false,      // guard double-click on mode buttons
    stoppingSong: false,       // guard double-click on Stop
    lastSongStatus: null,      // flash the panel when song_status changes
    songFlashTimer: null
  };

  // --- Formatting helpers ---
  function fmt1(n) {
    return (typeof n === "number" && isFinite(n)) ? n.toFixed(1) : "0.0";
  }
  function deg(n) {
    return (typeof n === "number" && isFinite(n)) ? Math.round(n) + "°" : "0°";
  }
  function num2(n) {
    return (typeof n === "number" && isFinite(n)) ? n.toFixed(2) : "0.00";
  }
  function clampPct(n) {
    if (typeof n !== "number" || !isFinite(n)) return 0;
    return Math.max(0, Math.min(100, Math.round(n * 100)));
  }
  function isNearBottom() {
    var f = el.feed;
    return (f.scrollHeight - f.scrollTop - f.clientHeight) < NEAR_BOTTOM_PX;
  }

  // Restart a one-shot flash animation cleanly (toggle class with reflow).
  function flash(node, cls, duration, timerKey) {
    if (!node) return;
    node.classList.remove(cls);
    void node.offsetWidth; // reflow to restart animation
    node.classList.add(cls);
    if (state[timerKey]) clearTimeout(state[timerKey]);
    state[timerKey] = setTimeout(function () { node.classList.remove(cls); }, duration);
  }

  function isIdle(s) {
    return (s.phase || "idle") === "idle" || s.started === false;
  }

  // --- Phase banner ---
  function renderPhase(s) {
    var phase = s.phase || "idle";
    var info = PHASES[phase] || PHASES.idle;
    el.phaseBanner.className = "phase-banner " + info.cls;
    el.phaseText.textContent = info.text;

    if (phase === "observing") {
      var rem = (typeof s.observe_remaining_s === "number" && isFinite(s.observe_remaining_s))
        ? Math.max(0, s.observe_remaining_s) : 0;
      el.phaseDetail.textContent = rem.toFixed(1) + "s";
    } else if (phase === "directing") {
      el.phaseDetail.textContent = "thinking…";
    } else if (phase === "streaming") {
      el.phaseDetail.textContent = s.music_ready ? "live" : "starting…";
    } else {
      el.phaseDetail.textContent = "";
    }
  }

  // --- Start gate (idle vs running) ---
  function renderGate(s) {
    if (isIdle(s)) {
      el.startGate.classList.remove("hidden");
      // reset button is meaningless before a session exists
      el.resetBtn.classList.add("hidden");
      el.resetBtn.disabled = true;
    } else {
      el.startGate.classList.add("hidden");
      el.resetBtn.classList.remove("hidden");
      if (!state.resetting) el.resetBtn.disabled = false;
    }
  }

  // --- Live situation (NOW) + Gemini-live badge ---
  function renderSituation(s) {
    if (!el.situationNow) return;
    var situation = (typeof s.situation === "string") ? s.situation.trim() : "";

    if (situation) {
      el.situationNow.classList.remove("empty");
      el.situationNowText.textContent = "Gemini sees: " + situation;
    } else {
      el.situationNow.classList.add("empty");
      el.situationNowText.textContent = "reading the room…";
    }

    // Flash when Gemini re-reads the scene (text changed to a new non-empty value).
    if (state.lastSituation !== null && situation && situation !== state.lastSituation) {
      flash(el.situationNow, "flash", SITUATION_FLASH_MS, "situationFlashTimer");
    }
    state.lastSituation = situation;

    // "live" badge only while streaming (screenshots sent every ~3s).
    if (el.geminiLive) {
      el.geminiLive.hidden = ((s.phase || "idle") !== "streaming");
    }
  }

  // --- Webcam overlays (face / emotion / motion / mood meter) ---
  function renderWebcam(s) {
    // Face presence
    if (s.face_present) {
      el.facePill.textContent = "face";
      el.facePill.className = "overlay-pill present";
    } else {
      el.facePill.textContent = "no face";
      el.facePill.className = "overlay-pill absent";
    }

    // Emotion dot + label — headline is the simplified 3-class mood3.
    // mood3 is one of happy/sad/normal; fall back to the raw 7-class emotion.
    var mood3 = (typeof s.mood3 === "string") ? s.mood3.trim() : "";
    var headline = mood3 || (typeof s.emotion === "string" ? s.emotion : "");
    if (s.face_present && headline) {
      el.emotionLabel.textContent = headline;
      var c = MOOD3_COLORS[headline] || EMOTION_COLORS[headline] || EMOTION_COLORS.neutral;
      el.emotionDot.style.background = c;
      el.emotionDot.style.color = c; // drives the glow (currentColor)
    } else {
      el.emotionLabel.textContent = "—";
      el.emotionDot.style.background = "#5b6378";
      el.emotionDot.style.color = "#5b6378";
    }

    // Moving / still
    if (s.moving) {
      el.motionTag.textContent = "MOVING";
      el.motionTag.className = "motion-tag moving";
    } else {
      el.motionTag.textContent = "STILL";
      el.motionTag.className = "motion-tag";
    }

    // Smile / mood meter
    var pct = clampPct(s.smile_score);
    el.smileFill.style.width = pct + "%";
    el.smileCaption.textContent = "mood " + pct + "%" + (s.smiling ? " · smiling" : "");
  }

  // --- Gesture pose + chip flash ---
  function renderGesture(s) {
    var g = s.gesture || {};
    el.poseYaw.textContent = deg(g.yaw);
    el.posePitch.textContent = deg(g.pitch);
    el.poseRoll.textContent = deg(g.roll);
    el.poseGaze.textContent = num2(g.gaze);

    var last = g.last || "";
    if (last && last !== state.lastGesture) {
      flashChip(last);
    }
    state.lastGesture = last;
  }

  function flashChip(name) {
    var chip = chipMap[name];
    if (!chip) return;
    chip.classList.remove("flash");
    void chip.offsetWidth; // restart transition
    chip.classList.add("flash");
    if (state.gestureFlashTimer) clearTimeout(state.gestureFlashTimer);
    state.gestureFlashTimer = setTimeout(function () {
      chip.classList.remove("flash");
    }, GESTURE_FLASH_MS);
  }

  // --- THE VIBE (opening directive) ---
  function renderVibe(d) {
    var has = d && (d.philosophy || d.observation || d.style);
    if (!has) {
      el.directive.classList.add("empty");
      el.directiveSource.textContent = "—";
      el.directiveSource.className = "directive-source";
      el.directivePhilosophy.textContent = "Waiting for the opening read…";
      el.directiveObservation.textContent = "";
      el.directiveStyle.textContent = "";
      return;
    }
    el.directive.classList.remove("empty");
    el.directivePhilosophy.textContent = d.philosophy || "—";
    el.directiveObservation.textContent = d.observation || "";
    el.directiveStyle.textContent = d.style ? ("style · " + d.style) : "";

    var src = d.source === "gemini" ? "gemini" : (d.source === "fallback" ? "fallback" : "");
    el.directiveSource.className = "directive-source" + (src ? " " + src : "");
    el.directiveSource.textContent = src
      ? (src === "gemini" ? "Gemini" : "fallback")
      : "—";
  }

  // --- Active sound-mode chip ---
  function renderSoundMode(s) {
    var sm = s.sound_mode || {};
    var source = SOUND_SOURCES[sm.source] ? sm.source : "idle";
    var label = sm.label || "";
    var style = sm.style || "";

    el.soundModeChip.className = "sound-mode-chip " + source;
    el.soundModeSource.textContent = source.toUpperCase();
    el.soundModeLabel.textContent = label || "—";
    el.soundModeStyle.textContent = style || "—";

    // Flash when a gesture just drove the sound, or when the style changed.
    var changed = false;
    if (state.lastSoundSource !== null) {
      if (source === "gesture" && (source !== state.lastSoundSource || label !== state.lastSoundLabel)) {
        changed = true;
      }
      if (style && style !== state.lastSoundStyle) {
        changed = true;
      }
    }
    if (changed && source !== "idle") {
      flash(el.soundMode, "flash", SOUND_FLASH_MS, "soundFlashTimer");
    }
    state.lastSoundSource = source;
    state.lastSoundLabel = label;
    if (style) state.lastSoundStyle = style;
  }

  // --- Audio health (underruns) ---
  function renderAudioHealth(s) {
    if (typeof s.audio_underruns !== "number" || !isFinite(s.audio_underruns)) {
      el.audioHealth.textContent = "audio: —";
      el.audioHealth.className = "audio-health mono";
      return;
    }
    var n = s.audio_underruns;
    state.underrunHistory.push(n);
    if (state.underrunHistory.length > AUDIO_HISTORY) state.underrunHistory.shift();

    // "smooth" = value hasn't increased across the recent window.
    var climbing = state.underrunHistory[state.underrunHistory.length - 1] > state.underrunHistory[0];

    if (climbing) {
      el.audioHealth.textContent = "audio: " + n + " drop" + (n === 1 ? "" : "s");
      el.audioHealth.className = "audio-health mono drops";
    } else {
      el.audioHealth.textContent = n === 0
        ? "audio: smooth ✓"
        : "audio: stable (" + n + ") ✓";
      el.audioHealth.className = "audio-health mono smooth";
    }
  }

  // --- Music readout (header) + meters ---
  function renderMusic(s) {
    var style = s.current_style || "—";
    el.musicStyle.textContent = style;
    if (state.lastStyle !== null && style !== state.lastStyle && style !== "—") {
      flash(el.musicReadout, "flash", 900, "musicFlashTimer");
    }
    if (style !== "—") state.lastStyle = style;

    var idle = isIdle(s);
    var live = !idle && !!s.music_ready;
    if (idle) {
      el.musicStatus.textContent = "silent — press Start";
    } else {
      el.musicStatus.textContent = live ? "live" : "warming up music…";
    }
    el.musicReadout.classList.toggle("playing", live);
  }

  function renderMeters(s) {
    // FPS
    var fps = s.fps || {};
    el.fpsCapture.textContent = fmt1(fps.capture);
    el.fpsDetect.textContent = fmt1(fps.detect);
    el.fpsEmotion.textContent = fmt1(fps.emotion);
    el.fpsGesture.textContent = fmt1(fps.gesture);

    // Latency — sub-second, shown proudly
    var lat = s.latency || {};
    if (lat.samples) {
      var headlineVal = (lat.last_s != null) ? lat.last_s : lat.avg_s;
      el.latencyHeadline.textContent = "Music reacts in ~" + fmt1(headlineVal) + "s";
      var subParts = [];
      if (lat.avg_s != null) subParts.push("avg " + fmt1(lat.avg_s) + "s");
      subParts.push(lat.samples + " change" + (lat.samples === 1 ? "" : "s"));
      el.latencySub.textContent = subParts.join(" · ");
    } else {
      el.latencyHeadline.textContent = "Music reacts in —";
      el.latencySub.textContent = "measuring…";
    }

    // Engines
    var eng = s.engine || {};
    if (eng.vision || eng.emotion || eng.gesture || eng.gemini) {
      el.feedEngines.textContent =
        (eng.vision || "?") + " / " + (eng.emotion || "?") +
        " / " + (eng.gesture || "?") + " / " + (eng.gemini || "?");
    }
  }

  // --- Gemini screenshot frequency slider ---
  function setIntervalLabel(seconds) {
    if (el.geminiIntervalValue) el.geminiIntervalValue.textContent = seconds + "s";
  }

  function renderGeminiInterval(s) {
    if (!el.geminiIntervalSlider) return;
    var iv = s.gemini_interval_s;
    if (typeof iv !== "number" || !isFinite(iv)) return;
    // Clamp to the slider's range so polling never writes an out-of-range value.
    var v = Math.max(1, Math.min(15, Math.round(iv)));
    // Seed once on first load; afterwards only mirror server state when the user
    // is NOT dragging, so polling doesn't fight the slider thumb.
    if (!state.intervalInitialized || !state.intervalDragging) {
      el.geminiIntervalSlider.value = String(v);
      setIntervalLabel(v);
    }
    state.intervalInitialized = true;
  }

  function postGeminiInterval(seconds) {
    try {
      fetch("/api/gemini-interval", {
        method: "POST",
        cache: "no-store",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ seconds: seconds })
      })
        .then(function (r) { return r.ok ? r.json() : null; })
        .catch(function () { /* swallow; state poll reflects truth */ });
    } catch (e) { /* never throw from a UI handler */ }
  }

  if (el.geminiIntervalSlider) {
    el.geminiIntervalSlider.addEventListener("input", function () {
      state.intervalDragging = true;
      setIntervalLabel(this.value);
    });
    el.geminiIntervalSlider.addEventListener("change", function () {
      state.intervalDragging = false;
      var v = parseInt(this.value, 10);
      if (isFinite(v)) postGeminiInterval(v);
    });
  }

  // --- Gemini model select ---
  // True while the user has the dropdown focused/open, so polling never
  // overwrites their selection mid-interaction.
  function modelSelectBusy() {
    return el.geminiModelSelect &&
      document.activeElement === el.geminiModelSelect;
  }

  // Build the option list from the available models, always including the
  // currently-selected model even if it's not in the fetched list.
  function buildModelOptions(models, current) {
    var list = [];
    var seen = {};
    for (var i = 0; i < models.length; i++) {
      var id = models[i];
      if (typeof id === "string" && id && !seen[id]) {
        seen[id] = 1;
        list.push(id);
      }
    }
    if (current && !seen[current]) list.unshift(current);
    return list;
  }

  function renderGeminiModel(s) {
    var sel = el.geminiModelSelect;
    if (!sel) return;

    var models = Array.isArray(s.gemini_models) ? s.gemini_models : [];
    var current = (typeof s.gemini_model === "string") ? s.gemini_model : "";

    // Empty list (briefly, at startup): show a single disabled placeholder.
    if (!models.length && !current) {
      if (state.modelListKey !== "__loading__") {
        sel.innerHTML = "";
        var opt = document.createElement("option");
        opt.value = "";
        opt.disabled = true;
        opt.selected = true;
        opt.textContent = "loading models…";
        sel.appendChild(opt);
        sel.disabled = true;
        state.modelListKey = "__loading__";
      }
      return;
    }

    var list = buildModelOptions(models, current);
    var key = list.join("");

    // Only rebuild the <option> list when the set of models actually changes —
    // rebuilding every 200ms would break an open dropdown.
    if (key !== state.modelListKey) {
      sel.innerHTML = "";
      for (var i = 0; i < list.length; i++) {
        var o = document.createElement("option");
        o.value = list[i];
        o.textContent = list[i];
        sel.appendChild(o);
      }
      sel.disabled = false;
      state.modelListKey = key;
    }

    // Mirror the active model — but don't fight the user while they're in it.
    if (current && !modelSelectBusy() && sel.value !== current) {
      sel.value = current;
    }
  }

  function postGeminiModel(model) {
    if (!model) return;
    if (state.modelPostInFlight) return; // guard a double-fire
    state.modelPostInFlight = true;
    try {
      fetch("/api/gemini-model", {
        method: "POST",
        cache: "no-store",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: model })
      })
        .then(function (r) { return r.ok ? r.json() : null; })
        .catch(function () { /* swallow; state poll reflects truth */ })
        .finally(function () { state.modelPostInFlight = false; });
    } catch (e) {
      state.modelPostInFlight = false; // never throw from a UI handler
    }
  }

  if (el.geminiModelSelect) {
    el.geminiModelSelect.addEventListener("change", function () {
      var model = this.value;
      if (model) postGeminiModel(model); // optimistic: keep the selection
    });
  }

  // --- Mode switcher (Local vs Suno) ---
  function renderMode(s) {
    if (!el.modeSwitch) return;

    // Show the switcher only once a session has started.
    el.modeSwitch.classList.toggle("hidden", isIdle(s));

    // Local model name label.
    if (el.modeLocalName && typeof s.local_model_name === "string" && s.local_model_name) {
      el.modeLocalName.textContent = s.local_model_name;
    }

    // Suno availability gating.
    var sunoAvailable = !!s.suno_available;
    if (el.modeSunoBtn) {
      el.modeSunoBtn.disabled = !sunoAvailable || state.switchingMode;
    }
    if (el.modeSunoNoKey) el.modeSunoNoKey.hidden = sunoAvailable;

    // Optimistic highlight: prefer pending mode, else server truth.
    var serverMode = (s.mode === "suno") ? "suno" : "local";
    var shown = state.pendingMode || serverMode;
    // Once the server agrees with our optimistic choice, clear the pending flag.
    if (state.pendingMode && serverMode === state.pendingMode) {
      state.pendingMode = null;
      state.switchingMode = false;
    }

    setActiveMode(shown);
    state.mode = shown;
  }

  function setActiveMode(mode) {
    var localActive = mode === "local";
    if (el.modeLocalBtn) {
      el.modeLocalBtn.classList.toggle("active", localActive);
      el.modeLocalBtn.setAttribute("aria-pressed", localActive ? "true" : "false");
    }
    if (el.modeSunoBtn) {
      el.modeSunoBtn.classList.toggle("active", !localActive);
      el.modeSunoBtn.setAttribute("aria-pressed", !localActive ? "true" : "false");
    }
    // Dim local-only widgets + toggle the Suno panel.
    document.body.classList.toggle("suno-active", !localActive);
    if (el.sunoPanel) el.sunoPanel.classList.toggle("hidden", localActive);
  }

  // --- Suno progress panel ---
  function renderSuno(s) {
    if (!el.sunoPanel) return;

    var mode = state.pendingMode || (s.mode === "suno" ? "suno" : "local");
    if (mode !== "suno") {
      el.sunoPanel.classList.add("hidden");
      return;
    }
    el.sunoPanel.classList.remove("hidden");

    var raw = (typeof s.song_status === "string") ? s.song_status : "";
    var playing = !!s.playing_song;
    var track = (typeof s.current_track === "string" && s.current_track) ? s.current_track : "";
    var isError = raw.indexOf("error") === 0;

    var label, working;
    if (isError) {
      label = "⚠ " + raw;
      working = false;
    } else if (playing) {
      label = "▶ Playing your song";
      working = false;
    } else {
      var step = SONG_STEPS[raw] || { text: raw || "Idle", working: false };
      label = step.text;
      working = step.working;
    }

    el.sunoStatus.textContent = label;
    el.sunoPanel.classList.toggle("error", isError);
    el.sunoPanel.classList.toggle("working", working);

    // Now-playing row + Stop button.
    var showNow = playing;
    el.nowPlaying.classList.toggle("hidden", !showNow);
    if (showNow) {
      el.nowPlayingText.textContent = "▶ Now playing: " + (track || "your song");
    }

    // Flash the panel when the pipeline step changes.
    if (state.lastSongStatus !== null && raw !== state.lastSongStatus) {
      flash(el.sunoPanel, "flash", SONG_FLASH_MS, "songFlashTimer");
    }
    state.lastSongStatus = raw;
  }

  // --- Top-level state render (update DOM in place, no full rebuild) ---
  function renderState(s) {
    renderMode(s);
    renderSuno(s);
    renderPhase(s);
    renderGate(s);
    renderSituation(s);
    renderWebcam(s);
    renderGesture(s);
    renderVibe(s.directive);
    renderSoundMode(s);
    renderAudioHealth(s);
    renderMusic(s);
    renderMeters(s);
    renderGeminiInterval(s);
    renderGeminiModel(s);
  }

  // --- Feed rendering (append-only) ---
  function appendEvent(ev) {
    var kind = FEED_KINDS[ev.kind] ? ev.kind : "observe";

    var line = document.createElement("div");
    line.className = "line " + kind + " fresh";

    var clock = document.createElement("span");
    clock.className = "clock";
    clock.textContent = ev.clock || "";

    var text = document.createElement("span");
    text.className = "text";
    text.textContent = ev.text || "";

    if ((kind === "change" || kind === "gemini") && ev.style) {
      var chip = document.createElement("span");
      chip.className = "style-chip";
      chip.textContent = ev.style;
      text.appendChild(chip);
    }

    line.appendChild(clock);
    line.appendChild(text);
    el.feed.appendChild(line);

    // strip the fresh class after the animation so re-layout doesn't replay it
    setTimeout(function () { line.classList.remove("fresh"); }, FRESH_MS);
  }

  function capFeed() {
    while (el.feed.childElementCount > MAX_FEED_LINES) {
      el.feed.removeChild(el.feed.firstElementChild);
    }
  }

  function renderFeed(data) {
    var events = data.events || [];
    if (!events.length) return;
    var stick = isNearBottom();
    for (var i = 0; i < events.length; i++) {
      appendEvent(events[i]);
    }
    capFeed();
    if (stick) {
      el.feed.scrollTop = el.feed.scrollHeight;
    }
  }

  // --- Reset (re-observe for a new person) ---
  function doReset() {
    if (state.resetting) return;
    state.resetting = true;
    el.resetBtn.disabled = true;
    el.resetBtn.textContent = "Re-watching…";

    fetch("/api/reset", { method: "POST", cache: "no-store" })
      .then(function (r) { return r.ok ? r.json() : null; })
      .catch(function () { /* swallow; phase banner reflects truth */ })
      .finally(function () {
        setTimeout(function () {
          state.resetting = false;
          el.resetBtn.disabled = false;
          el.resetBtn.innerHTML =
            '<span aria-hidden="true">&#8634;</span> Start Again <span class="reset-sub">new person</span>';
        }, RESET_REENABLE_MS);
      });
  }

  if (el.resetBtn) el.resetBtn.addEventListener("click", doReset);

  // --- Start (begin the session) ---
  function doStart() {
    if (state.starting) return;
    state.starting = true;
    el.startBtn.disabled = true;
    el.startBtn.textContent = "Starting…";

    fetch("/api/start", { method: "POST", cache: "no-store" })
      .then(function (r) { return r.ok ? r.json() : null; })
      .catch(function () { /* swallow; phase banner reflects truth */ })
      .finally(function () {
        setTimeout(function () {
          state.starting = false;
          el.startBtn.disabled = false;
          el.startBtn.innerHTML = "&#9654;&nbsp; Start";
        }, START_REENABLE_MS);
      });
  }

  if (el.startBtn) el.startBtn.addEventListener("click", doStart);

  // --- Mode switch (Local vs Suno) ---
  function setMode(mode) {
    if (state.switchingMode) return;        // guard double-click
    if (mode !== "local" && mode !== "suno") return;
    if (state.mode === mode && !state.pendingMode) return; // no-op

    // Optimistic highlight; the state poll confirms the truth.
    state.switchingMode = true;
    state.pendingMode = mode;
    setActiveMode(mode);

    try {
      fetch("/api/mode", {
        method: "POST",
        cache: "no-store",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode: mode })
      })
        .then(function (r) { return r.ok ? r.json() : null; })
        .catch(function () { /* swallow; state poll reflects truth */ })
        .finally(function () {
          // Release the guard; renderMode clears pendingMode once server agrees.
          state.switchingMode = false;
        });
    } catch (e) {
      state.switchingMode = false; // never get stuck
    }
  }

  if (el.modeLocalBtn) {
    el.modeLocalBtn.addEventListener("click", function () { setMode("local"); });
  }
  if (el.modeSunoBtn) {
    el.modeSunoBtn.addEventListener("click", function () {
      if (el.modeSunoBtn.disabled) return;
      setMode("suno");
    });
  }

  // --- Stop the current Suno song ---
  function stopSong() {
    if (state.stoppingSong) return;
    state.stoppingSong = true;
    el.stopSongBtn.disabled = true;

    try {
      fetch("/api/stop-song", { method: "POST", cache: "no-store" })
        .then(function (r) { return r.ok ? r.json() : null; })
        .catch(function () { /* swallow; state poll reflects truth */ })
        .finally(function () {
          state.stoppingSong = false;
          if (el.stopSongBtn) el.stopSongBtn.disabled = false;
        });
    } catch (e) {
      state.stoppingSong = false;
      if (el.stopSongBtn) el.stopSongBtn.disabled = false;
    }
  }

  if (el.stopSongBtn) el.stopSongBtn.addEventListener("click", stopSong);

  // --- Polling loops (rescheduled in finally so errors never stop them) ---
  function pollState() {
    fetch("/api/state", { cache: "no-store" })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (s) { if (s) renderState(s); })
      .catch(function () { /* keep polling silently */ })
      .finally(function () { setTimeout(pollState, STATE_INTERVAL_MS); });
  }

  function pollFeed() {
    fetch("/api/feed?since=" + state.lastSeq, { cache: "no-store" })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (data) {
        if (data) {
          renderFeed(data);
          if (typeof data.seq === "number") state.lastSeq = data.seq;
        }
      })
      .catch(function () { /* keep polling silently */ })
      .finally(function () { setTimeout(pollFeed, FEED_INTERVAL_MS); });
  }

  // --- Boot ---
  pollState();
  pollFeed();
})();

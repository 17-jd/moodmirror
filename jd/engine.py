"""JD orchestrator: webcam → vision → emotion → gesture → narrator → music.

The :class:`JDEngine` wires the pipeline together and runs it on a small set of
background daemon threads. The session flows through phases:

  * idle      — constructed but not yet started.
  * observing — watch the subject for ``OBSERVE_SECONDS`` capturing
                ``OBSERVE_PHOTOS`` frames (the only place the dashboard shows a
                countdown).
  * directing — hand those frames to Gemini for the OPENING directive (the ONLY
                network call; ~1-3s). Falls back to ``STARTER_STYLE`` offline.
  * streaming — the steady state: continuous music with local emotion + gesture
                nudges driving HOLD/CHANGE/gesture narration.

Reset re-enters ``observing`` WITHOUT stopping the music — the old stream keeps
playing right up until the new directive arrives, then the style swaps.

Threads (all daemon):
  * vision  — captures frames as fast as the camera allows (CAPTURE fps + MJPEG
              preview) and, a few times per second, runs the Haar face/motion
              detector (DETECT fps) and FUSES the latest DeepFace emotion.
  * emotion — copies the latest frame a couple of times per second → DeepFace.
  * gesture — copies the latest frame ~10x/sec → MediaPipe head-pose subprocess,
              firing micro-gesture music nudges (streaming phase only).
  * narrate — every tick, asks the narrator to HOLD/CHANGE (streaming phase only).

Everything shared between threads is guarded by a single lock; readings and
events are immutable and fused with :func:`dataclasses.replace`. The lock is
never held across the Gemini network call or across ``music.set_style``.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import replace
from datetime import datetime

import numpy as np

from . import config
from .emotion_proc import EmotionWorker
from .gemini_director import make_director
from .suno import make_suno
from .gesture_proc import GestureWorker
from .models import GeminiDirective, GestureReading, MoodReading, NarrationEvent
from .music import MusicEngine
from .narrator import Narrator
from .vision import MoodDetector

logger = logging.getLogger(__name__)

# EMA smoothing factor for the per-stage fps meters. Higher = more responsive,
# lower = steadier. 0.3 tracks real changes while damping per-frame jitter.
_EMA_ALPHA = 0.3


class _Meter:
    """Thread-safe EMA frames-per-second meter for one pipeline stage.

    ``tick`` is called once per processed unit of work. It derives an
    instantaneous fps from the gap since the previous tick and smooths it with an
    exponential moving average. ``last_ms`` records the wall-clock duration of the
    most recent timed call; it is left at 0 for stages that do not pass one in.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_ts: float | None = None
        self._fps: float = 0.0
        self._last_ms: float = 0.0

    def tick(self, last_ms: float = 0.0) -> None:
        """Record one processed frame and (optionally) its duration in ms."""
        now = time.monotonic()
        with self._lock:
            prev = self._last_ts
            self._last_ts = now
            if last_ms:
                self._last_ms = last_ms
            if prev is None:
                return
            dt = now - prev
            if dt <= 0:
                return
            inst = 1.0 / dt
            self._fps = inst if self._fps == 0.0 else (
                _EMA_ALPHA * inst + (1.0 - _EMA_ALPHA) * self._fps
            )

    def snapshot(self) -> tuple[float, float]:
        """Return ``(fps, last_ms)`` atomically."""
        with self._lock:
            return self._fps, self._last_ms


class JDEngine:
    """Webcam-mood → directed live-music orchestrator with a narration feed.

    The web server (and tests) depend on this class's public interface exactly:
    :meth:`start`, :meth:`shutdown`, :meth:`reset`, :meth:`get_state`,
    :meth:`feed_since`, and :meth:`latest_frame_jpeg`.

    Construction is cheap and side-effect-free with respect to hardware: it does
    NOT open the webcam or load the music model — only :meth:`start` does.
    """

    def __init__(self, webcam_index: int | None = None) -> None:
        self._webcam_index = (
            config.WEBCAM_INDEX if webcam_index is None else webcam_index
        )
        self._lock = threading.Lock()
        self._stop = threading.Event()

        # Pipeline components.
        self._detector = MoodDetector()
        self._emotion_worker = EmotionWorker()
        self._emotion_available = self._emotion_worker.available
        self._gesture_worker = GestureWorker()
        self._gesture_available = self._gesture_worker.available
        self._narrator = Narrator()
        self._director = make_director()  # may be None (offline → fallback)

        session_dir = config.SESSIONS_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")
        session_dir.mkdir(parents=True, exist_ok=True)
        self._session_dir = session_dir
        self._music = MusicEngine(
            session_dir=session_dir,
            on_change_latency=self._record_latency,
        )

        # Phase machine.
        self._phase: str = "idle"
        self._observe_deadline: float = 0.0
        self._observing = threading.Event()  # guard: an observe→direct is running
        self._directive: GeminiDirective | None = None

        # Shared, lock-guarded state.
        self._recent: deque[MoodReading] = deque(maxlen=config.RECENT_WINDOW)
        self._latest: MoodReading = MoodReading.absent()
        self._gesture: GestureReading = GestureReading.absent()
        self._last_gesture: str = ""  # last fired gesture label
        self._last_gesture_ts: float = 0.0  # monotonic ts of the last fired gesture

        # Active "sound mode": what last drove the music. source ∈
        # {"idle","gemini","emotion","gesture"}; label is the mood / gesture /
        # directive style that caused it. Idle until the user starts a session.
        self._sound_source: str = "idle"
        self._sound_label: str = ""
        self._latest_frame: np.ndarray | None = None
        self._df_emotion: str = "neutral"  # latest DeepFace emotion (emotion thread)
        self._mood3: str = "normal"  # simplified 3-class mood (happy/sad/normal)
        self._mood3_applied: str = ""  # last 3-class mood whose music we set
        self._df_happy: float = 0.0  # latest DeepFace happy probability
        self._situation: str = ""  # latest Gemini situational read (continuous loop)
        # User-adjustable cadence for the continuous Gemini director loop. Seeded
        # from config; live-tunable from the dashboard via set_gemini_interval().
        self._gemini_interval: float = float(config.GEMINI_INTERVAL_S)

        # Live-selectable Gemini model + cached vision-model list (fetched on
        # start, off-thread). The director starts on the configured model.
        self._gemini_model: str = config.GEMINI_MODEL
        self._gemini_models: list[str] = []
        if self._director is not None:
            self._director.set_model(self._gemini_model)

        # Generation MODE: "local" (live MRT2 stream) vs "suno" (full vocal song).
        # In "suno" mode the local continuous-Gemini + emotion/gesture style nudges
        # PAUSE so they don't fight the Suno song; gestures still narrate.
        self._suno = make_suno()  # may be None (no key → song mode unavailable)
        self._mode: str = config.MODE_DEFAULT  # "local" | "suno"
        # One of: ""/describing/submitting/generating/downloading/playing/done/error:…
        self._song_status: str = ""
        self._song_thread: threading.Thread | None = None
        self._song_inflight = threading.Event()  # guard: only one song at a time

        # Narration feed ring buffer: (seq, NarrationEvent) tuples.
        self._feed: deque[tuple[int, NarrationEvent]] = deque(maxlen=config.FEED_MAXLEN)
        self._feed_seq = 0

        # Music-change latency samples (request → audible swap), in seconds.
        self._latencies: list[float] = []

        # Per-stage fps meters.
        self._cap_meter = _Meter()
        self._det_meter = _Meter()
        self._emo_meter = _Meter()
        self._ges_meter = _Meter()

        self._threads: list[threading.Thread] = []

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> None:
        """Start music + workers and spawn the pipeline threads — but stay SILENT.

        This brings the camera, emotion/gesture subprocesses and the four daemon
        threads online, yet the phase stays ``"idle"`` and the music NEVER gets a
        style (we never call ``music.set_style`` here). No audio plays until the
        user clicks "Start" on the dashboard → :meth:`begin_session`.

        Back-compat: only when ``config.AUTOSTART`` is True do we auto-kick the
        observe→direct flow (which is what eventually starts the audio).

        Returns immediately.
        """
        self._music.start()
        if self._emotion_available:
            self._emotion_worker.start()
        if self._gesture_available:
            self._gesture_available = self._gesture_worker.start()

        loops = (
            (self._vision_loop, "vision"),
            (self._emotion_loop, "emotion"),
            (self._gesture_loop, "gesture"),
            (self._narrate_loop, "narrate"),
            (self._director_loop, "director"),
        )
        for target, name in loops:
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)

        # Fetch the vision-model dropdown list ONCE, off-thread (network call —
        # must not block start). If there's no director we leave the list empty.
        if self._director is not None:
            mt = threading.Thread(
                target=self._fetch_gemini_models, name="gemini-models", daemon=True
            )
            mt.start()
            self._threads.append(mt)

        # Manual gating: phase stays "idle" and the music stays silent. Only with
        # AUTOSTART do we kick the opening observe→direct flow automatically.
        if config.AUTOSTART:
            self.begin_session()

        logger.info(
            "JDEngine started (vision=%s, emotion=%s, gesture=%s, gemini=%s, "
            "autostart=%s).",
            self._detector.backend,
            "deepface" if self._emotion_available else "none",
            "mediapipe" if self._gesture_available else "none",
            "on" if self._director is not None else "off",
            config.AUTOSTART,
        )

    def begin_session(self) -> None:
        """Kick off the opening observe→direct flow in the background (idempotent).

        This is what the dashboard "Start" button calls. The music only begins
        once Gemini returns and the flow calls ``music.set_style`` on the
        directive. Guarded by ``self._observing`` so repeated clicks are no-ops.
        """
        if self._observing.is_set():
            logger.info("begin_session() ignored — already observing.")
            return
        threading.Thread(
            target=self._observe_and_direct,
            kwargs={"reason": "start"},
            name="observe",
            daemon=True,
        ).start()

    def shutdown(self) -> None:
        """Signal all threads to stop, then tear down every component."""
        self._stop.set()
        for t in self._threads:  # let loops exit before closing the detector
            t.join(timeout=5.0)
        song_thread = self._song_thread
        if song_thread is not None:  # let any in-flight Suno song thread wind down
            song_thread.join(timeout=5.0)
        self._music.stop()
        self._emotion_worker.stop()
        self._gesture_worker.stop()
        self._detector.close()
        logger.info("JDEngine shut down.")

    def reset(self) -> None:
        """Re-run the observe→direct flow for a (possibly) new person.

        The music KEEPS PLAYING throughout: only the final step swaps the style.
        No-op (with a log line) if an observe→direct flow is already running, so
        it is safe to call at any time.
        """
        if self._observing.is_set():
            logger.info("reset() ignored — already observing.")
            return
        threading.Thread(
            target=self._observe_and_direct,
            kwargs={"reason": "reset"},
            name="observe-reset",
            daemon=True,
        ).start()

    # -- live tunables (called from the web server) -------------------------
    def set_gemini_interval(self, seconds: float) -> float:
        """Set the continuous Gemini director cadence, clamped to [1.0, 30.0]s.

        The change takes effect on the next director-loop tick (the loop re-reads
        this value each iteration). Returns the stored, clamped value.
        """
        clamped = max(1.0, min(30.0, float(seconds)))
        with self._lock:
            self._gemini_interval = clamped
        return clamped

    def _fetch_gemini_models(self) -> None:
        """Fetch the vision-model list ONCE (network) and cache it under the lock.

        Runs on a short daemon thread from :meth:`start` so the network call
        never blocks startup. Ensures the currently-selected model is present
        in the list (inserted first if missing). Never raises.
        """
        if self._director is None:
            return
        try:
            models = self._director.list_vision_models()  # NOT under the lock
        except Exception:  # noqa: BLE001 — list_vision_models never raises, but be safe
            logger.debug("vision-model fetch failed", exc_info=True)
            return
        with self._lock:
            current = self._gemini_model
            if current and current not in models:
                models = [current, *models]
            self._gemini_models = list(models)

    def set_gemini_model(self, model: str) -> str:
        """Switch the Gemini model used by all director calls (live).

        Accepts any non-empty string (typically from the dashboard dropdown, but
        not required to be in the fetched list so the user is never blocked). The
        director's ``set_model`` is called OUTSIDE the lock. Returns the stored
        model (unchanged if ``model`` is empty/falsy).
        """
        if model:
            with self._lock:
                self._gemini_model = model
            director = self._director  # local ref; set_model is cheap + lock-free
            if director is not None:
                director.set_model(model)
        with self._lock:
            return self._gemini_model

    # -- generation mode (local stream vs Suno full song) -------------------
    def set_mode(self, mode: str) -> str:
        """Switch the generation mode between "local" and "suno".

        Unknown values are ignored (the current mode is kept). Returns the
        stored mode.

        * "suno": if no Suno client is configured the mode still reflects
          "suno" but ``song_status`` is set to an error and nothing is kicked.
          Otherwise a one-shot :meth:`_make_song` is launched on a background
          daemon thread (guarded so two never overlap).
        * "local": any playing Suno song is stopped (so the live MRT2 stream
          resumes) and ``song_status`` is cleared.
        """
        normalized = str(mode).strip().lower()
        if normalized not in ("local", "suno"):
            with self._lock:
                return self._mode  # ignore unknown values; keep current

        with self._lock:
            self._mode = normalized

        if normalized == "suno":
            if self._suno is None:
                self._set_song_status("error: no Suno API key")
                return normalized
            # Kick a one-shot song generation (guarded against overlap).
            if not self._song_inflight.is_set():
                t = threading.Thread(
                    target=self._make_song, name="suno-song", daemon=True
                )
                self._song_thread = t
                t.start()
        else:  # "local": stop any Suno song so the live stream resumes.
            self._music.skip_external()
            self._set_song_status("")

        return normalized

    def stop_song(self) -> None:
        """Stop the current Suno song (if any) and clear the song status."""
        self._music.skip_external()
        self._set_song_status("")

    def _set_song_status(self, status: str) -> None:
        """Store the current Suno song status under the lock (thread-safe)."""
        with self._lock:
            self._song_status = status

    def _make_song(self) -> None:
        """Snapshot → Gemini brief → Suno full song → play it (daemon thread).

        Guarded by ``self._song_inflight`` so two generations never overlap.
        Only proceeds while the mode is still "suno", a Suno client exists, and
        the engine has begun streaming. The Gemini call, the Suno network flow,
        and the music playback are all made WITHOUT holding ``self._lock``.
        Progress is surfaced to the dashboard via ``song_status`` and the feed.
        """
        if self._song_inflight.is_set():
            return
        self._song_inflight.set()
        try:
            with self._lock:
                mode = self._mode
                phase = self._phase
            # Only generate a song in suno mode, with a client, once streaming.
            if mode != "suno" or self._suno is None:
                return
            if phase not in ("streaming", "directing", "observing"):
                return

            self._set_song_status("describing")

            # Grab the latest webcam frame (thread-safe copy).
            with self._lock:
                frame = (
                    None if self._latest_frame is None else self._latest_frame.copy()
                )
            if frame is None:
                self._set_song_status("error: no camera frame")
                return

            # Gemini song brief (NOT under the lock); director may be None.
            if self._director is not None:
                style, lyrics = self._director.describe_for_song([frame])
            else:
                style, lyrics = (
                    "warm uplifting indie pop, gentle vocals, acoustic, 90bpm",
                    "",
                )

            title = f"MoodMirror — {datetime.now().strftime('%H:%M:%S')}"
            try:
                path = self._suno.generate(
                    lyrics=lyrics or None,
                    style=style,
                    title=title,
                    dest_dir=self._session_dir,
                    on_status=lambda s: self._on_song_status(s),
                )
            except Exception as exc:  # noqa: BLE001 — never kill the thread
                reason = str(exc).splitlines()[0][:80] if str(exc) else type(exc).__name__
                self._set_song_status(f"error: {reason}")
                self._emit(
                    NarrationEvent(
                        ts=time.time(),
                        kind="gemini",
                        text=f"Suno song failed: {reason}",
                        style=style,
                    )
                )
                return

            # Success: play the song over the (muted) MRT2 stream.
            self._set_song_status("playing")
            self._emit(
                NarrationEvent(
                    ts=time.time(),
                    kind="gemini",
                    text="Suno song ready — playing your moment.",
                    style=style,
                )
            )
            self._music.play_external_track(path, label="MoodMirror song")
        except Exception:  # noqa: BLE001 — the song thread must never crash
            logger.exception("Suno song generation failed")
            self._set_song_status("error: song generation crashed")
        finally:
            self._song_inflight.clear()

    def _on_song_status(self, status: str) -> None:
        """Suno progress callback: store status + surface key milestones to feed."""
        self._set_song_status(status)
        # Show progress in the feed when entering "generating" (the long wait).
        if status == "generating":
            self._emit(
                NarrationEvent(
                    ts=time.time(),
                    kind="gemini",
                    text="Suno is composing your full song…",
                )
            )

    # -- read APIs (called from the web server) -----------------------------
    def get_state(self) -> dict:
        """Thread-safe snapshot of the whole pipeline for the dashboard."""
        with self._lock:
            phase = self._phase
            latest = self._latest
            gesture = self._gesture
            last_gesture = self._last_gesture
            directive = self._directive
            latencies = list(self._latencies)
            deadline = self._observe_deadline
            sound_source = self._sound_source
            sound_label = self._sound_label
            situation = self._situation
            gemini_interval = self._gemini_interval
            gemini_model = self._gemini_model
            gemini_models = list(self._gemini_models)
            mode = self._mode
            song_status = self._song_status
            mood3 = self._mood3

        observe_remaining = (
            round(max(0.0, deadline - time.monotonic()), 1)
            if phase == "observing"
            else 0.0
        )

        cap_fps, _ = self._cap_meter.snapshot()
        det_fps, _ = self._det_meter.snapshot()
        emo_fps, _ = self._emo_meter.snapshot()
        ges_fps, _ = self._ges_meter.snapshot()

        last_s = round(latencies[-1], 2) if latencies else None
        avg_s = round(sum(latencies) / len(latencies), 2) if latencies else None

        directive_dict = (
            {
                "style": directive.style,
                "philosophy": directive.philosophy,
                "observation": directive.observation,
                "source": directive.source,
            }
            if directive is not None
            else {}
        )

        return {
            "phase": phase,
            "started": phase != "idle",
            "observe_remaining_s": observe_remaining,
            "face_present": latest.face_present,
            "emotion": latest.emotion,
            "mood3": mood3,
            "smile_score": round(latest.smile_score, 3),
            "smiling": latest.smiling,
            "moving": latest.moving,
            "motion_score": round(latest.motion_score, 3),
            "gesture": {
                "yaw": round(gesture.yaw, 1),
                "pitch": round(gesture.pitch, 1),
                "roll": round(gesture.roll, 1),
                "gaze": round(gesture.gaze, 2),
                "last": last_gesture,
            },
            "directive": directive_dict,
            "situation": situation,
            "gemini_interval_s": round(gemini_interval, 1),
            "gemini_model": gemini_model,
            "gemini_models": gemini_models,
            "mode": mode,
            "local_model_name": config.LOCAL_MODEL_NAME,
            "suno_available": self._suno is not None,
            "song_status": song_status,
            "playing_song": self._music.playing_external,
            "current_track": self._music.current_track,
            "current_style": self._music.current_style,
            "sound_mode": {
                "source": sound_source,
                "label": sound_label,
                "style": self._music.current_style,
            },
            "music_ready": self._music.is_alive(),
            "audio_underruns": self._music.underruns,
            "fps": {
                "capture": round(cap_fps, 1),
                "detect": round(det_fps, 1),
                "emotion": round(emo_fps, 1),
                "gesture": round(ges_fps, 1),
            },
            "latency": {
                "last_s": last_s,
                "avg_s": avg_s,
                "samples": len(latencies),
            },
            "engine": {
                "vision": self._detector.backend,
                "emotion": "deepface" if self._emotion_available else "none",
                "gesture": "mediapipe" if self._gesture_available else "none",
                "gemini": "on" if self._director is not None else "off",
            },
        }

    def feed_since(self, seq: int) -> dict:
        """Return narration events with ``seq`` greater than the given cursor.

        Powers incremental polling: the dashboard passes the last ``seq`` it has
        seen and receives only newer events. A non-positive ``seq`` returns the
        whole current buffer (initial load / reconnect).
        """
        with self._lock:
            feed = list(self._feed)
            max_seq = self._feed_seq

        if seq <= 0:
            selected = feed
        else:
            selected = [(s, e) for (s, e) in feed if s > seq]

        events = [
            {
                "seq": s,
                "ts": e.ts,
                "clock": e.clock,
                "kind": e.kind,
                "text": e.text,
                "emotion": e.emotion,
                "style": e.style,
            }
            for (s, e) in selected
        ]
        return {"seq": max_seq, "events": events}

    def latest_frame_jpeg(self) -> bytes | None:
        """JPEG-encode the most recent webcam frame (thread-safe copy)."""
        import cv2  # local import: keep OpenCV out of module import time

        with self._lock:
            frame = None if self._latest_frame is None else self._latest_frame.copy()
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame)
        return buf.tobytes() if ok else None

    # -- observe → direct flow ----------------------------------------------
    def _observe_and_direct(self, reason: str) -> None:
        """Watch for ``OBSERVE_SECONDS``, ask Gemini, then stream the directive.

        On reset, the OLD music keeps streaming through the observe + direct
        steps; only the final ``music.set_style`` swaps it. Guarded by
        ``self._observing`` so concurrent flows can't overlap.
        """
        if self._observing.is_set():
            logger.info("_observe_and_direct(%s) skipped — already observing.", reason)
            return
        self._observing.set()
        try:
            # 1. Enter observing; announce it; arm the countdown.
            deadline = time.monotonic() + config.OBSERVE_SECONDS
            text = (
                "New person? Watching for 5 seconds…"
                if reason == "reset"
                else "Watching you for 5 seconds to set the vibe…"
            )
            with self._lock:
                self._phase = "observing"
                self._observe_deadline = deadline
            self._emit(NarrationEvent(ts=time.time(), kind="observe", text=text))

            # 2. Capture OBSERVE_PHOTOS frames spread over OBSERVE_SECONDS.
            frames = self._capture_observe_frames(deadline)

            # 3. Direct: the ONLY network call. Falls back when offline.
            with self._lock:
                self._phase = "directing"
            directive = self._make_directive(frames)
            with self._lock:
                self._directive = directive

            # 4. Anchor the narrator + swap the (streaming) music style.
            self._narrator.set_philosophy(directive.style, directive.philosophy)
            self._music.set_style(directive.style)
            with self._lock:
                self._sound_source = "gemini"
                self._sound_label = directive.style
            self._emit(
                NarrationEvent(
                    ts=time.time(),
                    kind="gemini",
                    text=f"{directive.observation} — philosophy: {directive.philosophy}",
                    style=directive.style,
                )
            )

            # 5. Steady state.
            with self._lock:
                self._phase = "streaming"
                self._observe_deadline = 0.0
        except Exception:  # noqa: BLE001 — never let the flow kill the engine
            logger.exception("observe→direct flow failed (%s)", reason)
            with self._lock:
                self._phase = "streaming"
                self._observe_deadline = 0.0
        finally:
            self._observing.clear()

    def _capture_observe_frames(self, deadline: float) -> list[np.ndarray]:
        """Capture ~OBSERVE_PHOTOS frames spread over the observe window."""
        frames: list[np.ndarray] = []
        gap = config.OBSERVE_SECONDS / max(1, config.OBSERVE_PHOTOS)
        for _ in range(config.OBSERVE_PHOTOS):
            if self._stop.is_set():
                break
            with self._lock:
                frame = (
                    None if self._latest_frame is None else self._latest_frame.copy()
                )
            if frame is not None:
                frames.append(frame)
            # Update the dashboard countdown while we wait.
            with self._lock:
                self._observe_deadline = deadline
            if self._stop.wait(gap):
                break
        return frames

    def _make_directive(self, frames: list[np.ndarray]) -> GeminiDirective:
        """Get the opening directive from Gemini, or a fallback when offline."""
        if self._director is not None and frames:
            return self._director.direct(frames)
        return GeminiDirective.fallback(config.STARTER_STYLE)

    # -- latency callback ---------------------------------------------------
    def _record_latency(self, seconds: float) -> None:
        """Record a measured music-change latency (called by the music engine)."""
        with self._lock:
            self._latencies.append(seconds)
        logger.info("Music-change latency: %.2fs", seconds)

    # -- feed helper --------------------------------------------------------
    def _emit(self, event: NarrationEvent) -> None:
        """Append a narration event to the feed under the lock."""
        with self._lock:
            self._feed_seq += 1
            self._feed.append((self._feed_seq, event))

    # -- vision thread ------------------------------------------------------
    def _vision_loop(self) -> None:
        """Capture frames fast; run detect+fuse on a slower cadence."""
        import cv2  # local import: keep OpenCV out of module import time

        cap = cv2.VideoCapture(self._webcam_index)
        warned = False
        while not cap.isOpened() and not self._stop.is_set():
            if not warned:
                logger.warning(
                    "Webcam %s not open yet — if macOS is asking for camera "
                    "permission, approve it (System Settings → Privacy & Security "
                    "→ Camera). Retrying…",
                    self._webcam_index,
                )
                warned = True
            cap.release()
            if self._stop.wait(1.5):
                return
            cap = cv2.VideoCapture(self._webcam_index)
        if not cap.isOpened():
            return
        logger.info("Webcam %s open.", self._webcam_index)

        last_detect = 0.0
        try:
            while not self._stop.is_set():
                t0 = time.time()
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.05)
                    continue

                # Always refresh the preview frame + count a capture tick.
                with self._lock:
                    self._latest_frame = frame
                self._cap_meter.tick()

                # Heavier detection runs far less often than capture.
                if t0 - last_detect >= config.DETECT_INTERVAL_S:
                    last_detect = t0
                    d0 = time.monotonic()
                    base = self._detector.detect(frame)
                    detect_ms = (time.monotonic() - d0) * 1000.0
                    self._det_meter.tick(last_ms=detect_ms)

                    with self._lock:
                        # FUSE: prefer the richer out-of-process DeepFace emotion
                        # when available and a face is present; else keep the Haar
                        # smile fallback.
                        if self._emotion_available and base.face_present:
                            reading = replace(
                                base,
                                emotion=self._df_emotion,  # type: ignore[arg-type]
                                smile_score=self._df_happy,
                            )
                        else:
                            reading = base
                        self._latest = reading
                        self._recent.append(reading)

                # Respect the capture cadence; 0 means "as fast as possible" but
                # still yield a hair so we don't busy-spin a core.
                if config.CAPTURE_INTERVAL_S > 0:
                    time.sleep(max(0.0, config.CAPTURE_INTERVAL_S - (time.time() - t0)))
                else:
                    time.sleep(0.001)
        finally:
            cap.release()

    # -- emotion thread -----------------------------------------------------
    def _emotion_loop(self) -> None:
        """Run the DeepFace subprocess on the latest frame a few times per second."""
        if not self._emotion_available:
            return
        while not self._stop.is_set():
            with self._lock:
                frame = None if self._latest_frame is None else self._latest_frame.copy()
            if frame is not None:
                e0 = time.monotonic()
                result = self._emotion_worker.classify(frame)
                emotion_ms = (time.monotonic() - e0) * 1000.0
                if result is not None:
                    emo, happy = result
                    self._emo_meter.tick(last_ms=emotion_ms)
                    mood3 = config.EMOTION_BUCKET.get(emo, "normal")
                    with self._lock:
                        self._df_emotion, self._df_happy = emo, happy
                        self._mood3 = mood3
                    self._maybe_apply_mood(mood3)
            if self._stop.wait(config.EMOTION_INTERVAL_S):
                return

    def _maybe_apply_mood(self, mood3: str) -> None:
        """In LOCAL emotion-driven mode, set the music to the 3-class mood's style.

        Smooth (flush=False) so it never stutters; only when the mood bucket
        actually changes, while streaming local music and no Suno song is playing.
        """
        if config.LOCAL_DRIVER != "emotion":
            return
        with self._lock:
            if (
                self._mode != "local"
                or self._phase != "streaming"
                or self._music.playing_external
                or mood3 == self._mood3_applied
            ):
                return
            self._mood3_applied = mood3
        style = config.EMOTION3_SOUND.get(mood3)
        if style:
            self._music.set_style(style, flush=False)  # smooth mood change
            with self._lock:
                self._sound_source = "emotion"
                self._sound_label = mood3
            self._emit(NarrationEvent(ts=time.time(), kind="change",
                                      text=f"Mood looks {mood3} — matching the music.",
                                      emotion=self._df_emotion, style=style))

    # -- gesture thread -----------------------------------------------------
    def _gesture_loop(self) -> None:
        """Sample head pose ~10x/sec; fire micro-gesture music nudges (streaming)."""
        if not self._gesture_available:
            return
        while not self._stop.is_set():
            with self._lock:
                frame = None if self._latest_frame is None else self._latest_frame.copy()
            if frame is not None:
                g0 = time.monotonic()
                reading = self._gesture_worker.read(frame)
                gesture_ms = (time.monotonic() - g0) * 1000.0
                if reading is not None:
                    self._ges_meter.tick(last_ms=gesture_ms)
                    with self._lock:
                        self._gesture = reading
                        phase = self._phase
                    # Micro-gestures nudge the music ONLY while streaming.
                    if phase == "streaming":
                        for g in reading.gestures:
                            self._handle_gesture(g)
            if self._stop.wait(config.GESTURE_INTERVAL_S):
                return

    def _handle_gesture(self, gesture: str) -> None:
        """Relay one fired gesture to the narrator and nudge the music."""
        try:
            event = self._narrator.note_gesture(gesture)
        except Exception:  # noqa: BLE001 — a gesture must never kill the loop
            logger.exception("Narrator.note_gesture failed")
            return
        if event is None:
            return  # on cooldown / unknown
        self._emit(event)
        # The gesture always narrates (emitted above). It only NUDGES the music in
        # "local" mode and while no Suno song is playing — otherwise we'd fight the
        # full song / paused local stream.
        with self._lock:
            mode = self._mode
        if mode != "local" or self._music.playing_external:
            with self._lock:
                self._last_gesture = gesture
                self._last_gesture_ts = time.monotonic()
            return
        # narrator.current_style is now the FIXED config.GESTURE_SOUND[gesture]
        # prompt → a wholesale, clearly audible swap. Apply it immediately.
        self._music.set_style(self._narrator.current_style)
        with self._lock:
            self._last_gesture = gesture
            self._last_gesture_ts = time.monotonic()
            self._sound_source = "gesture"
            self._sound_label = gesture

    # -- narrate thread -----------------------------------------------------
    def _narrate_loop(self) -> None:
        """Ask the narrator to HOLD/CHANGE the music while streaming."""
        while not self._stop.is_set():
            if self._stop.wait(config.NARRATE_INTERVAL_S):
                return
            with self._lock:
                phase = self._phase
                recent = list(self._recent)
            # During startup / reset the observe→direct flow drives narration.
            if phase != "streaming":
                continue
            try:
                event = self._narrator.observe(recent)
            except Exception:  # noqa: BLE001 - narration must never kill the loop
                logger.exception("Narrator.observe failed")
                continue

            self._emit(event)

            # The narrator owns commit timing; the engine just relays a change.
            # When continuous Gemini is on, Gemini owns the mood/style — so we
            # still surface the narrator's read to the feed (emitted above) but do
            # NOT let it drive the music (avoids two brains fighting over style).
            with self._lock:
                mode = self._mode
            if (
                not config.CONTINUOUS_GEMINI
                and mode == "local"
                and not self._music.playing_external
                and event.kind == "change"
                and event.style
            ):
                self._music.set_style(event.style, flush=False)  # smooth mood change
                with self._lock:
                    self._sound_source = "emotion"
                    self._sound_label = event.emotion

    # -- director thread ----------------------------------------------------
    def _director_loop(self) -> None:
        """Continuously re-read the webcam → ask Gemini → update the live music.

        After the opening directive, Gemini becomes the live mood brain: every
        ``GEMINI_INTERVAL_S`` it gets the current frame and the session
        philosophy and returns a fresh (style, situation). A non-empty style
        swaps the streaming music's style and records the situation.

        Inert (returns immediately) unless ``config.CONTINUOUS_GEMINI`` is on AND
        a director is available. Only acts while streaming. The Gemini network
        call and ``music.set_style`` are NEVER made while holding the lock, and
        the body is fully guarded so a failure can never kill the loop. The
        natural cadence is GEMINI_INTERVAL_S between ticks (the ~1-2s call just
        spaces calls a little further apart — calls never pile up).
        """
        if not config.CONTINUOUS_GEMINI or self._director is None:
            return
        while not self._stop.is_set():
            # Re-read the user-adjustable cadence each tick so a dashboard change
            # takes effect on the NEXT tick. Floor at 1.0s as a safety net.
            with self._lock:
                interval = max(1.0, self._gemini_interval)
            if self._stop.wait(interval):
                return
            try:
                with self._lock:
                    phase = self._phase
                    mode = self._mode
                # Only steer the music while streaming (skip idle/observe/direct).
                if phase != "streaming":
                    continue
                # In "suno" mode (or while a Suno song plays) the local continuous
                # Gemini updates PAUSE so they don't fight the full song.
                if mode != "local" or self._music.playing_external:
                    continue

                with self._lock:
                    frame = (
                        None
                        if self._latest_frame is None
                        else self._latest_frame.copy()
                    )
                    philosophy = (
                        self._directive.philosophy if self._directive else ""
                    )
                if frame is None:
                    continue

                # Network call — NOT under the lock.
                style, situation = self._director.update([frame], philosophy)
                if not style:
                    continue  # error / rate-limit / no change → keep current music

                # Record the live "NOW" read either way (it drives the dashboard
                # readout + narration). Only STEER THE MUSIC from Gemini when the
                # local driver is "gemini"; in "emotion" mode the 3-class emotion +
                # left/right turns own the music, so Gemini just narrates.
                with self._lock:
                    self._situation = situation
                if config.LOCAL_DRIVER == "gemini":
                    self._music.set_style(style, flush=False)  # smooth crossfade
                    with self._lock:
                        self._sound_source = "gemini"
                        self._sound_label = situation or "live read"
                self._emit(
                    NarrationEvent(
                        ts=time.time(),
                        kind="gemini",
                        text=situation or "Updated the vibe to the moment.",
                        style=style,
                    )
                )
            except Exception:  # noqa: BLE001 — the director loop must never die
                logger.debug("continuous director tick failed", exc_info=True)

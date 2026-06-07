"""Headless tests for JD's pure logic.

These tests deliberately need NO webcam, NO MRT2 model, and NO audio hardware.
They cover the offline brain (narrator), the immutable models, the OpenCV-only
vision contract, and the FastAPI routes (against a stub engine). The real
:class:`~jd.engine.JDEngine` is never instantiated — it opens a camera and loads
the music model.
"""

from __future__ import annotations

import time

import pytest

from jd import config
from jd.models import (
    FpsStat,
    GeminiDirective,
    GestureReading,
    MoodReading,
    NarrationEvent,
)
from jd.narrator import Narrator


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
class TestModels:
    def test_absent_reading_fields(self):
        r = MoodReading.absent()
        assert r.face_present is False
        assert r.smile_score == 0.0
        assert r.emotion == "neutral"
        assert r.moving is False
        assert r.motion_score == 0.0
        assert isinstance(r.ts, float) and r.ts > 0

    def test_smiling_threshold(self):
        # Threshold is >= 0.35.
        assert MoodReading.absent().smiling is False
        assert _reading(smile=0.34).smiling is False
        assert _reading(smile=0.35).smiling is True
        assert _reading(smile=0.9).smiling is True

    def test_narration_event_clock_format(self):
        ev = NarrationEvent(ts=0.0, kind="observe", text="hi")
        clock = ev.clock
        # HH:MM:SS
        assert len(clock) == 8
        h, m, s = clock.split(":")
        assert h.isdigit() and m.isdigit() and s.isdigit()
        # localtime of a known ts is stable within the same tz.
        assert ev.clock == time.strftime("%H:%M:%S", time.localtime(0.0))

    def test_fps_stat_empty(self):
        f = FpsStat.empty()
        assert f.fps == 0.0
        assert f.last_ms == 0.0

    def test_gesture_reading_absent(self):
        g = GestureReading.absent()
        assert g.face_present is False
        assert g.yaw == 0.0
        assert g.pitch == 0.0
        assert g.roll == 0.0
        assert g.gaze == 0.0
        assert g.motion == 0.0
        assert g.gestures == ()
        assert isinstance(g.ts, float) and g.ts > 0

    def test_gemini_directive_fallback(self):
        style = "warm lo-fi, 72bpm"
        d = GeminiDirective.fallback(style)
        assert d.source == "fallback"
        assert d.style == style
        assert d.philosophy  # non-empty
        assert isinstance(d.observation, str) and d.observation

    def test_narration_event_accepts_gemini_and_gesture_kinds(self):
        ev_g = NarrationEvent(ts=0.0, kind="gemini", text="opening directive")
        assert ev_g.kind == "gemini"
        ev_x = NarrationEvent(ts=0.0, kind="gesture", text="you nodded")
        assert ev_x.kind == "gesture"


# ---------------------------------------------------------------------------
# config — the FIXED sound map sanity (gesture + emotion prompts)
# ---------------------------------------------------------------------------
class TestConfigSoundMap:
    def test_gesture_sound_has_all_six_keys(self):
        expected = {"shake", "nod", "tilt", "rotate", "eye_roll", "eyes_closed"}
        assert set(config.GESTURE_SOUND) == expected
        for key, value in config.GESTURE_SOUND.items():
            assert isinstance(value, str) and value.strip(), key

    def test_emotion_sound_has_seven_emotions(self):
        expected = {
            "happy", "sad", "angry", "fear", "surprise", "disgust", "neutral",
        }
        assert set(config.EMOTION_SOUND) == expected
        for key, value in config.EMOTION_SOUND.items():
            assert isinstance(value, str) and value.strip(), key

    def test_session_flags_are_safe_defaults(self):
        assert config.AUTOSTART is False
        assert config.SAVE_SESSION_AUDIO is False
        assert isinstance(config.PREFILL_BEFORE_PLAY, bool)

    def test_continuous_gemini_flags(self):
        # CONTINUOUS_GEMINI is a bool; GEMINI_INTERVAL_S is a positive number.
        assert isinstance(config.CONTINUOUS_GEMINI, bool)
        assert isinstance(config.GEMINI_INTERVAL_S, (int, float))
        assert config.GEMINI_INTERVAL_S > 0

    def test_all_styles_are_instrumental_no_vocals(self):
        # The user wants NO lyrics: every fixed prompt must suppress singing.
        assert "no vocals" in config.STARTER_STYLE
        for key, value in config.GESTURE_SOUND.items():
            assert "no vocals" in value, key
        for key, value in config.EMOTION_SOUND.items():
            assert "no vocals" in value, key


# ---------------------------------------------------------------------------
# narrator — the offline brain (most important)
# ---------------------------------------------------------------------------
def _reading(
    *,
    emotion: str = "neutral",
    face: bool = True,
    smile: float = 0.5,
    moving: bool = False,
    ts: float | None = None,
) -> MoodReading:
    return MoodReading(
        ts=time.time() if ts is None else ts,
        face_present=face,
        smile_score=smile,
        emotion=emotion,  # type: ignore[arg-type]
        moving=moving,
    )


class TestNarrator:
    def test_no_face_returns_observe_and_keeps_style(self):
        n = Narrator()
        ev = n.observe([])
        assert ev.kind == "observe"
        assert n.current_style == config.STARTER_STYLE

        # A window with only absent readings → still observe, style unchanged.
        ev2 = n.observe([MoodReading.absent() for _ in range(5)])
        assert ev2.kind == "observe"
        assert n.current_style == config.STARTER_STYLE

    def test_positive_mood_does_not_change_style(self):
        n = Narrator()
        window = [_reading(emotion="happy", face=True, smile=0.8) for _ in range(10)]
        for _ in range(5):
            ev = n.observe(window)
            assert ev.kind in {"hold", "observe"}
        assert n.current_style == config.STARTER_STYLE

    def test_lift_mood_commits_after_window(self, monkeypatch):
        """A SUSTAINED lift mood commits a CHANGE only after MOOD_COMMIT_SECONDS.

        The narrator measures streaks with ``time.time()``, so we drive a fake
        clock to keep the test fast and deterministic.
        """
        clock = {"t": 1000.0}
        monkeypatch.setattr("jd.narrator.time.time", lambda: clock["t"])

        n = Narrator()
        window = [_reading(emotion="sad", smile=0.0) for _ in range(8)]

        # First observe starts the streak; not yet committed.
        ev0 = n.observe(window)
        assert ev0.kind == "observe"
        assert n.current_style == config.STARTER_STYLE

        # Still before the commit window → stays observe.
        clock["t"] += config.MOOD_COMMIT_SECONDS * 0.5
        ev1 = n.observe(window)
        assert ev1.kind == "observe"
        assert n.current_style == config.STARTER_STYLE

        # Past the commit window → CHANGE, to the FIXED EMOTION_SOUND["sad"] prompt.
        clock["t"] += config.MOOD_COMMIT_SECONDS + 0.01
        ev2 = n.observe(window)
        assert ev2.kind == "change"
        assert ev2.emotion == "sad"
        assert ev2.style == config.EMOTION_SOUND["sad"]
        assert n.current_style == ev2.style
        assert n.current_style != config.STARTER_STYLE

    def test_repeated_changes_use_fixed_emotion_prompt(self, monkeypatch):
        """Each committed change emits the SAME fixed EMOTION_SOUND prompt.

        The old design cycled through MOOD_STYLES variants; the new fixed-prompt
        behaviour always commits to config.EMOTION_SOUND[mood].
        """
        clock = {"t": 5000.0}
        monkeypatch.setattr("jd.narrator.time.time", lambda: clock["t"])

        n = Narrator()
        window = [_reading(emotion="sad", smile=0.0) for _ in range(8)]
        fixed = config.EMOTION_SOUND["sad"]

        # Prime the streak: the first observe just starts the streak (elapsed=0).
        first = n.observe(window)
        assert first.kind == "observe"

        chosen: list[str] = []
        for _ in range(3):
            # Advance past the commit window each time; after a commit the streak
            # resets to `now`, so we must step forward again for the next commit.
            clock["t"] += config.MOOD_COMMIT_SECONDS + 0.01
            ev = n.observe(window)
            assert ev.kind == "change"
            chosen.append(ev.style)

        # Always the same fixed prompt (no cycling).
        assert chosen == [fixed, fixed, fixed]
        assert n.current_style == fixed

    def test_narration_text_varies_for_steady_positive_mood(self):
        n = Narrator()
        window = [_reading(emotion="happy", smile=0.8) for _ in range(6)]
        texts = {n.observe(window).text for _ in range(6)}
        # Anti-spam rotation means a steady mood does not read identically.
        assert all(t for t in texts)  # all non-empty
        assert len(texts) > 1

    def test_set_philosophy_adopts_style(self):
        n = Narrator()
        assert n.current_style == config.STARTER_STYLE
        n.set_philosophy("warm lo-fi, 72bpm", "keep it calm")
        # current_style becomes the given style (whitespace collapse aside).
        assert n.current_style == "warm lo-fi, 72bpm"

    def test_note_gesture_sets_fixed_style_and_cools_down(self, monkeypatch):
        clock = {"t": 10_000.0}
        monkeypatch.setattr("jd.narrator.time.time", lambda: clock["t"])

        n = Narrator()

        ev = n.note_gesture("nod")
        assert ev is not None
        assert ev.kind == "gesture"
        assert ev.text  # non-empty narration
        # The whole style becomes the FIXED GESTURE_SOUND prompt (not a suffix).
        assert ev.style == config.GESTURE_SOUND["nod"]
        assert n.current_style == config.GESTURE_SOUND["nod"]

        # Immediate repeat of the SAME gesture is on cooldown → None.
        assert n.note_gesture("nod") is None

    def test_note_gesture_eyes_closed_and_shake_fixed_prompts(self, monkeypatch):
        """After set_philosophy, gestures still SET the fixed GESTURE_SOUND prompt.

        Drive a fake clock so the per-gesture cooldown is deterministically clear
        between the two different gestures.
        """
        clock = {"t": 20_000.0}
        monkeypatch.setattr("jd.narrator.time.time", lambda: clock["t"])

        n = Narrator()
        n.set_philosophy("warm lo-fi, 72bpm", "keep it calm")

        ev_closed = n.note_gesture("eyes_closed")
        assert ev_closed is not None
        assert ev_closed.kind == "gesture"
        assert ev_closed.style == config.GESTURE_SOUND["eyes_closed"]
        assert n.current_style == config.GESTURE_SOUND["eyes_closed"]

        # Advance past the cooldown, then a different gesture also sets its prompt.
        clock["t"] += config.GESTURE_COOLDOWN_S + 0.01
        ev_shake = n.note_gesture("shake")
        assert ev_shake is not None
        assert ev_shake.kind == "gesture"
        assert ev_shake.style == config.GESTURE_SOUND["shake"]
        assert n.current_style == config.GESTURE_SOUND["shake"]

        # Unknown gesture → None, style unchanged.
        before = n.current_style
        assert n.note_gesture("bogus") is None
        assert n.current_style == before

    def test_note_gesture_bogus_returns_none(self):
        n = Narrator()
        before = n.current_style
        assert n.note_gesture("bogus") is None
        assert n.current_style == before

    def test_observe_after_philosophy_holds_steady(self):
        n = Narrator()
        n.set_philosophy("warm lo-fi, 72bpm", "keep it calm")
        window = [_reading(emotion="happy", smile=0.8) for _ in range(6)]
        for _ in range(5):
            ev = n.observe(window)
            assert ev.kind in {"hold", "observe"}  # no thrash
            assert ev.text  # narration non-empty


# ---------------------------------------------------------------------------
# vision — OpenCV only, no camera
# ---------------------------------------------------------------------------
class TestVision:
    def test_detect_contract_on_synthetic_frames(self):
        pytest.importorskip("cv2")
        import numpy as np

        from jd.vision import MoodDetector

        det = MoodDetector()
        assert det.backend == "opencv"

        rng = np.random.default_rng(0)
        frame = rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8)

        # First call: no previous frame → motion_score is exactly 0.0.
        r1 = det.detect(frame)
        assert isinstance(r1, MoodReading)
        assert r1.motion_score == 0.0

        # Second call (different frame) → motion_score is a float in [0, 1].
        frame2 = rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8)
        r2 = det.detect(frame2)
        assert isinstance(r2, MoodReading)
        assert isinstance(r2.motion_score, float)
        assert 0.0 <= r2.motion_score <= 1.0

        det.close()

    def test_detect_on_zeros_frame_no_exception(self):
        pytest.importorskip("cv2")
        import numpy as np

        from jd.vision import MoodDetector

        det = MoodDetector()
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        r = det.detect(frame)
        assert isinstance(r, MoodReading)
        # A flat black frame has no motion against an identical/absent prior.
        assert r.motion_score == 0.0
        det.close()


# ---------------------------------------------------------------------------
# gemini_director — the only online seam; never hit the network in tests
# ---------------------------------------------------------------------------
class _RaisingClient:
    """Stub genai client whose generate_content always raises."""

    class _Models:
        def generate_content(self, *a, **k):  # noqa: ANN002, ANN003
            raise RuntimeError("network down (stub)")

    def __init__(self):
        self.models = self._Models()


class _TextResponse:
    """Minimal stand-in for a genai response object (only `.text` is read)."""

    def __init__(self, text: str):
        self.text = text


class _TextClient:
    """Stub genai client whose generate_content returns a fixed JSON `.text`."""

    class _Models:
        def __init__(self, text: str):
            self._text = text

        def generate_content(self, *a, **k):  # noqa: ANN002, ANN003
            return _TextResponse(self._text)

    def __init__(self, text: str):
        self.models = self._Models(text)


class TestGeminiDirector:
    def test_make_director_depends_on_key(self):
        from jd.gemini_director import GeminiDirector, make_director

        d = make_director()
        # Either disabled (no key) → None, or a real director with .direct().
        assert d is None or hasattr(d, "direct")
        if d is not None:
            assert isinstance(d, GeminiDirector)

    def test_make_director_none_without_key(self, monkeypatch):
        from jd import gemini_director

        monkeypatch.setattr(gemini_director.config, "GEMINI_API_KEY", None)
        assert gemini_director.make_director() is None

    def test_direct_falls_back_on_client_error(self, monkeypatch):
        """direct() must NEVER raise and never hit the network.

        We build a GeminiDirector without running __init__ (which would import
        google.genai and construct a real client), then inject a stub client
        whose generate_content raises. direct() should swallow the error and
        return a fallback GeminiDirective.
        """
        pytest.importorskip("cv2")
        import numpy as np

        from jd.gemini_director import GeminiDirector

        director = GeminiDirector.__new__(GeminiDirector)
        director._client = _RaisingClient()
        director._model = "stub-model"

        # A real (non-empty) frame so encoding succeeds and we reach the client.
        frame = np.zeros((16, 16, 3), dtype=np.uint8)
        directive = director.direct([frame])
        assert isinstance(directive, GeminiDirective)
        assert directive.source == "fallback"
        assert directive.style == config.STARTER_STYLE
        assert directive.philosophy  # non-empty

    def test_clean_style_enforces_instrumental(self):
        from jd.gemini_director import _clean_style

        cleaned = _clean_style("joyful pop, soul").lower()
        assert "no vocals" in cleaned or "instrumental" in cleaned

    def test_update_falls_back_on_client_error(self, monkeypatch):
        """update() must NEVER raise and never hit the network on error.

        Same seam as direct(): inject a stub client whose generate_content
        raises. update() should swallow the error and return ("", "").
        """
        pytest.importorskip("cv2")
        import numpy as np

        from jd.gemini_director import GeminiDirector

        director = GeminiDirector.__new__(GeminiDirector)
        director._client = _RaisingClient()
        director._model = "stub-model"

        frame = np.zeros((16, 16, 3), dtype=np.uint8)
        result = director.update([frame], "calm")
        assert result == ("", "")

    def test_update_happy_path_mocked(self, monkeypatch):
        """update() parses Gemini JSON → (clean style w/ no-vocals, situation)."""
        pytest.importorskip("cv2")
        import numpy as np

        from jd.gemini_director import GeminiDirector

        director = GeminiDirector.__new__(GeminiDirector)
        director._client = _TextClient(
            '{"situation":"calm focused","style":"lofi, 70bpm"}'
        )
        director._model = "stub-model"

        frame = np.zeros((16, 16, 3), dtype=np.uint8)
        style, situation = director.update([frame], "calm")
        assert situation == "calm focused"
        assert "no vocals" in style.lower() or "instrumental" in style.lower()


# ---------------------------------------------------------------------------
# gesture detection — pure manager-side detector, no subprocess spawned
# ---------------------------------------------------------------------------
class TestGestureDetection:
    def test_available_is_bool(self):
        from jd.gesture_proc import GestureWorker

        w = GestureWorker()
        assert isinstance(w.available, bool)

    def _feed(self, worker, samples, *, dt=0.1, t0=1000.0):
        """Push synthetic (yaw, pitch, roll, gaze, blink) samples into history.

        History entries are 6-tuples (ts, yaw, pitch, roll, gaze, blink).
        Returns the timestamp of the last sample (the "now" to detect against).
        """
        for i, sample in enumerate(samples):
            yaw, pitch, roll, gaze, blink = sample
            worker._hist.append((t0 + i * dt, yaw, pitch, roll, gaze, blink))
        return t0 + (len(samples) - 1) * dt

    def test_oscillating_yaw_detects_shake(self):
        from jd.gesture_proc import GestureWorker

        w = GestureWorker()
        amp = config.SHAKE_YAW_DEG + 5.0
        # Oscillating yaw within the osc window → a head SHAKE.
        samples = [(amp if i % 2 == 0 else -amp, 0.0, 0.0, 0.0, 0.0) for i in range(6)]
        now = self._feed(w, samples)
        assert "shake" in w._detect(now, 0.0)

    def test_oscillating_pitch_detects_nod(self):
        from jd.gesture_proc import GestureWorker

        w = GestureWorker()
        amp = config.NOD_PITCH_DEG + 5.0
        samples = [(0.0, amp if i % 2 == 0 else -amp, 0.0, 0.0, 0.0) for i in range(6)]
        now = self._feed(w, samples)
        assert "nod" in w._detect(now, 0.0)

    def test_steady_pose_detects_no_gesture(self):
        from jd.gesture_proc import GestureWorker

        w = GestureWorker()
        samples = [(1.0, 1.0, 1.0, 0.0, 0.0) for _ in range(6)]
        now = self._feed(w, samples)
        assert w._detect(now, 0.0) == ()

    def test_sustained_blink_detects_eyes_closed(self):
        """A blink held past EYES_CLOSED_HOLD_S fires 'eyes_closed' exactly once."""
        from jd.gesture_proc import GestureWorker

        w = GestureWorker()
        high = config.EYES_CLOSED_BLINK + 0.1
        t0 = 1000.0
        # Steady pose, eyes shut the whole time so only eyes_closed can fire.
        samples = [(1.0, 1.0, 1.0, 0.0, high) for _ in range(8)]
        # Feed history up to (but not including) the final sample's detect call.
        self._feed(w, samples, dt=0.1, t0=t0)

        # First crossing arms the timer (no fire yet).
        armed = w._detect(t0, high)
        assert "eyes_closed" not in armed

        # Held past the hold window → fires once.
        held = w._detect(t0 + config.EYES_CLOSED_HOLD_S + 0.05, high)
        assert "eyes_closed" in held

    def test_quick_blink_does_not_detect_eyes_closed(self):
        """A brief blink that re-opens before HOLD_S must NOT fire eyes_closed."""
        from jd.gesture_proc import GestureWorker

        w = GestureWorker()
        high = config.EYES_CLOSED_BLINK + 0.1
        t0 = 1000.0
        self._feed(w, [(1.0, 1.0, 1.0, 0.0, high) for _ in range(4)], dt=0.1, t0=t0)

        # Arm at t0 with eyes shut, but re-open well before HOLD_S elapses.
        assert "eyes_closed" not in w._detect(t0, high)
        # Eyes open (blink below threshold) re-arms → never fired.
        reopened = w._detect(t0 + config.EYES_CLOSED_HOLD_S * 0.5, 0.0)
        assert "eyes_closed" not in reopened


# ---------------------------------------------------------------------------
# server — routes against a stub engine (no real JDEngine)
# ---------------------------------------------------------------------------
class _FakeEngine:
    """Minimal stand-in implementing the engine surface the server uses."""

    STATE = {
        "face_present": True,
        "emotion": "happy",
        "current_style": "test style",
        "fps": {"capture": 12.0, "detect": 8.0, "emotion": 2.0},
    }
    FEED = {"seq": 3, "events": [{"seq": 1, "kind": "observe", "text": "hi"}]}

    def __init__(self):
        self.started = False
        self.shut = False
        self.reset_calls = 0
        self.begin_calls = 0

    def start(self) -> None:
        self.started = True

    def begin_session(self) -> None:
        self.begin_calls += 1

    def shutdown(self) -> None:
        self.shut = True

    def reset(self) -> None:
        self.reset_calls += 1

    def get_state(self) -> dict:
        return dict(self.STATE)

    def feed_since(self, seq: int) -> dict:
        return dict(self.FEED)

    def latest_frame_jpeg(self):
        return None


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    from jd.server import create_app

    engine = _FakeEngine()
    app = create_app(engine)
    with TestClient(app) as c:
        yield c, engine


class TestServer:
    def test_health(self, client):
        c, _ = client
        resp = c.get("/api/health")
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}

    def test_state_returns_stub(self, client):
        c, _ = client
        resp = c.get("/api/state")
        assert resp.status_code == 200
        assert resp.json() == _FakeEngine.STATE

    def test_feed_returns_stub(self, client):
        c, _ = client
        resp = c.get("/api/feed?since=0")
        assert resp.status_code == 200
        assert resp.json() == _FakeEngine.FEED

    def test_index_serves_html(self, client):
        c, _ = client
        resp = c.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "<html" in resp.text.lower()

    def test_start_calls_begin_session(self, client):
        c, engine = client
        assert engine.begin_calls == 0
        resp = c.post("/api/start")
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}
        assert engine.begin_calls == 1

    def test_reset_calls_engine(self, client):
        c, engine = client
        assert engine.reset_calls == 0
        resp = c.post("/api/reset")
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}
        assert engine.reset_calls == 1

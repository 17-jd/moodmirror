"""Manages the MediaPipe gesture subprocess (gesture_worker.py).

Keeps MediaPipe out of the main process (it aborts alongside MLX/TF). The worker
is persistent — the FaceLandmarker loads once — and we round-trip a frame per
request over pipes. This manager also turns the raw per-frame pose+gaze
time-series into DISCRETE, debounced micro-gestures (shake/nod/tilt/turn_left/
turn_right/eye_roll), since the worker emits only raw pose.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import subprocess
import sys
import threading
import time
from collections import deque

import numpy as np

from . import config
from .models import GestureReading

logger = logging.getLogger(__name__)
_WORKER = config.BASE_DIR / "jd" / "gesture_worker.py"
_HISTORY = 24  # ~2.4s of samples at 10fps; covers the oscillation window
_MOTION_NORM_DEG = 25.0  # pose delta (deg/sample) that maps to motion == 1.0
_SUSTAIN = 2  # frames |roll|/|yaw| must hold for tilt/rotate


class GestureWorker:
    """Out-of-process MediaPipe pose reader + discrete gesture detector.

    `read()` is synchronous and thread-safe (a lock guards the pipe + history).
    """

    def __init__(self) -> None:
        self.available = (
            importlib.util.find_spec("mediapipe") is not None
            and config.FACE_LANDMARKER_PATH.exists()
        )
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._frame_path = config.DATA_DIR / ".gesture_frame.jpg"
        # Rolling history of (ts, yaw, pitch, roll, gaze, blink).
        self._hist: deque[tuple[float, float, float, float, float, float]] = deque(maxlen=_HISTORY)
        # Per-gesture last-fired timestamps for cooldown debouncing.
        self._last_fired: dict[str, float] = {}
        # Eyes-closed state: when the eyes first crossed the threshold this closure
        # (None = currently open / re-armed), and whether we've already fired for it.
        self._eyes_closed_since: float | None = None
        self._eyes_closed_fired: bool = False
        # Directional turn re-arm: a turn may fire again only after the head has
        # returned toward center (|yaw| < ROTATE_YAW_DEG*0.5) since the last fire.
        self._turn_armed: dict[str, bool] = {"turn_left": True, "turn_right": True}

    def start(self) -> bool:
        if not self.available:
            logger.info("MediaPipe / face_landmarker.task unavailable — gestures disabled.")
            return False
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, JD_FACE_LANDMARKER_PATH=str(config.FACE_LANDMARKER_PATH))
        try:
            self._proc = subprocess.Popen(
                [sys.executable, str(_WORKER)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
                env=env,
            )
            logger.info("MediaPipe gesture worker started (pid %s)", self._proc.pid)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Gesture worker failed to start: %s", exc)
            self._proc = None
            return False

    def read(self, frame_bgr: np.ndarray) -> GestureReading | None:
        """Round-trip a frame and return a fused GestureReading.

        Returns GestureReading.absent() when no face; None if the subprocess is dead.
        """
        proc = self._proc
        if proc is None or proc.poll() is not None or proc.stdin is None or proc.stdout is None:
            return None
        import cv2  # noqa: PLC0415

        with self._lock:
            try:
                cv2.imwrite(str(self._frame_path), frame_bgr)
                proc.stdin.write(f"{self._frame_path}\n")
                proc.stdin.flush()
                line = proc.stdout.readline()
                if not line:
                    return None
                d = json.loads(line)
            except Exception as exc:  # noqa: BLE001
                logger.debug("gesture read failed: %s", exc)
                return None

            if "error" in d or not d.get("face", False):
                self._hist.clear()
                # No face → eyes can't be tracked; re-arm so re-appearing re-fires.
                self._eyes_closed_since = None
                self._eyes_closed_fired = False
                return GestureReading.absent()

            ts = time.time()
            yaw = float(d.get("yaw", 0.0))
            pitch = float(d.get("pitch", 0.0))
            roll = float(d.get("roll", 0.0))
            gaze = float(d.get("gaze", 0.0))
            blink = float(d.get("blink", 0.0))
            motion = self._motion(yaw, pitch, roll)
            self._hist.append((ts, yaw, pitch, roll, gaze, blink))
            gestures = self._detect(ts, blink)
            return GestureReading(
                ts=ts,
                face_present=True,
                yaw=yaw,
                pitch=pitch,
                roll=roll,
                gaze=gaze,
                motion=motion,
                gestures=gestures,
            )

    def _motion(self, yaw: float, pitch: float, roll: float) -> float:
        """0..1 pose change magnitude vs the previous sample."""
        if not self._hist:
            return 0.0
        _, py, pp, pr, _, _ = self._hist[-1]  # (ts, yaw, pitch, roll, gaze, blink)
        delta = abs(yaw - py) + abs(pitch - pp) + abs(roll - pr)
        return float(min(1.0, delta / _MOTION_NORM_DEG))

    def _detect(self, now: float, blink: float) -> tuple[str, ...]:
        """Run all discrete detectors over the rolling history; apply cooldowns."""
        fired: list[str] = []
        if self._oscillates(1, config.SHAKE_YAW_DEG, now):
            fired.append("shake")
        if self._oscillates(2, config.NOD_PITCH_DEG, now):
            fired.append("nod")
        if self._sustained(3, config.TILT_ROLL_DEG):
            fired.append("tilt")
        # Directional head turn (yaw: + = RIGHT, - = LEFT). Re-arm runs every
        # frame (independent of whether a turn is currently sustained) so the
        # latch resets once the head returns toward center.
        self._update_turn_rearm()
        if self._sustained_dir(1, -config.ROTATE_YAW_DEG) and self._fire_turn("turn_left"):
            fired.append("turn_left")
        if self._sustained_dir(1, +config.ROTATE_YAW_DEG) and self._fire_turn("turn_right"):
            fired.append("turn_right")
        if self._spike(4, config.EYEROLL_GAZE):
            fired.append("eye_roll")
        if self._eyes_closed(now, blink):
            fired.append("eyes_closed")
        return tuple(g for g in fired if self._cooldown_ok(g, now))

    def _eyes_closed(self, now: float, blink: float) -> bool:
        """Fire once when eyes have been shut (blink >= thr) for >= HOLD_S.

        Tracks when the eyes first crossed the threshold; fires a single time once
        they've held that long. Re-arms only after the eyes re-open (blink drops
        below the threshold) — so it won't spam while the eyes stay shut.
        """
        if blink >= config.EYES_CLOSED_BLINK:
            if self._eyes_closed_since is None:
                self._eyes_closed_since = now
            if (
                not self._eyes_closed_fired
                and now - self._eyes_closed_since >= config.EYES_CLOSED_HOLD_S
            ):
                self._eyes_closed_fired = True
                return True
            return False
        # Eyes open → re-arm for the next closure.
        self._eyes_closed_since = None
        self._eyes_closed_fired = False
        return False

    def _window(self, now: float) -> list[tuple[float, float, float, float, float]]:
        cutoff = now - config.GESTURE_OSC_WINDOW_S
        return [s for s in self._hist if s[0] >= cutoff]

    def _oscillates(self, idx: int, amp_deg: float, now: float) -> bool:
        """Sign-change in the (mean-centred) axis with peak-to-peak amplitude > amp."""
        win = self._window(now)
        if len(win) < 3:
            return False
        vals = [s[idx] for s in win]
        if max(vals) - min(vals) < amp_deg:
            return False
        centred = [v - (sum(vals) / len(vals)) for v in vals]
        signs = [1 if v > 0 else -1 for v in centred if abs(v) > amp_deg * 0.25]
        if len(signs) < 2:
            return False
        return any(a != b for a, b in zip(signs, signs[1:]))

    def _sustained(self, idx: int, thr_deg: float) -> bool:
        """The last _SUSTAIN samples all exceed |thr| on the given axis."""
        if len(self._hist) < _SUSTAIN:
            return False
        recent = list(self._hist)[-_SUSTAIN:]
        return all(abs(s[idx]) > thr_deg for s in recent)

    def _sustained_dir(self, idx: int, thr_deg: float) -> bool:
        """The last _SUSTAIN samples all pass a SIGNED threshold on the given axis.

        thr_deg > 0 → all samples must be >= thr_deg (turned positive/RIGHT).
        thr_deg < 0 → all samples must be <= thr_deg (turned negative/LEFT).
        """
        if len(self._hist) < _SUSTAIN:
            return False
        recent = list(self._hist)[-_SUSTAIN:]
        if thr_deg >= 0:
            return all(s[idx] >= thr_deg for s in recent)
        return all(s[idx] <= thr_deg for s in recent)

    def _update_turn_rearm(self) -> None:
        """Re-arm directional turns once the head returns toward center.

        Runs every frame: when |yaw| drops below ROTATE_YAW_DEG*0.5, both turn
        latches re-arm so the next sustained turn can fire again.
        """
        rearm_thr = config.ROTATE_YAW_DEG * 0.5
        cur_yaw = self._hist[-1][1] if self._hist else 0.0
        if abs(cur_yaw) < rearm_thr:
            self._turn_armed["turn_left"] = True
            self._turn_armed["turn_right"] = True

    def _fire_turn(self, gesture: str) -> bool:
        """Consume the re-arm latch: fire once, then stay disarmed until re-armed.

        A turn fires only when armed; firing disarms it. Combined with
        _update_turn_rearm (re-arms at center), each turn fires once per movement.
        """
        if self._turn_armed.get(gesture, True):
            self._turn_armed[gesture] = False
            return True
        return False

    def _spike(self, idx: int, thr: float) -> bool:
        """A quick excursion: current gaze high while the recent baseline was low."""
        if len(self._hist) < 2:
            return False
        cur = self._hist[-1][idx]
        prev = [s[idx] for s in list(self._hist)[-5:-1]]
        baseline = min(prev) if prev else 0.0
        return cur > thr and (cur - baseline) > thr * 0.5

    def _cooldown_ok(self, gesture: str, now: float) -> bool:
        last = self._last_fired.get(gesture, 0.0)
        if now - last < config.GESTURE_COOLDOWN_S:
            return False
        self._last_fired[gesture] = now
        return True

    def stop(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            try:
                proc.kill()
                proc.wait(timeout=2)
            except Exception:  # noqa: BLE001
                pass

"""Manages the DeepFace subprocess (emotion_worker.py).

Keeps TensorFlow out of the main process (it aborts alongside MLX). The worker is
persistent — TF loads once — and we round-trip a frame per request over pipes.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import subprocess
import sys
import threading

import numpy as np

from . import config

logger = logging.getLogger(__name__)
_VALID = {"happy", "sad", "angry", "surprise", "fear", "disgust", "neutral"}
_WORKER = config.BASE_DIR / "jd" / "emotion_worker.py"


class EmotionWorker:
    """Out-of-process DeepFace classifier. `classify()` is synchronous + thread-safe."""

    def __init__(self) -> None:
        self.available = importlib.util.find_spec("deepface") is not None
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._frame_path = config.DATA_DIR / ".emotion_frame.jpg"

    def start(self) -> bool:
        if not self.available:
            logger.info("DeepFace not installed — emotion worker disabled.")
            return False
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        try:
            self._proc = subprocess.Popen(
                [sys.executable, str(_WORKER)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
            logger.info("DeepFace emotion worker started (pid %s)", self._proc.pid)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Emotion worker failed to start: %s", exc)
            self._proc = None
            return False

    def classify(self, frame_bgr: np.ndarray) -> tuple[str, float] | None:
        """Returns (emotion, happy_score 0..1), or None if unavailable/failed."""
        proc = self._proc
        if proc is None or proc.poll() is not None or proc.stdin is None or proc.stdout is None:
            return None
        import cv2

        with self._lock:
            try:
                cv2.imwrite(str(self._frame_path), frame_bgr)
                proc.stdin.write(f"{self._frame_path}\n")
                proc.stdin.flush()
                line = proc.stdout.readline()
                if not line:
                    return None
                d = json.loads(line)
                if "error" in d:
                    return None
                emo = str(d.get("emotion", "neutral"))
                return (emo if emo in _VALID else "neutral"), float(d.get("happy", 0.0))
            except Exception as exc:  # noqa: BLE001
                logger.debug("emotion classify failed: %s", exc)
                return None

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

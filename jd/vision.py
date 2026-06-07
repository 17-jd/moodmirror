"""Local webcam face + smile + motion detection via OpenCV Haar cascades.

Pure OpenCV, in-process and fast. This module NEVER imports TensorFlow, DeepFace,
or MediaPipe: DeepFace/TF aborts natively if loaded in the same process as the MLX
music engine ("libc++abi mutex lock failed"). The real 7-class emotion is handled
OUT OF PROCESS by emotion_proc.EmotionWorker and fused in later by the engine; the
smile-based emotion here is only a fast fallback.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from . import config
from .models import MoodReading

logger = logging.getLogger(__name__)

# Downscaled gray size used for cheap mean-abs frame-diff motion estimation.
_MOTION_W = 64
_MOTION_H = 48


class MoodDetector:
    """Fast Haar face + binary smile + frame-diff motion.

    Stateful: keeps the previous downscaled grayscale frame so each ``detect``
    call can estimate scene/face motion against the prior frame. Emotion is a
    smile-based fallback; the real 7-class emotion is filled in by EmotionWorker.
    """

    def __init__(self) -> None:
        import cv2  # noqa: PLC0415 — keep OpenCV import local; never import TF/MediaPipe

        from pathlib import Path

        self._cv2 = cv2
        base = Path(cv2.data.haarcascades)
        self._face = cv2.CascadeClassifier(
            str(base / "haarcascade_frontalface_default.xml")
        )
        self._smile = cv2.CascadeClassifier(str(base / "haarcascade_smile.xml"))
        self._prev_small: np.ndarray | None = None  # previous downscaled gray frame
        self._backend = "opencv"
        logger.info("MoodDetector backend: %s", self._backend)

    @property
    def backend(self) -> str:
        """Name of the active detection backend (always ``"opencv"``)."""
        return self._backend

    def _motion(self, gray: np.ndarray) -> tuple[bool, float]:
        """Compute mean-abs frame-diff motion against the previous frame.

        Downscales the gray frame to a small fixed size and compares it to the
        prior frame. Returns ``(moving, motion_score)`` where ``motion_score`` is
        in ``0..1``. The first call (no previous frame) yields ``(False, 0.0)``.
        Always stores the current small frame as the new previous frame.
        """
        cv2 = self._cv2
        small = cv2.resize(gray, (_MOTION_W, _MOTION_H), interpolation=cv2.INTER_AREA)
        prev = self._prev_small
        self._prev_small = small
        if prev is None:
            return False, 0.0
        score = float(np.mean(np.abs(small.astype(np.int16) - prev.astype(np.int16)))) / 255.0
        return score >= config.MOTION_THRESHOLD, score

    def detect(self, frame_bgr: np.ndarray) -> MoodReading:
        """Detect the largest face, a binary smile, and frame-diff motion.

        Motion is always computed (it can be present without a face). When no face
        is found, returns an absent reading that still carries the motion fields.
        Builds a new immutable ``MoodReading`` — never mutates an existing one.
        """
        cv2 = self._cv2
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        moving, score = self._motion(gray)

        faces = self._face.detectMultiScale(gray, 1.3, 5, minSize=(80, 80))
        if len(faces) == 0:
            return MoodReading(
                ts=time.time(),
                face_present=False,
                smile_score=0.0,
                emotion="neutral",
                moving=moving,
                motion_score=round(score, 4),
            )

        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        roi = gray[y + h // 2 : y + h, x : x + w]  # lower half of the face
        smiling = len(self._smile.detectMultiScale(roi, 1.7, 20)) > 0
        return MoodReading(
            ts=time.time(),
            face_present=True,
            smile_score=0.8 if smiling else 0.15,
            emotion="happy" if smiling else "neutral",
            moving=moving,
            motion_score=round(score, 4),
        )

    def close(self) -> None:
        """Release resources. No-op for the pure-OpenCV backend."""

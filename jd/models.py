"""Immutable shared data models for JD.

Small frozen dataclasses passed between the vision, narrator, music, and server
layers. Keeping them in one place avoids circular imports.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

Emotion = Literal["happy", "sad", "angry", "surprise", "fear", "disgust", "neutral"]
# Session phase: idle → observing (5s) → directing (Gemini) → streaming. Reset
# re-enters observing while the stream keeps playing.
Phase = Literal["idle", "observing", "directing", "streaming"]
# Discrete micro-gestures that drive the music (each maps to a fixed sound).
Gesture = Literal[
    "shake", "nod", "tilt", "rotate", "turn_left", "turn_right", "eye_roll", "eyes_closed"
]
# Simplified 3-class mood (user request): just normal / happy / sad.
Mood3 = Literal["happy", "sad", "normal"]


@dataclass(frozen=True)
class MoodReading:
    """A single webcam mood sample (one fused vision+emotion observation)."""

    ts: float
    face_present: bool
    smile_score: float  # 0..1 (DeepFace 'happy' probability, or Haar smile fallback)
    emotion: Emotion
    moving: bool = False  # face/scene motion since the previous frame (Haar/frame-diff)
    motion_score: float = 0.0  # 0..1 continuous motion magnitude

    @property
    def smiling(self) -> bool:
        return self.smile_score >= 0.35

    @staticmethod
    def absent() -> "MoodReading":
        return MoodReading(
            ts=time.time(),
            face_present=False,
            smile_score=0.0,
            emotion="neutral",
        )


@dataclass(frozen=True)
class GestureReading:
    """One head-pose / micro-gesture sample from the MediaPipe gesture subprocess."""

    ts: float
    face_present: bool
    yaw: float = 0.0  # degrees, +right / -left (head turn)
    pitch: float = 0.0  # degrees, +up / -down (nod axis)
    roll: float = 0.0  # degrees (head tilt)
    gaze: float = 0.0  # 0..1 normalized iris excursion from center (eye-roll proxy)
    motion: float = 0.0  # 0..1 landmark motion magnitude since last sample
    # Discrete one-shot gestures detected this sample (already debounced upstream).
    gestures: tuple[str, ...] = ()

    @staticmethod
    def absent() -> "GestureReading":
        return GestureReading(ts=time.time(), face_present=False)


@dataclass(frozen=True)
class GeminiDirective:
    """The opening instruction Gemini writes after watching the subject for 5s.

    `style` is an MRT2/MusicCoCa short comma-separated tag prompt (the thing the
    music engine streams). `philosophy` is the consistent musical direction to
    hold across the session (gestures/emotions nudge WITHIN it; it doesn't break).
    """

    style: str  # e.g. "warm lo-fi, soft rhodes, 72bpm, mellow"
    philosophy: str  # e.g. "keep it calm and grounding, gentle evolution only"
    observation: str  # Gemini's one-line read of the person/scene
    source: Literal["gemini", "fallback"] = "gemini"

    @staticmethod
    def fallback(style: str, observation: str = "Offline default — Gemini unavailable.") -> "GeminiDirective":
        return GeminiDirective(
            style=style,
            philosophy="hold a steady, comforting groove; evolve gently",
            observation=observation,
            source="fallback",
        )


@dataclass(frozen=True)
class NarrationEvent:
    """One line in the live narration feed shown on the dashboard.

    `kind` styles the line in the UI:
      * observe  — neutral status ("subject moving, mood looks good")
      * hold     — decided to KEEP the current music
      * change   — decided to CHANGE the music (carries the new style)
      * gemini   — the opening directive from the 5s observe → Gemini step
      * gesture  — a micro-gesture (shake/nod/eye-roll/rotate) nudged the music
    """

    ts: float
    kind: Literal["observe", "hold", "change", "gemini", "gesture"]
    text: str  # human-readable narration shown in the feed
    emotion: Emotion = "neutral"
    style: str = ""  # the music style prompt when kind == "change"

    @property
    def clock(self) -> str:
        return time.strftime("%H:%M:%S", time.localtime(self.ts))


@dataclass(frozen=True)
class FpsStat:
    """A rolling frames-per-second measurement for one pipeline stage."""

    fps: float  # smoothed frames/sec
    last_ms: float  # latency of the most recent single call, milliseconds

    @staticmethod
    def empty() -> "FpsStat":
        return FpsStat(fps=0.0, last_ms=0.0)

"""The offline, deterministic narration brain for JD.

The :class:`Narrator` is JD's brain — but fully OFFLINE: no Gemini, no network,
no LLM. It is a small rule-based engine that, every ``config.NARRATE_INTERVAL_S``,
looks at the recent webcam mood readings and:

  1. decides whether to HOLD the current music or CHANGE to a mood-lifting style, and
  2. emits ONE human-readable :class:`NarrationEvent` for the live dashboard feed.

Design rules (see ``config.py`` for the knobs):

* Positive moods (happy / neutral / surprise) → HOLD. Don't disrupt a working
  groove; this mirrors the mere-exposure effect.
* Lift moods (sad / angry / fear / disgust) → CHANGE to an uplifting style, but
  only COMMIT once the lift mood has PERSISTED for ``MOOD_COMMIT_SECONDS`` of
  wall-clock so a single noisy frame can't thrash the music.

The engine depends on the exact public interface: ``current_style`` and
``observe``. ``observe`` is deterministic given the same inputs + internal state
(aside from wall-clock streak timing) and does NO IO and NO sleeping. All state
(mood streaks, per-mood style cursors, rotation counter, last text) is internal,
and every event is a brand-new immutable object — nothing is ever mutated.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter

from . import config
from .models import Emotion, MoodReading, NarrationEvent

logger = logging.getLogger(__name__)

# Cap on any style tag string we build (keeps MusicCoCa prompts terse).
_MAX_STYLE_CHARS = 120


def _dominant_emotion(present: list[MoodReading]) -> Emotion:
    """Return the most frequent emotion among face-present readings.

    Ties are broken by recency: the latest reading's emotion wins among the
    tied candidates, so the feed leans toward what the camera sees right now.
    Assumes ``present`` is non-empty (callers guarantee this).
    """
    counts = Counter(r.emotion for r in present)
    top = counts.most_common(1)[0][1]
    tied = {emotion for emotion, n in counts.items() if n == top}
    for reading in reversed(present):
        if reading.emotion in tied:
            return reading.emotion
    return present[-1].emotion


def _fraction_moving(present: list[MoodReading]) -> float:
    """Fraction of face-present readings flagged as moving (0..1)."""
    if not present:
        return 0.0
    return sum(1 for r in present if r.moving) / len(present)


# Narration phrasings, grouped by situation. A rotating counter picks within a
# group so the live feed never reads identically two ticks running. Templates use
# {mood} (human emotion label) and {style} (first words of the chosen style).
_WAITING_TEXT = "Waiting for a face in frame…"

_HOLD_MOVING = (
    "You're moving and the vibe is good — keeping this groove going.",
    "Nice energy in the frame, and the mood's holding — staying on this track.",
    "Movement plus a good mood — this one's working, so I'm letting it ride.",
)

_HOLD_STILL = (
    "Calm and steady. Holding the current track.",
    "Settled and content — no reason to touch the music.",
    "Relaxed in frame and the mood looks fine — keeping things as they are.",
)

_OBSERVE_QUIET = (
    "Still reading you as {mood} — letting the current track breathe.",
    "Mood's sitting at {mood} and it's fine; holding steady.",
    "Looking {mood} and comfortable — no change needed right now.",
)

# Hold phrasings that reference the anchored philosophy (used once one is set).
_HOLD_PHILOSOPHY = (
    "Mood's steady — keeping the vibe you started with.",
    "All good in frame; staying true to the feel we opened on.",
    "No need to move — holding the original groove and its character.",
)

_NOTICING_DIP = (
    "Picking up a low mood ({mood})… giving it a moment before I switch.",
    "I'm noticing a dip toward {mood} — watching to see if it sticks.",
    "Mood's leaning {mood}; I'll hold a beat before changing anything.",
    "Catching some {mood} in your expression — not switching just yet.",
)

_COMMITTED_CHANGE = (
    "Mood dipped to {mood} — lifting it now with {style}.",
    "That {mood} held on, so I'm changing things up: {style}.",
    "Time to lift the {mood} — bringing in {style}.",
    "You've been {mood} for a bit; let's turn it around with {style}.",
)

# Lift-mood changes that also acknowledge the held philosophy (used once a base
# philosophy has been set by the engine after the Gemini directive).
_COMMITTED_CHANGE_PHILOSOPHY = (
    "Mood dipped to {mood} — easing toward {style}, still holding the vibe you started with.",
    "That {mood} stuck around; nudging into {style} while keeping the original feel.",
    "Lifting the {mood} with {style}, staying true to the philosophy we opened on.",
)

# One-shot gesture lines. Each gesture SETS current_style to the FIXED prompt
# config.GESTURE_SOUND[gesture] (a wholesale, clearly AUDIBLE change), and the
# narration names what it just did. The style itself comes from config, not here.
_GESTURE_TEXT: dict[str, str] = {
    "shake": "Head shake — going wobbly and unstable.",
    "nod": "You're nodding — punchy, driving groove.",
    "tilt": "Head tilt — curious, playful melody.",
    "rotate": "You turned away — distant and washed-out.",
    "eye_roll": "Eye-roll — bright and airy now.",
    "eyes_closed": "Eyes closed — taking it deep and slow.",
}


def _style_lead(style: str, words: int = 4) -> str:
    """Return the first few human-readable words of a style prompt."""
    cleaned = style.replace(",", " ").split()
    return " ".join(cleaned[:words]) if cleaned else style


class Narrator:
    """Offline, deterministic HOLD/CHANGE engine + live narration feed.

    All decisions are rule-based and local. The narrator keeps internal state so
    repeated calls produce a varied, readable feed and so mood changes are only
    committed after a persistent streak. It never mutates the readings it is
    given and returns a fresh :class:`NarrationEvent` on every tick.
    """

    def __init__(self) -> None:
        # current_style and cooldowns are touched by observe() and note_gesture(),
        # which the engine may call from different threads — guard with a lock.
        self._lock = threading.Lock()
        self._current_style: str = config.STARTER_STYLE
        # Base philosophy set by the engine from the Gemini directive (the anchor
        # that emotion/gesture nudges must stay tonally consistent with).
        self._philosophy: str = ""
        # Per-mood streak start (monotonic seconds) for the *current* lift mood.
        self._lift_mood: str | None = None
        self._lift_started: float = 0.0
        # Per-mood cursor so repeated changes cycle through MOOD_STYLES[mood].
        self._style_cursor: dict[str, int] = {}
        # Last time each discrete gesture was acted on (for the per-gesture cooldown).
        self._gesture_last: dict[str, float] = {}
        # Rotation counter for deterministic phrasing variety + last text emitted.
        self._rotation: int = 0
        self._last_text: str = ""
        logger.debug("Narrator initialized with starter style: %s", self._current_style)

    @property
    def current_style(self) -> str:
        """The music style prompt JD is currently playing."""
        with self._lock:
            return self._current_style

    def set_philosophy(self, style: str, philosophy: str) -> None:
        """Anchor the session to a Gemini directive's base style + philosophy.

        Called by the engine after the opening Gemini directive (and on Reset).
        The narrator no longer owns the opening style — it adopts ``style`` as the
        current style and keeps ``philosophy`` as the tonal anchor that all later
        emotion-driven changes and gesture nudges stay consistent with. Resets the
        pending lift streak so a fresh directive starts cleanly.
        """
        clean_style = " ".join(style.split()) if style else config.STARTER_STYLE
        if len(clean_style) > _MAX_STYLE_CHARS:
            clean_style = clean_style[:_MAX_STYLE_CHARS].rstrip(", ").rstrip()
        with self._lock:
            self._current_style = clean_style
            self._philosophy = " ".join(philosophy.split()) if philosophy else ""
            self._lift_mood = None
            self._lift_started = 0.0
        logger.info("Philosophy set → style=%r philosophy=%r", clean_style, self._philosophy)

    def note_gesture(self, gesture: str) -> NarrationEvent | None:
        """React to one discrete micro-gesture with a FIXED, distinct sound.

        Called by the engine when the gesture subprocess fires a discrete gesture
        (``"shake"``, ``"nod"``, ``"tilt"``, ``"rotate"``, ``"eye_roll"``,
        ``"eyes_closed"``). Instead of layering a vague suffix, this SETS
        ``current_style`` to the fixed prompt ``config.GESTURE_SOUND[gesture]`` so
        the whole style becomes that prompt — the change is clearly AUDIBLE.
        Returns a fresh ``NarrationEvent(kind="gesture", ...)`` carrying that exact
        style (the emotion is left unchanged at ``"neutral"``).

        Repeated identical gestures inside ``config.GESTURE_COOLDOWN_S`` return
        ``None`` (the engine then skips), so a held or jittery gesture can't spam
        the music. Unknown gestures return ``None``.
        """
        text = _GESTURE_TEXT.get(gesture)
        style = config.GESTURE_SOUND.get(gesture)
        if text is None or style is None:
            logger.debug("Ignoring unknown gesture: %s", gesture)
            return None

        now = time.time()
        with self._lock:
            last = self._gesture_last.get(gesture, 0.0)
            if now - last < config.GESTURE_COOLDOWN_S:
                return None  # on cooldown — engine skips
            self._gesture_last[gesture] = now
            self._current_style = style
            event = self._emit(now, "gesture", text, "neutral", style=style)
        logger.info("Gesture %s → fixed style: %s", gesture, event.style)
        return event

    def observe(self, recent: list[MoodReading]) -> NarrationEvent:
        """Evaluate recent mood and emit ONE narration event.

        Deterministic and cheap: no sleeping, no IO. Called by the engine every
        ``config.NARRATE_INTERVAL_S``.
        """
        now = time.time()
        present = [r for r in recent if r.face_present]
        latest = recent[-1] if recent else None

        with self._lock:
            # 1. No data / no face → wait, don't touch the music or the streak.
            if not present or latest is None or not latest.face_present:
                self._lift_mood = None
                return self._emit(now, "observe", _WAITING_TEXT, "neutral")

            mood = _dominant_emotion(present)
            mean_smile = sum(r.smile_score for r in present) / len(present)
            moving = bool(latest.moving) or _fraction_moving(present) >= 0.5

            # 2. Positive mood → HOLD the groove (reset any pending lift streak).
            if mood in config.POSITIVE_MOODS:
                self._lift_mood = None
                return self._hold(now, mood, moving, mean_smile)

            # 3. Lift mood → maybe CHANGE, but only after a persistent streak.
            if mood in config.LIFT_MOODS:
                return self._handle_lift(now, mood)

            # Fallback for any unforeseen label: treat as a hold.
            self._lift_mood = None
            return self._hold(now, mood, moving, mean_smile)

    # -- internal helpers (all called while holding self._lock) -------------

    def _hold(
        self, now: float, mood: Emotion, moving: bool, mean_smile: float
    ) -> NarrationEvent:
        """Build a HOLD event, downgrading repeats to a quieter observe."""
        if moving:
            text = self._rotate(_HOLD_MOVING)
        elif self._philosophy:
            # Anchored to a Gemini philosophy → acknowledge we're holding it.
            text = self._rotate(_HOLD_PHILOSOPHY)
        else:
            text = self._rotate(_HOLD_STILL, mood=_label(mood))

        # Anti-spam: if we'd repeat the previous line, drop to a quieter observe.
        kind = "hold"
        if text == self._last_text:
            text = self._rotate(_OBSERVE_QUIET, mood=_label(mood))
            kind = "observe"
        return self._emit(now, kind, text, mood)

    def _handle_lift(self, now: float, mood: Emotion) -> NarrationEvent:
        """Track the lift-mood streak; commit a change once it persists."""
        # (Re)start the streak when the dominant lift mood changes.
        if self._lift_mood != mood:
            self._lift_mood = mood
            self._lift_started = now

        elapsed = now - self._lift_started
        if elapsed < config.MOOD_COMMIT_SECONDS:
            text = self._rotate(_NOTICING_DIP, mood=_label(mood))
            return self._emit(now, "observe", text, mood)

        # Committed: switch to the FIXED, distinct sound for this mood so the
        # change is clearly audible (no cycling through MOOD_STYLES variants).
        chosen = config.EMOTION_SOUND.get(mood) or self._next_style(mood)
        self._current_style = chosen
        # Reset the streak so we don't immediately re-commit next tick; further
        # changes require the mood to keep persisting.
        self._lift_started = now
        # When a philosophy is anchored, prefer phrasing that signals we're staying
        # tonally consistent with the opening directive.
        templates = _COMMITTED_CHANGE_PHILOSOPHY if self._philosophy else _COMMITTED_CHANGE
        text = self._rotate(templates, mood=_label(mood), style=_style_lead(chosen))
        logger.info("Mood %s committed → changing style to: %s", mood, chosen)
        return self._emit(now, "change", text, mood, style=chosen)

    def _next_style(self, mood: str) -> str:
        """Cycle through ``config.MOOD_STYLES[mood]`` across repeated changes."""
        styles = config.MOOD_STYLES.get(mood) or config.MOOD_STYLES.get("neutral", [])
        if not styles:
            return config.STARTER_STYLE
        idx = self._style_cursor.get(mood, 0) % len(styles)
        self._style_cursor[mood] = idx + 1
        return styles[idx]

    def _rotate(self, templates: tuple[str, ...], **fields: str) -> str:
        """Pick a template deterministically by the rotation counter and format it."""
        template = templates[self._rotation % len(templates)]
        self._rotation += 1
        return template.format(**fields)

    def _emit(
        self,
        ts: float,
        kind: str,
        text: str,
        emotion: Emotion,
        style: str = "",
    ) -> NarrationEvent:
        """Build a fresh NarrationEvent and remember the text for anti-spam."""
        self._last_text = text
        return NarrationEvent(
            ts=ts,
            kind=kind,  # type: ignore[arg-type]
            text=text,
            emotion=emotion,
            style=style,
        )


def _label(mood: str) -> str:
    """Human-friendly mood label (currently identity; a hook for future tweaks)."""
    return mood

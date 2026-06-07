"""Gemini "director" — the ONLY online part of JD.

Used ONLY at session start and on Reset. The engine watches the subject for
~5 seconds (5 photos, ~1/sec) and hands the frames here. We send them to Gemini
and get back the OPENING music directive: a starting MRT2/MusicCoCa style, a
"philosophy" (the consistent direction to hold for the whole session), and a
one-line observation. The engine then streams instrumental music from that
style and holds the philosophy; local emotion/gestures nudge WITHIN it.

Everything else in JD is fully offline. This module imports `google.genai`
lazily so importing it stays cheap and safe with no network / no key present.
"""

from __future__ import annotations

import json
import logging

import numpy as np

from . import config
from .models import GeminiDirective

logger = logging.getLogger(__name__)

# Self-contained instruction so Gemini knows EXACTLY our use case + output shape.
_DIRECT_PROMPT = (
    "You are the opening music director for a live, adaptive, INSTRUMENTAL music "
    "stream that reacts in real time to the person on camera — their emotion and "
    "micro-gestures (head shake/nod/tilt/turn, eye-roll). These are several photos "
    "of the SAME person and their surroundings, taken over ~5 seconds.\n\n"
    "Your output feeds an MRT2 / MusicCoCa real-time music model. It wants a SHORT "
    "comma-separated list of MusicCoCa STYLE TAGS — NOT prose, NOT lyrics. The music "
    "MUST be PURELY INSTRUMENTAL: NO vocals, NO singing, NO choir, NO 'vocal' tags. "
    "AVOID vocal-heavy genres (pop, soul, motown, R&B, indie pop, rap) because the "
    "model will sing gibberish — prefer instrumental genres (lo-fi, ambient, jazz, "
    "electronic, synthwave, classical, post-rock, downtempo). ALWAYS END the style "
    "with the tags 'instrumental, no vocals'. Include a tempo as a bpm tag. "
    'Example style: "warm lo-fi, soft rhodes, 72bpm, mellow, vinyl crackle, instrumental, no vocals".\n\n'
    "Look at the person (age vibe, apparent mood, energy level) and their "
    "environment (room, lighting, time-of-day feel). Choose music that genuinely "
    "SUITS them and their space and that gently lifts and supports them — never "
    "jarring, something that holds and steadies.\n\n"
    "Also write a one-line PHILOSOPHY: the consistent musical direction to hold for "
    "the whole session, so the stream does not wander. Gestures and emotions will "
    "nudge the music WITHIN this philosophy; it must not break.\n\n"
    "Respond with STRICT JSON, exactly these keys:\n"
    '  "observation": a one-line read of the person/scene (<= 20 words)\n'
    '  "philosophy":  the consistent direction to maintain (<= 15 words)\n'
    '  "style":       the MusicCoCa style-tag string (short, comma-separated, with a bpm)\n\n'
    "Example response:\n"
    '{"observation": "Young adult at a dim desk, tired but calm, soft evening light.", '
    '"philosophy": "keep it warm and grounding, evolve gently, never busy", '
    '"style": "warm lo-fi, soft rhodes, 72bpm, mellow, vinyl crackle, instrumental, no vocals"}'
)

_MAX_FRAMES = 5
_STYLE_MAX_CHARS = 120


def _loads(text: str | None) -> dict:
    """Parse Gemini JSON, stripping any ```/```json code fences first."""
    if not text:
        raise ValueError("empty response")
    text = text.strip()
    if text.startswith("```"):  # strip ``` / ```json fences line-wise
        text = "\n".join(
            ln for ln in text.splitlines() if not ln.strip().startswith("```")
        ).strip()
    return json.loads(text)


def _clean_style(style: str) -> str:
    """Clamp to a short tag string: collapse whitespace, strip quotes, cap length.

    Also GUARANTEES the music stays instrumental: if Gemini omitted a no-vocals
    cue, append "instrumental, no vocals" (MRT2 sings gibberish otherwise).
    """
    style = " ".join(str(style).split()).strip().strip('"').strip("'").strip()
    has_nv = "no vocal" in style.lower() or "instrumental" in style.lower()
    tag = ", instrumental, no vocals"
    budget = _STYLE_MAX_CHARS - (0 if has_nv else len(tag))
    if len(style) > budget:  # cap the base on a tag boundary, leave room for the tag
        style = style[:budget].rsplit(",", 1)[0].strip()
    if not has_nv:
        style = f"{style}{tag}"
    return style


class GeminiDirector:
    """Sends the 5-photo observe burst to Gemini → an opening GeminiDirective."""

    def __init__(self, api_key: str, model: str):
        from google import genai  # lazy: keep module import cheap / offline-safe

        self._client = genai.Client(api_key=api_key)
        self._model = model

    def direct(self, frames: list[np.ndarray]) -> GeminiDirective:
        """Watch the subject (frames) → opening music directive.

        Encodes up to ~5 frames as JPEG, asks Gemini for strict JSON
        {observation, philosophy, style}, and returns a "gemini" directive.
        On ANY error (network/parse/empty) returns a fallback directive built
        from STARTER_STYLE and logs a warning. NEVER raises.
        """
        try:
            import cv2
            from google.genai import types

            parts: list = [_DIRECT_PROMPT]
            for fr in frames[:_MAX_FRAMES]:
                ok, buf = cv2.imencode(".jpg", fr)
                if ok:
                    parts.append(
                        types.Part.from_bytes(
                            data=buf.tobytes(), mime_type="image/jpeg"
                        )
                    )
            if len(parts) == 1:
                raise RuntimeError("no frames to encode")

            resp = self._client.models.generate_content(
                model=self._model,
                contents=parts,
                config=types.GenerateContentConfig(
                    temperature=0.6, response_mime_type="application/json"
                ),
            )
            data = _loads(resp.text)

            style = _clean_style(data.get("style", "") or "")
            if not style:
                raise ValueError("no style in response")
            philosophy = " ".join(str(data.get("philosophy", "") or "").split()).strip()
            observation = " ".join(str(data.get("observation", "") or "").split()).strip()
            if not philosophy:
                philosophy = "hold a steady, comforting groove; evolve gently"

            return GeminiDirective(
                style=style,
                philosophy=philosophy,
                observation=observation or "Gemini directed the opening style.",
                source="gemini",
            )
        except Exception as exc:  # noqa: BLE001 — director must never raise
            reason = str(exc).splitlines()[0][:80] if str(exc) else type(exc).__name__
            logger.warning("Gemini director failed (%s) — using fallback.", reason)
            return GeminiDirective.fallback(
                config.STARTER_STYLE,
                observation=f"Gemini call failed: {reason}",
            )

    def update(self, frames: list[np.ndarray], philosophy: str = "") -> tuple[str, str]:
        """Re-read the webcam DURING streaming → an updated (style, situation).

        Called every few seconds by the engine with ~1 current frame. Encodes
        the latest frame(s) as JPEG, asks Gemini for the music that fits the
        person RIGHT NOW (kept within the session `philosophy` so it doesn't
        wander), and returns (style, situation). The style is run through
        `_clean_style`, which GUARANTEES the "instrumental, no vocals" tag.

        On ANY error (network/parse/empty/rate-limit) returns ("", "") and NEVER
        raises — the engine keeps the current music when the style is empty.
        """
        try:
            import cv2
            from google.genai import types

            philosophy = " ".join(str(philosophy or "").split()).strip()
            prompt = (
                "You are continuously directing INSTRUMENTAL music for the person "
                "on camera. Look at this snapshot of them RIGHT NOW. Read their "
                "current mood/energy and what's happening. Output music that fits "
                "the CURRENT moment. The music is PURELY INSTRUMENTAL — NO "
                "vocals/singing/lyrics, avoid vocal genres (pop/soul/indie/rap); "
                "prefer lo-fi/ambient/electronic/jazz/classical/post-rock; include "
                "a bpm; ALWAYS end with 'instrumental, no vocals'."
            )
            if philosophy:
                prompt += (
                    f" Stay within this session philosophy so it doesn't wander: "
                    f"'{philosophy}'."
                )
            prompt += (
                " Respond STRICT JSON, exactly keys: situation (<=12 words "
                "describing what you see/their mood right now), style (short "
                "comma-separated MusicCoCa tags incl bpm, instrumental no vocals)."
            )

            parts: list = [prompt]
            for fr in frames[:_MAX_FRAMES]:
                ok, buf = cv2.imencode(".jpg", fr)
                if ok:
                    parts.append(
                        types.Part.from_bytes(
                            data=buf.tobytes(), mime_type="image/jpeg"
                        )
                    )
            if len(parts) == 1:
                raise RuntimeError("no frames to encode")

            resp = self._client.models.generate_content(
                model=self._model,
                contents=parts,
                config=types.GenerateContentConfig(
                    temperature=0.5, response_mime_type="application/json"
                ),
            )
            data = _loads(resp.text)

            style = _clean_style(data.get("style", "") or "")
            situation = str(data.get("situation", "") or "").strip()[:120]
            return (style, situation)
        except Exception as exc:  # noqa: BLE001 — live update must never raise
            reason = str(exc).splitlines()[0][:80] if str(exc) else type(exc).__name__
            logger.debug("Gemini update failed (%s) — keeping current music.", reason)
            return ("", "")

    def describe_for_song(self, frames: list[np.ndarray]) -> tuple[str, str]:
        """Turn a webcam snapshot into a FULL-SONG brief for Suno → (style, lyrics).

        Unlike `direct`/`update` (which produce PURELY INSTRUMENTAL MRT2 styles),
        this is for the Suno music model: VOCALS AND LYRICS ARE WANTED. Encodes
        the snapshot frame(s) as JPEG, asks Gemini (high temperature for
        creativity) for strict JSON {style, lyrics}, and returns (style, lyrics).

        On ANY error (network/parse/empty) returns a safe default style with
        empty lyrics (Suno can still produce) and NEVER raises.
        """
        try:
            import cv2
            from google.genai import types

            prompt = (
                "You are a songwriter looking at a snapshot of a person and their "
                "surroundings. Capture this MOMENT as a short, FULLY-PRODUCED song "
                "for the Suno music model. Read their mood, energy, setting, and "
                "vibe. Output STRICT JSON, exactly:\n"
                '  "style": a Suno style/genre description (comma-separated tags: '
                "genre, mood, instrumentation, tempo, vocal type — VOCALS ARE "
                "WELCOME, this is a full song, e.g. 'warm indie folk-pop, gentle "
                "male vocals, acoustic guitar, 90bpm, hopeful')\n"
                '  "lyrics": short song lyrics (a verse + a chorus, ~6-10 lines) '
                "that reflect what you see and the person's apparent mood — "
                "uplifting and personal."
            )

            parts: list = [prompt]
            for fr in frames[:_MAX_FRAMES]:
                ok, buf = cv2.imencode(".jpg", fr)
                if ok:
                    parts.append(
                        types.Part.from_bytes(
                            data=buf.tobytes(), mime_type="image/jpeg"
                        )
                    )
            if len(parts) == 1:
                raise RuntimeError("no frames to encode")

            resp = self._client.models.generate_content(
                model=self._model,
                contents=parts,
                config=types.GenerateContentConfig(
                    temperature=0.9, response_mime_type="application/json"
                ),
            )
            data = _loads(resp.text)

            style = " ".join(str(data.get("style", "")).split())[:200]
            lyrics = str(data.get("lyrics", "")).strip()[:1200]
            if not style:
                raise ValueError("no style in response")
            return (style, lyrics)
        except Exception as exc:  # noqa: BLE001 — song brief must never raise
            reason = str(exc).splitlines()[0][:80] if str(exc) else type(exc).__name__
            logger.warning("Gemini song brief failed (%s) — using default.", reason)
            return ("warm uplifting indie pop, gentle vocals, acoustic, 90bpm", "")


def make_director() -> GeminiDirector | None:
    """Build a GeminiDirector, or None if no key (engine then uses fallback)."""
    if not config.GEMINI_API_KEY:
        logger.info("No Gemini key — director disabled (engine will use fallback).")
        return None
    try:
        return GeminiDirector(config.GEMINI_API_KEY, config.GEMINI_MODEL)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not init GeminiDirector (%s).", exc)
        return None

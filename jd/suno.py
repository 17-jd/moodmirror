"""Minimal Suno API client (stdlib only) for MoodMirror's full-song mode.

Async submit → poll → download, matching the Suno public API:
  POST /v0/audio        {lyrics, style, title}  (custom vocal) or {style, instrumental}
  GET  /v0/audio/{id}   → {status, audio_url, ...}

MoodMirror's Suno mode wants FULL produced songs WITH vocals, so the lyrics path
(`submit_song`) is primary; instrumental is only a fallback when no lyrics are given.

Ported from the proven moodjam client (tested against the live Suno API). The
`_UA` browser User-Agent bypasses Cloudflare's bot block (error 1010) on
api.suno.com. `generate()` and `wait()` accept an optional `on_status` callback so
the caller (e.g. the web dashboard) can surface progress: submitting → generating
→ downloading → (caller) playing. We download the resulting .m4a/.mp3 for playback.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from . import config

logger = logging.getLogger(__name__)

_TERMINAL = {"complete", "error"}
# api.suno.com is behind Cloudflare, which 403s (error 1010) bot-like User-Agents.
_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


class SunoError(RuntimeError):
    pass


def _notify(on_status: Callable[[str], None] | None, status: str) -> None:
    """Invoke an optional progress callback, swallowing any callback errors."""
    if on_status is None:
        return
    try:
        on_status(status)
    except Exception:  # never let a misbehaving callback break generation
        logger.debug("on_status callback raised for %r", status, exc_info=True)


class SunoClient:
    def __init__(self, api_key: str, base_url: str = config.SUNO_BASE_URL):
        self._key = api_key
        self._base = base_url.rstrip("/") + "/"

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        url = self._base + path.lstrip("/")
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Authorization", f"Bearer {self._key}")
        req.add_header("User-Agent", _UA)  # bypass Cloudflare 1010
        req.add_header("Accept", "application/json")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:200]
            raise SunoError(f"Suno {method} {path} → HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise SunoError(f"Suno {method} {path} failed: {exc.reason}") from exc

    def submit_song(self, lyrics: str, style: str, title: str) -> str:
        body = {"lyrics": lyrics, "style": style, "title": title}
        return str(self._request("POST", "v0/audio", body)["id"])

    def submit_instrumental(self, style: str, title: str) -> str:
        body = {"style": style, "title": title, "instrumental": True}
        return str(self._request("POST", "v0/audio", body)["id"])

    def poll(self, clip_id: str) -> dict:
        return self._request("GET", f"v0/audio/{clip_id}")

    def account_usage(self) -> dict:
        return self._request("GET", "v0/account/usage")

    def wait(
        self,
        clip_id: str,
        timeout: float = config.SUNO_TIMEOUT_S,
        interval: float = config.SUNO_POLL_S,
        on_status: Callable[[str], None] | None = None,
    ) -> dict:
        _notify(on_status, "generating")
        deadline = time.time() + timeout
        while time.time() < deadline:
            clip = self.poll(clip_id)
            if clip.get("status") in _TERMINAL:
                return clip
            time.sleep(interval)
        raise SunoError(f"Suno clip {clip_id} timed out after {timeout}s")

    def download(self, audio_url: str, dest: Path) -> Path:
        if not audio_url.startswith(("http://", "https://")):
            raise SunoError(f"Unexpected audio_url: {audio_url!r}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.time() + 120  # wall-clock cap so a stall can't hang forever
        dl_req = urllib.request.Request(audio_url, headers={"User-Agent": _UA})
        with urllib.request.urlopen(dl_req, timeout=30) as resp, dest.open("wb") as f:
            while chunk := resp.read(65536):
                if time.time() > deadline:
                    raise SunoError("Suno download exceeded 120s")
                f.write(chunk)
        return dest

    def generate(
        self,
        *,
        lyrics: str | None,
        style: str,
        title: str,
        dest_dir: Path,
        on_status: Callable[[str], None] | None = None,
    ) -> Path:
        """Full flow: submit (vocal if lyrics given, else instrumental) → wait → download.

        Progress is reported via the optional ``on_status`` callback:
        "submitting" → "generating" → "downloading". The caller marks "playing"
        once the returned audio file starts. Callback exceptions are swallowed.
        """
        _notify(on_status, "submitting")
        clip_id = (
            self.submit_song(lyrics, style, title)
            if lyrics
            else self.submit_instrumental(style, title)
        )
        logger.info("Suno submitted clip %s (%s)", clip_id, "vocal" if lyrics else "instrumental")
        clip = self.wait(clip_id, on_status=on_status)
        if clip.get("status") == "error" or not clip.get("audio_url"):
            raise SunoError(f"Suno generation failed: {clip.get('error') or 'no audio_url'}")
        _notify(on_status, "downloading")
        ext = "m4a" if clip["audio_url"].split("?")[0].endswith(".m4a") else "mp3"
        return self.download(clip["audio_url"], dest_dir / f"{clip_id}.{ext}")


def make_suno() -> SunoClient | None:
    if not config.SUNO_API_KEY:
        logger.info("No Suno key — song mode disabled.")
        return None
    return SunoClient(config.SUNO_API_KEY)

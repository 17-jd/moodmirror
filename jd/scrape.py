"""Robust URL scraper for MoodMirror.

Reads an EVENT web page and distills it into a compact context-text string
plus a representative image (as a BGR numpy array), so downstream Gemini can
turn the page into a song brief.

Public API:
    scrape_url(url) -> (context_text, image_bgr_or_None)

Design goals:
- NEVER raises: any failure returns ("", None) or partial text with None image.
- Cheap to import: requests / bs4 / PIL / cv2 are lazily imported inside the
  function, so `import jd.scrape` stays fast.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
_HEADERS = {
    "User-Agent": _USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/*;q=0.8,*/*;q=0.5",
    "Accept-Language": "en-US,en;q=0.9",
}
_PAGE_TIMEOUT = 12  # seconds
_IMAGE_TIMEOUT = 12  # seconds
_MAX_CONTEXT_CHARS = 1500
_MAX_BODY_TEXT_CHARS = 1200  # trimmed before we assemble, body is lowest priority
_MAX_IMAGE_DIM = 1024
_MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20 MB safety cap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _normalize_url(url: str) -> str:
    """Ensure the URL has an http(s) scheme."""
    url = (url or "").strip()
    if url and not re.match(r"^https?://", url, re.IGNORECASE):
        url = "https://" + url
    return url


def _collapse_ws(text: str) -> str:
    """Collapse all runs of whitespace into single spaces and strip."""
    return re.sub(r"\s+", " ", text or "").strip()


def _build_context_text(soup, page_url: str) -> str:
    """Extract a readable, prioritized context string from parsed HTML."""
    title = ""
    description = ""
    site_name = ""
    event_bits: list[str] = []
    headings: list[str] = []

    # --- <title> ---
    try:
        if soup.title and soup.title.string:
            title = _collapse_ws(soup.title.string)
    except Exception as exc:  # noqa: BLE001
        logger.debug("title extraction failed: %s", exc)

    # --- <meta> tags (description, OpenGraph, event:*) ---
    try:
        for meta in soup.find_all("meta"):
            name = (meta.get("name") or "").lower().strip()
            prop = (meta.get("property") or "").lower().strip()
            content = _collapse_ws(meta.get("content") or "")
            if not content:
                continue

            if name == "description" and not description:
                description = content
            elif prop == "og:title" and not title:
                title = content
            elif prop == "og:description" and not description:
                description = content
            elif prop == "og:site_name" and not site_name:
                site_name = content
            elif prop.startswith("event:"):
                label = prop.split(":", 1)[1].replace("_", " ").strip()
                event_bits.append(f"{label}: {content}" if label else content)
    except Exception as exc:  # noqa: BLE001
        logger.debug("meta extraction failed: %s", exc)

    # --- headings ---
    try:
        for tag in soup.find_all(["h1", "h2"]):
            h = _collapse_ws(tag.get_text(" "))
            if h and h not in headings:
                headings.append(h)
            if len(headings) >= 6:
                break
    except Exception as exc:  # noqa: BLE001
        logger.debug("heading extraction failed: %s", exc)

    # --- visible body text ---
    body_text = ""
    try:
        for junk in soup(["script", "style", "noscript", "nav", "header", "footer", "form", "aside"]):
            junk.decompose()
        raw = soup.get_text(" ")
        body_text = _collapse_ws(raw)[:_MAX_BODY_TEXT_CHARS]
    except Exception as exc:  # noqa: BLE001
        logger.debug("body text extraction failed: %s", exc)

    # --- assemble (priority order) ---
    parts: list[str] = []
    if title:
        parts.append(f"Title: {title}")
    if site_name:
        parts.append(f"Site: {site_name}")
    if description:
        parts.append(f"Description: {description}")
    if event_bits:
        parts.append("Event: " + " | ".join(event_bits))
    if headings:
        parts.append("Headings: " + " | ".join(headings))
    if body_text:
        parts.append(f"Text: {body_text}")
    if not parts and page_url:
        parts.append(f"URL: {page_url}")

    return "\n".join(parts)[:_MAX_CONTEXT_CHARS].strip()


def _find_image_url(soup, base_url: str) -> str:
    """Find a representative image URL: og:image first, else first large <img>."""
    try:
        from urllib.parse import urljoin
    except Exception:  # pragma: no cover
        urljoin = lambda b, u: u  # noqa: E731

    # og:image (also twitter:image as a fallback)
    try:
        for prop_key in ("property", "name"):
            for val in ("og:image", "og:image:url", "twitter:image"):
                meta = soup.find("meta", attrs={prop_key: val})
                if meta and meta.get("content"):
                    return urljoin(base_url, _collapse_ws(meta["content"]))
    except Exception as exc:  # noqa: BLE001
        logger.debug("og:image lookup failed: %s", exc)

    # First plausibly-large <img>
    try:
        best = ""
        best_score = -1
        for img in soup.find_all("img"):
            src = img.get("src") or img.get("data-src") or ""
            src = _collapse_ws(src)
            if not src or src.startswith("data:"):
                continue
            score = 0
            for attr in ("width", "height"):
                try:
                    score += int(re.sub(r"[^0-9]", "", str(img.get(attr) or "0")) or 0)
                except Exception:  # noqa: BLE001
                    pass
            if score > best_score:
                best_score = score
                best = src
        if best:
            return urljoin(base_url, best)
    except Exception as exc:  # noqa: BLE001
        logger.debug("<img> lookup failed: %s", exc)

    return ""


def _download_image_bgr(image_url: str):
    """Download an image URL and return it as a BGR numpy array, or None."""
    if not image_url:
        return None
    try:
        import io

        import cv2
        import numpy as np
        import requests
        from PIL import Image

        resp = requests.get(
            image_url, headers=_HEADERS, timeout=_IMAGE_TIMEOUT, allow_redirects=True, stream=True
        )
        resp.raise_for_status()

        content = resp.content
        if not content or len(content) > _MAX_IMAGE_BYTES:
            logger.debug("image skipped (empty or too large): %s", image_url)
            return None

        pil = Image.open(io.BytesIO(content))
        pil = pil.convert("RGB")

        # Resize if huge (preserve aspect ratio).
        w, h = pil.size
        longest = max(w, h)
        if longest > _MAX_IMAGE_DIM:
            scale = _MAX_IMAGE_DIM / float(longest)
            pil = pil.resize((max(1, int(w * scale)), max(1, int(h * scale))))

        rgb = np.array(pil)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            logger.debug("decoded image has unexpected shape: %r", getattr(rgb, "shape", None))
            return None

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return bgr
    except Exception as exc:  # noqa: BLE001
        logger.warning("image download/decode failed for %s: %s", image_url, exc)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def scrape_url(url: str) -> tuple[str, "np.ndarray | None"]:
    """Scrape an event page into (context_text, image_bgr).

    Args:
        url: Event page URL. If it lacks an http(s) scheme, "https://" is added.

    Returns:
        A tuple ``(context_text, image)`` where ``context_text`` is a readable,
        capped (~1500 char) summary string (possibly "" on failure) and
        ``image`` is a BGR ``numpy.ndarray`` (cv2 convention) or ``None``.

    This function NEVER raises; on any failure it returns ("", None) or a
    partial result with a None image.
    """
    page_url = _normalize_url(url)
    if not page_url:
        logger.warning("scrape_url: empty/invalid url")
        return ("", None)

    # --- Fetch the page ---
    try:
        import requests

        resp = requests.get(
            page_url, headers=_HEADERS, timeout=_PAGE_TIMEOUT, allow_redirects=True
        )
        resp.raise_for_status()
        html = resp.text
        final_url = str(getattr(resp, "url", page_url)) or page_url
    except Exception as exc:  # noqa: BLE001
        logger.warning("scrape_url: fetch failed for %s: %s", page_url, exc)
        return ("", None)

    # --- Parse ---
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
    except Exception as exc:  # noqa: BLE001
        logger.warning("scrape_url: parse failed for %s: %s", final_url, exc)
        return ("", None)

    # --- Image (find URL from the *un-decomposed* soup before text strips it) ---
    image_bgr = None
    try:
        image_url = _find_image_url(soup, final_url)
    except Exception as exc:  # noqa: BLE001
        logger.debug("scrape_url: image url discovery failed: %s", exc)
        image_url = ""

    # --- Context text (this mutates soup by decomposing junk tags) ---
    try:
        context_text = _build_context_text(soup, final_url)
    except Exception as exc:  # noqa: BLE001
        logger.warning("scrape_url: context build failed for %s: %s", final_url, exc)
        context_text = ""

    # --- Download image (after text extraction; independent of its success) ---
    try:
        image_bgr = _download_image_bgr(image_url)
    except Exception as exc:  # noqa: BLE001
        logger.warning("scrape_url: image fetch failed: %s", exc)
        image_bgr = None

    return (context_text, image_bgr)

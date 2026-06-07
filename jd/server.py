"""FastAPI dashboard server for JD (fully offline webcam-mood → live music).

Hosts the single-page dashboard, a small JSON API, and an MJPEG webcam feed.
JD is intentionally minimal: no login, no settings, no API keys, no skip/stop —
just a live mirror of the local engine's state and narration feed.

The :class:`~jd.engine.JDEngine` is injected via :func:`create_app`, which keeps
the routing layer testable without a real webcam or music model.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse

from .engine import JDEngine

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"

# Preview cadence for the MJPEG stream (~30 fps). The capture thread refreshes
# the underlying frame independently; this only paces how often we re-emit it.
_FRAME_INTERVAL_S = 0.03


def create_app(engine: JDEngine) -> FastAPI:
    """Build a FastAPI app wired to a specific :class:`JDEngine`.

    Args:
        engine: A (typically already started) engine instance to serve.

    Returns:
        A configured FastAPI application.
    """
    app = FastAPI(title="JD")

    @app.on_event("shutdown")
    def _on_shutdown() -> None:
        """Stop the engine cleanly when the ASGI server shuts down."""
        logger.info("Server shutting down — stopping engine.")
        engine.shutdown()

    @app.get("/")
    def index() -> FileResponse:
        """Serve the single-page dashboard."""
        index_path = STATIC_DIR / "index.html"
        if not index_path.is_file():
            raise HTTPException(status_code=404, detail="index.html not found")
        return FileResponse(index_path)

    @app.get("/static/{path:path}")
    def static_file(path: str) -> FileResponse:
        """Serve a static asset, guarding against path traversal."""
        candidate = (STATIC_DIR / path).resolve()
        # Ensure the resolved path is still inside STATIC_DIR.
        if STATIC_DIR not in candidate.parents and candidate != STATIC_DIR:
            raise HTTPException(status_code=404, detail="Not found")
        if not candidate.is_file():
            raise HTTPException(status_code=404, detail="Not found")
        return FileResponse(candidate)

    @app.get("/video")
    async def video() -> StreamingResponse:
        """Stream the live webcam preview as MJPEG (multipart/x-mixed-replace)."""

        async def _frames():
            while True:
                jpeg = engine.latest_frame_jpeg()
                if jpeg is not None:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                        + jpeg + b"\r\n"
                    )
                await asyncio.sleep(_FRAME_INTERVAL_S)

        return StreamingResponse(
            _frames(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/api/state")
    def get_state() -> dict:
        """Return the engine's current snapshot state."""
        return engine.get_state()

    @app.get("/api/feed")
    def get_feed(since: int = 0) -> dict:
        """Return narration events newer than ``since`` (a monotonic seq)."""
        return engine.feed_since(since)

    @app.get("/api/health")
    def health() -> dict:
        """Liveness probe."""
        return {"ok": True}

    @app.post("/api/start")
    def start() -> dict:
        """Begin a JD session (observe → direct → stream); idle until called.

        ``engine.begin_session()`` returns immediately (the 5s observe →
        Gemini → stream flow runs async in the engine). Errors are reported in
        the body with a 200 so the dashboard's fetch never throws.
        """
        try:
            engine.begin_session()
            return {"ok": True}
        except Exception as exc:  # noqa: BLE001 — surface, never raise to the UI
            logger.exception("engine.begin_session() failed")
            return {"ok": False, "detail": str(exc)[:140]}

    @app.post("/api/reset")
    def reset() -> dict:
        """Re-run the observe→direct flow for a (possibly) new person.

        ``engine.reset()`` returns immediately (the flow runs async in the
        engine, without stopping the music). Errors are reported in the body
        with a 200 so the dashboard's fetch never throws.
        """
        try:
            engine.reset()
            return {"ok": True}
        except Exception as exc:  # noqa: BLE001 — surface, never raise to the UI
            logger.exception("engine.reset() failed")
            return {"ok": False, "detail": str(exc)[:140]}

    @app.post("/api/gemini-interval")
    async def set_interval(request: Request) -> dict:
        """Live-adjust the continuous Gemini director cadence (seconds).

        Accepts the desired seconds as a JSON body ``{"seconds": <number>}`` OR a
        query param ``?seconds=<number>`` (JSON is tried first, then the query).
        The value is clamped to [1.0, 30.0] by the engine. Errors are reported in
        the body with a 200 so the dashboard's fetch never throws.
        """
        try:
            seconds: object | None = None
            try:
                body = await request.json()
                if isinstance(body, dict):
                    seconds = body.get("seconds")
            except Exception:  # noqa: BLE001 — no/invalid JSON → fall back to query
                seconds = None
            if seconds is None:
                seconds = request.query_params.get("seconds")
            if seconds is None:
                raise ValueError("missing 'seconds'")
            clamped = engine.set_gemini_interval(float(seconds))
            return {"ok": True, "seconds": clamped}
        except Exception as exc:  # noqa: BLE001 — surface, never raise to the UI
            logger.exception("engine.set_gemini_interval() failed")
            return {"ok": False, "detail": str(exc)[:140]}

    @app.post("/api/gemini-model")
    async def set_gemini_model(request: Request) -> dict:
        """Live-switch the Gemini model used by all director calls.

        Accepts the model as a JSON body ``{"model": "<name>"}`` OR a query param
        ``?model=<name>`` (JSON is tried first, then the query). The engine stores
        it and returns the stored model. Errors are reported in the body with a
        200 so the dashboard's fetch never throws.
        """
        try:
            model: object | None = None
            try:
                body = await request.json()
                if isinstance(body, dict):
                    model = body.get("model")
            except Exception:  # noqa: BLE001 — no/invalid JSON → fall back to query
                model = None
            if model is None:
                model = request.query_params.get("model")
            if model is None:
                raise ValueError("missing 'model'")
            stored = engine.set_gemini_model(str(model))
            return {"ok": True, "model": stored}
        except Exception as exc:  # noqa: BLE001 — surface, never raise to the UI
            logger.exception("engine.set_gemini_model() failed")
            return {"ok": False, "detail": str(exc)[:140]}

    @app.post("/api/mode")
    async def set_mode(request: Request) -> dict:
        """Switch the engine's generation mode ("local" or "suno").

        Accepts the desired mode as a JSON body ``{"mode": "local|suno"}`` OR a
        query param ``?mode=<value>`` (JSON is tried first, then the query). The
        engine validates the value and returns the stored mode. Errors are
        reported in the body with a 200 so the dashboard's fetch never throws.
        """
        try:
            mode: object | None = None
            try:
                body = await request.json()
                if isinstance(body, dict):
                    mode = body.get("mode")
            except Exception:  # noqa: BLE001 — no/invalid JSON → fall back to query
                mode = None
            if mode is None:
                mode = request.query_params.get("mode")
            if mode is None:
                raise ValueError("missing 'mode'")
            stored = engine.set_mode(str(mode))
            return {"ok": True, "mode": stored}
        except Exception as exc:  # noqa: BLE001 — surface, never raise to the UI
            logger.exception("engine.set_mode() failed")
            return {"ok": False, "detail": str(exc)[:140]}

    @app.post("/api/stop-song")
    def stop_song() -> dict:
        """Stop the currently playing song without ending the session.

        Errors are reported in the body with a 200 so the dashboard's fetch
        never throws.
        """
        try:
            engine.stop_song()
            return {"ok": True}
        except Exception as exc:  # noqa: BLE001 — surface, never raise to the UI
            logger.exception("engine.stop_song() failed")
            return {"ok": False, "detail": str(exc)[:140]}

    return app

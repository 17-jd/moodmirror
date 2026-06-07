"""CLI entry point: ``python -m jd`` launches the offline web dashboard.

Builds a :class:`~jd.engine.JDEngine` (webcam → local emotion → local MRT2
music), wires it into the FastAPI app, and serves it with uvicorn. The engine
is shut down cleanly on exit via both a FastAPI shutdown handler and a
try/finally backstop here.
"""

from __future__ import annotations

import argparse
import logging

import uvicorn

from . import config
from .engine import JDEngine
from .server import create_app

logger = logging.getLogger(__name__)


def main() -> None:
    """Parse CLI args, start the engine, and run the dashboard server."""
    parser = argparse.ArgumentParser(
        prog="jd",
        description="MoodMirror — webcam-mood → live-music dashboard (local Magenta + Suno).",
    )
    parser.add_argument("--host", default=config.SERVER_HOST)
    parser.add_argument("--port", type=int, default=config.SERVER_PORT)
    parser.add_argument("--webcam", type=int, default=config.WEBCAM_INDEX)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    engine = JDEngine(webcam_index=args.webcam)
    engine.start()
    app = create_app(engine)

    url = f"http://{args.host}:{args.port}"
    print(f"\n🪞  MoodMirror dashboard → {url}\n    (loading the music model takes a few seconds)\n")

    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        # Backstop in case the ASGI shutdown handler did not fire (e.g. startup error).
        engine.shutdown()


if __name__ == "__main__":
    main()

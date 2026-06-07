"""Music engine: TRUE LOW-LATENCY STREAMING with MRT2 (fully OFFLINE).

The old design generated a ~3s block per style and looped it, so a music/mood
change took ~6s to become audible. This engine instead STREAMS continuously and
reacts to a style change in ~0.5s.

MEASURED on this M1 (2026-06-07):
  * ``generate(style=emb, frames=N, state=state)`` makes N frames where 25
    frames = 1.0s of 48kHz stereo audio, and returns ``(waveform, new_state)``.
    Feeding ``state`` back in CONTINUES the music seamlessly — this is what
    keeps the musical "philosophy" intact across chunks (no hard cut).
  * Throughput ~27-30 steps/s ≈ 1.1x realtime (a 10-frame ≈ 0.4s chunk takes
    ~0.37s). So with a small buffer we can stream continuously.
  * ``embed_style(text, use_mapper=True)`` costs ~0.585s, so it MUST run off the
    audio/generation path — a dedicated embed thread precomputes it.
  * A style change (new embed + first chunk) reacts in ~0.4-0.8s.

Architecture (three cooperating threads + the audio callback):
  1. Audio callback: drains a thread-safe ring buffer (lock-light, never touches
    MLX). On underrun it FADES the last sample to silence (no DC click) and counts
    it (``underruns``). The stream is opened at MRT2's native 48kHz so CoreAudio
    does the 48k→device-rate conversion continuously — resampling each chunk
    separately in Python zero-padded the chunk edges → an audible "breaking" wobble.
  2. Generator thread: owns a persistent ``state``; while ahead-buffer <
    BUFFER_TARGET_S it generates ONE chunk with the CURRENT embedding and pushes it
    (no per-chunk resample). On a style change it FLUSHES the ring so the new style
    is heard within ~one chunk despite the buffer. Bigger chunks (CHUNK_FRAMES) keep
    throughput >1x realtime so the buffer stays full (gapless).
  3. Embed thread: waits on an event; when a new style is requested it computes
    the embedding and atomically swaps it into the slot the generator reads. The
    next generated chunk morphs to the new style while ``state`` keeps it musical.

Latency: a genuinely-new style stamps a monotonic time; when the FIRST chunk
built from its embedding is PUSHED to the ring buffer, the elapsed reaction time
is reported once via ``on_change_latency``.
"""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.signal import resample_poly

try:  # high-quality STATEFUL streaming resampler (no per-chunk edge artifacts)
    import soxr
except Exception:  # noqa: BLE001
    soxr = None  # type: ignore[assignment]

from . import config

logger = logging.getLogger(__name__)

_MRT_SR = config.SAMPLE_RATE  # 48000, MRT2's native output rate


def _clean_style(text: str) -> str:
    """MusicCoCa prefers short tag-style prompts: collapse whitespace, cap length."""
    return " ".join(text.split())[:120]


class _RingBuffer:
    """Thread-safe fixed-capacity ring buffer of stereo float32 samples."""

    def __init__(self, capacity: int):
        self._buf = np.zeros((capacity, 2), dtype=np.float32)
        self._cap = capacity
        self._write = 0
        self._avail = 0  # number of valid samples not yet read
        self._last = np.zeros((1, 2), dtype=np.float32)
        self._lock = threading.Lock()
        self.underruns = 0  # count of callbacks that couldn't be fully filled

    def available(self) -> int:
        with self._lock:
            return self._avail

    def free(self) -> int:
        with self._lock:
            return self._cap - self._avail

    def push(self, samples: np.ndarray) -> None:
        """Write samples; if they don't fit, drop the oldest (keep newest)."""
        n = len(samples)
        if n == 0:
            return
        if n > self._cap:  # only keep the most recent capacity samples
            samples = samples[-self._cap :]
            n = self._cap
        with self._lock:
            end = self._write + n
            if end <= self._cap:
                self._buf[self._write : end] = samples
            else:
                first = self._cap - self._write
                self._buf[self._write :] = samples[:first]
                self._buf[: n - first] = samples[first:]
            self._write = end % self._cap
            self._avail = min(self._cap, self._avail + n)

    def clear(self) -> None:
        """Drop all queued samples (used to flush stale audio on a style change)."""
        with self._lock:
            self._avail = 0
            self._write = 0

    def read(self, frames: int, out: np.ndarray) -> int:
        """Fill ``out`` (frames, 2) with the oldest samples. Returns count read."""
        with self._lock:
            n = min(frames, self._avail)
            if n > 0:
                start = (self._write - self._avail) % self._cap
                end = start + n
                if end <= self._cap:
                    out[:n] = self._buf[start:end]
                else:
                    first = self._cap - start
                    out[:first] = self._buf[start:]
                    out[first:n] = self._buf[: n - first]
                self._avail -= n
                self._last = out[n - 1 : n].copy()
            if n < frames:  # underrun: fade the last sample to silence (no DC click)
                gap = frames - n
                fade = np.linspace(1.0, 0.0, gap, dtype=np.float32)[:, None]
                out[n:frames] = self._last * fade
                self.underruns += 1
        return n


class MusicEngine:
    """Streaming MRT2 engine with measured, sub-second music-change latency."""

    def __init__(
        self,
        initial_style: str = "",
        session_dir: Path | None = None,
        on_change_latency: Callable[[float], None] | None = None,
    ):
        self._on_change_latency = on_change_latency
        self._session_dir = session_dir
        self._session_start = time.time()

        # Lifecycle events.
        self._running = threading.Event()
        self._ready = threading.Event()
        self._embed_wake = threading.Event()

        # Threads.
        self._gen_thread: threading.Thread | None = None
        self._embed_thread: threading.Thread | None = None

        # Model + device.
        self._mrt: Any = None
        self._out_sr = _MRT_SR  # output stream rate, set when the stream opens
        # Stream-start guard: the stream is opened in setup but may be started
        # later (after prefill). Ensure it starts exactly once.
        self._stream_start_lock = threading.Lock()
        self._stream_started = False

        # Style state (guarded by _style_lock).
        self._style_lock = threading.Lock()
        self._requested_style = initial_style.strip()
        self._current_style = ""  # style of the embedding now in the slot
        # The embedding slot the generator reads, plus the style it represents.
        self._embedding: Any = None
        self._embedding_style = ""
        # Whether the pending/applied style should FLUSH the ring (instant, for
        # gestures) or crossfade smoothly without flushing (continuous Gemini).
        self._requested_flush = True
        self._embedding_flush = True
        # Stateful 48k→device-rate resampler (soxr); None = output at 48k natively.
        self._resampler: Any = None
        # Latency tracking: monotonic stamp of a pending genuinely-new style.
        self._change_requested_at: float | None = None
        self._pending_latency_style = ""  # style whose first push we're timing
        self._latency_armed = False

        self._ring: _RingBuffer | None = None
        # Session save accumulator (device-rate stereo float32).
        self._sess_lock = threading.Lock()
        self._sess_blocks: list[np.ndarray] = []
        self._sess_idx = 0

        # External Suno song playback via afplay. While a song plays the audio
        # callback outputs SILENCE (muted) so the song is heard cleanly over the
        # top; the MRT2 generator keeps running so the stream resumes seamlessly
        # when the song ends (or is skipped) and the mute is cleared.
        self._muted = threading.Event()
        self._ext_lock = threading.Lock()
        self._external_proc: subprocess.Popen | None = None
        self._external_thread: threading.Thread | None = None
        self._skip_external = threading.Event()
        self._playing_external = threading.Event()
        self._external_label = ""

    # -- public interface ---------------------------------------------------
    @property
    def current_style(self) -> str:
        with self._style_lock:
            return self._current_style

    @property
    def underruns(self) -> int:
        """Audio-callback underruns since start (0 = perfectly gapless playback)."""
        return self._ring.underruns if self._ring is not None else 0

    def set_style(self, style: str, flush: bool = True) -> None:
        """Request a new style; takes effect within ~one chunk (~0.5s).

        ``flush=True`` (default, for instant GESTURE reactions) drops the queued
        audio so the new style is heard immediately. ``flush=False`` (for the
        continuous Gemini mood updates that arrive every few seconds) keeps the
        buffer full and lets the new style crossfade in via the persistent MRT2
        state — no flush means no periodic underrun/stutter.
        """
        style = style.strip()
        if not style:
            return
        with self._style_lock:
            if style != self._requested_style:
                self._requested_style = style
                self._requested_flush = flush
                self._change_requested_at = time.monotonic()
                self._pending_latency_style = style
                self._latency_armed = True
        self._embed_wake.set()

    # -- external (Suno) song playback -------------------------------------
    @property
    def playing_external(self) -> bool:
        """True while an external song is playing over the (muted) MRT2 stream."""
        return self._playing_external.is_set()

    @property
    def current_track(self) -> str:
        """External song label while playing; otherwise the current MRT2 style."""
        if self._playing_external.is_set() and self._external_label:
            return self._external_label
        return self.current_style

    def play_external_track(self, path: Path, label: str = "") -> None:
        """Play a song file over the top while the live MRT2 stream is muted.

        Returns immediately; the song plays on a daemon thread which mutes the
        audio callback, launches ``afplay``, waits for it to finish (or be
        skipped), then unmutes so the MRT2 stream resumes seamlessly. If a song
        is already playing, the new request is ignored (no double playback).
        """
        with self._ext_lock:
            if self._playing_external.is_set() or (
                self._external_thread is not None and self._external_thread.is_alive()
            ):
                logger.debug("external song already playing; ignoring %s", path)
                return
            self._external_thread = threading.Thread(
                target=self._play_external,
                args=(Path(path), label),
                name="music-external",
                daemon=True,
            )
            self._external_thread.start()

    def skip_external(self) -> None:
        """Stop the current external song (terminate afplay) and unmute."""
        self._skip_external.set()
        with self._ext_lock:
            proc = self._external_proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except Exception as exc:  # noqa: BLE001
                logger.debug("external terminate failed: %s", exc)

    def _play_external(self, path: Path, label: str) -> None:
        """Daemon worker: mute → afplay → wait/skip → unmute (always unmutes)."""
        self._skip_external.clear()
        self._external_label = label or path.name
        self._muted.set()  # callback now outputs silence; generator keeps running
        self._playing_external.set()
        logger.info("playing song: %s", self._external_label)
        proc: subprocess.Popen | None = None
        try:
            proc = subprocess.Popen(
                ["afplay", str(path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            with self._ext_lock:
                self._external_proc = proc
            while self._running.is_set() and proc.poll() is None:
                if self._skip_external.is_set():
                    proc.terminate()
                    break
                time.sleep(0.1)
        except FileNotFoundError:
            logger.error("afplay not found; cannot play song")
        except Exception as exc:  # noqa: BLE001
            logger.debug("external playback failed: %s", exc)
        finally:
            if proc is not None:
                if proc.poll() is None:
                    proc.terminate()
                try:
                    proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            with self._ext_lock:
                self._external_proc = None
            self._playing_external.clear()
            self._external_label = ""
            self._muted.clear()  # MRT2 stream resumes (callback drains ring again)

    def start(self) -> None:
        if self._gen_thread is not None:
            return
        self._running.set()
        self._embed_thread = threading.Thread(
            target=self._embed_loop, name="music-embed", daemon=True
        )
        self._gen_thread = threading.Thread(
            target=self._gen_loop, name="music-gen", daemon=True
        )
        self._embed_thread.start()
        self._gen_thread.start()

    def wait_until_ready(self, timeout: float = 120.0) -> bool:
        return self._ready.wait(timeout)

    def stop(self) -> None:
        self._running.clear()
        self.skip_external()  # terminate any external afplay and unmute
        self._embed_wake.set()  # unblock the embed thread so it can exit
        for t in (self._gen_thread, self._embed_thread):
            if t is not None:
                t.join(timeout=10.0)
        self._gen_thread = None
        self._embed_thread = None

    def is_alive(self) -> bool:
        return self._gen_thread is not None and self._gen_thread.is_alive()

    # -- audio callback (lock-light; never calls MLX) -----------------------
    def _callback(self, outdata, frames, time_info, status) -> None:  # noqa: ANN001
        if status:
            logger.debug("audio status: %s", status)
        if self._muted.is_set():  # an external song is playing → output silence
            outdata.fill(0.0)
            return
        ring = self._ring
        if ring is None:
            outdata.fill(0.0)
            return
        got = ring.read(frames, outdata)
        if got < frames:
            logger.debug("ring underrun: %d/%d", got, frames)

    # -- embed thread (off the audio/generation path) -----------------------
    def _embed_loop(self) -> None:
        # Block until the model is loaded before attempting to embed.
        while self._running.is_set() and self._mrt is None:
            self._running.wait(0.05)
        while self._running.is_set():
            self._embed_wake.wait(0.5)
            self._embed_wake.clear()
            with self._style_lock:
                want = self._requested_style
            if not want or want == self._embedding_style:
                continue
            try:
                emb = self._mrt.embed_style(_clean_style(want), use_mapper=True)
            except Exception as exc:  # noqa: BLE001
                logger.debug("embed_style failed: %s", exc)
                continue
            with self._style_lock:
                # Only commit if it's still the style the user wants.
                if want == self._requested_style:
                    self._embedding = emb
                    self._embedding_style = want
                    self._embedding_flush = self._requested_flush

    # -- generator thread (owns persistent state) ---------------------------
    def _gen_loop(self) -> None:
        try:
            self._load_model_and_stream()
        except Exception:  # noqa: BLE001
            logger.exception("MusicEngine failed to start")
            self._ready.set()
            self._running.clear()
            return

        state: list | None = None
        gen_style = ""  # style of the embedding the LAST chunk used
        target_ahead = config.BUFFER_TARGET_S * self._out_sr
        try:
            while self._running.is_set():
                with self._style_lock:
                    emb = self._embedding
                    emb_style = self._embedding_style
                    emb_flush = self._embedding_flush
                if emb is None:  # idle: nothing requested yet → stay silent
                    self._embed_wake.set()  # nudge embed thread if a style waits
                    self._running.wait(0.1)
                    continue
                assert self._ring is not None
                if self._ring.available() >= target_ahead:
                    # Ring is prefilled to target → safe to begin playback now so
                    # the callback never drains an empty buffer at startup. No-op
                    # after the first call (guarded).
                    self._start_stream()
                    self._running.wait(0.008)  # buffer full enough; don't spin
                    continue

                is_new = emb_style != gen_style
                try:
                    wav, state = self._mrt.generate(
                        style=emb, frames=config.CHUNK_FRAMES, state=state
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.debug("generate failed: %s", exc)
                    self._running.wait(0.05)
                    continue
                chunk = self._prepare_chunk(wav, ramp=is_new)
                if len(chunk) == 0:  # soxr priming emitted nothing yet → keep going
                    continue
                if is_new and emb_flush:  # flush only for instant (gesture) changes;
                    self._ring.clear()  # continuous Gemini updates crossfade (no stutter)
                self._ring.push(chunk)
                self._save_block(chunk)
                gen_style = emb_style
                if is_new:
                    with self._style_lock:
                        self._current_style = emb_style
                    self._maybe_report_latency(emb_style)
        finally:
            self._close_stream()
            self._flush_session()
            self._running.clear()

    # -- helpers ------------------------------------------------------------
    def _load_model_and_stream(self) -> None:
        import sounddevice as sd
        from magenta_rt import MagentaRT2Mlxfn

        logger.info("Loading MRT2 model '%s' (warming up)...", config.MRT2_MODEL)
        self._mrt = MagentaRT2Mlxfn(
            size=config.MRT2_MODEL,
            temperature=config.TEMPERATURE,
            top_k=config.TOP_K,
            cfg_musiccoca=config.CFG_MUSICCOCA,
        )
        self._embed_wake.set()  # let the embed thread start on any pending style

        try:
            dev_native = int(sd.query_devices(kind="output")["default_samplerate"])
        except Exception:  # noqa: BLE001
            dev_native = _MRT_SR

        # Output at the DEVICE'S NATIVE rate and resample 48k→device ourselves with
        # a stateful soxr stream. Outputting 48k to a 44.1k device made CoreAudio do
        # the SRC on the fly, which STUTTERED (verified: our pre-SRC audio was clean
        # but playback glitched). Native-rate out + soxr = no CoreAudio SRC, and
        # soxr's stateful stream joins chunks seamlessly (no per-chunk edge clicks).
        if soxr is not None and dev_native != _MRT_SR:
            self._out_sr = dev_native
            self._resampler = soxr.ResampleStream(
                _MRT_SR, dev_native, 2, dtype="float32", quality="HQ"
            )
        else:  # soxr missing (or device already 48k) → output 48k, let OS handle it
            self._out_sr = _MRT_SR
            self._resampler = None
        capacity = max(1, int(5.0 * self._out_sr))  # a few seconds of cushion

        def _open(rate: int) -> Any:
            return sd.OutputStream(
                samplerate=rate, channels=2, dtype="float32",
                callback=self._callback, blocksize=2048, latency="high",
            )

        self._ring = _RingBuffer(capacity)
        try:
            self._stream = _open(self._out_sr)
        except Exception as exc:  # noqa: BLE001 - native rate failed → try 48k raw
            logger.warning("%dHz output failed (%s); falling back to 48kHz raw.",
                           self._out_sr, exc)
            self._out_sr = _MRT_SR
            self._resampler = None
            self._ring = _RingBuffer(max(1, int(5.0 * self._out_sr)))
            self._stream = _open(self._out_sr)
        # When PREFILL_BEFORE_PLAY, the stream is OPENED (device ready) but NOT
        # started here — the generator prefills the ring to BUFFER_TARGET_S first,
        # then calls _start_stream() so playback never drains an empty buffer at
        # startup. When prefill is disabled, start immediately (old behavior).
        if not config.PREFILL_BEFORE_PLAY:
            self._start_stream()
        logger.info("Output stream rate: %d Hz (device native %d Hz)", self._out_sr, dev_native)
        self._ready.set()  # model loaded + stream open == ready (silent until styled)
        logger.info("MusicEngine ready (streaming; silent until first style).")

    def _start_stream(self) -> None:
        """Start the output stream exactly once (guarded; race-free)."""
        with self._stream_start_lock:
            if self._stream_started:
                return
            self._stream_started = True
        try:
            self._stream.start()
        except Exception as exc:  # noqa: BLE001
            logger.debug("stream start failed: %s", exc)

    def _prepare_chunk(self, wav: Any, ramp: bool) -> np.ndarray:
        seg = np.ascontiguousarray(wav.samples, dtype=np.float32)  # 48kHz (T, 2)
        if self._resampler is not None:  # stateful soxr stream → device rate, seamless
            seg = np.ascontiguousarray(self._resampler.resample_chunk(seg), dtype=np.float32)
        elif self._out_sr != _MRT_SR:  # fallback: stateless per-chunk resample
            seg = resample_poly(seg, self._out_sr, _MRT_SR, axis=0).astype(np.float32)
        if len(seg) == 0:  # soxr may emit nothing while priming its filter
            return seg
        if ramp:  # tiny amplitude ramp on the first chunk of a new style → no click
            n = min(len(seg), int(config.STYLE_SWAP_RAMP_S * self._out_sr))
            if n > 1:
                env = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
                seg[:n] *= env
        return seg

    def _maybe_report_latency(self, style: str) -> None:
        with self._style_lock:
            if not self._latency_armed or style != self._pending_latency_style:
                return
            requested_at = self._change_requested_at
            self._latency_armed = False
            self._change_requested_at = None
        if requested_at is None or self._on_change_latency is None:
            return
        elapsed = time.monotonic() - requested_at
        try:
            self._on_change_latency(elapsed)
        except Exception as exc:  # noqa: BLE001
            logger.debug("on_change_latency callback failed: %s", exc)

    # -- session save (best-effort) ----------------------------------------
    def _save_block(self, chunk: np.ndarray) -> None:
        # Disabled by default: writing a WAV inline on the generator thread every
        # ~10s stalled generation → a periodic audible hitch. When enabled, only
        # accumulate here (cheap, lock-light); the actual disk write happens once
        # in _flush_session() at stop, OFF the hot loop.
        if not config.SAVE_SESSION_AUDIO or self._session_dir is None:
            return
        with self._sess_lock:
            self._sess_blocks.append(chunk)

    def _write_session_file_locked(self) -> None:
        if not self._sess_blocks:
            return
        try:
            import soundfile as sf

            data = np.concatenate(self._sess_blocks, axis=0)
            self._sess_blocks = []
            self._sess_idx += 1
            elapsed = int(time.time() - self._session_start)
            sf.write(
                str(self._session_dir / f"stream_{self._sess_idx:03d}_{elapsed:04d}s.wav"),
                data,
                self._out_sr,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("session save failed: %s", exc)

    def _flush_session(self) -> None:
        if not config.SAVE_SESSION_AUDIO or self._session_dir is None:
            return
        with self._sess_lock:
            self._write_session_file_locked()

    def _close_stream(self) -> None:
        stream = getattr(self, "_stream", None)
        if stream is None:
            return
        try:
            stream.stop()
            stream.close()
        except Exception as exc:  # noqa: BLE001
            logger.debug("stream close failed: %s", exc)

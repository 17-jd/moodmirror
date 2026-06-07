"""Configuration, paths, and tunables for JD.

JD is FULLY OFFLINE: webcam → local emotion (OpenCV Haar + DeepFace subprocess)
→ local MRT2 music. No network calls at runtime (no Gemini, no Suno). All knobs
live here so the rest of the code reads cleanly.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # quiet TensorFlow logs (DeepFace)

# --- Paths -----------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
SESSIONS_DIR = DATA_DIR / "sessions"  # per-run audio archive (timestamped folders)
FACE_LANDMARKER_PATH = MODELS_DIR / "face_landmarker.task"  # MediaPipe (gesture subprocess)


def _load_env_file(path: Path) -> None:
    """Minimal .env loader (no dependency). Does not override real env vars."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env_file(BASE_DIR / ".env")

# --- Gemini (used ONLY for the initial 5-photo "director" prompt; NOT at runtime) ---
# The runtime music+emotion+gesture loop stays 100% offline. Gemini is consulted
# once per session start AND on each Reset to write the opening style + "philosophy".
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# --- Observe → Direct phase ------------------------------------------------
OBSERVE_SECONDS = 5.0  # watch the subject this long before the first directive
OBSERVE_PHOTOS = 5  # photos captured during the observe phase (≈1/sec) for Gemini
# Do NOT begin a session automatically on launch. The engine stays IDLE and SILENT
# until the user clicks "Start" on the dashboard (POST /api/start) → which snapshots
# the subject, sends to Gemini for analysis, and ONLY THEN starts the music.
AUTOSTART = False

# --- Continuous Gemini director --------------------------------------------
# After the opening directive, Gemini KEEPS re-reading the webcam every few seconds,
# understands the CURRENT situation/mood, and updates the (instrumental) music to
# match. This makes Gemini the live mood brain. Calls run on their own thread and
# never block audio; on error/rate-limit the current style is kept. The music is
# ALWAYS instrumental — no lyrics/vocals (enforced in the prompt + _clean_style).
CONTINUOUS_GEMINI = True
GEMINI_INTERVAL_S = 3.0  # snapshot → Gemini → update the music mood, every N seconds

# --- Generation MODE: local streaming model vs Suno full-song ---------------
# "local" = the live, instant MRT2 (Magenta RealTime) instrumental stream.
# "suno"  = snapshot → Gemini describes it → Suno SDK produces a FULL produced
#           song (with vocals) → auto-plays when ready (MRT2 mutes meanwhile).
# Start in SUNO mode by default: on Start, a full produced song is generated and
# the dashboard shows "Ready" + auto-plays it. A local instrumental stream plays as
# a BRIDGE during the ~30-60s generation so there's never dead silence.
MODE_DEFAULT = "suno"
LOCAL_MODEL_NAME = "Magenta RT2"  # display name for the local streaming model
SUNO_AUTOSTART = True  # in suno mode, auto-generate a song when the session begins

# --- Suno (full-song mode) -------------------------------------------------
SUNO_API_KEY = os.environ.get("SUNO_API_KEY")
SUNO_BASE_URL = os.environ.get("SUNO_BASE_URL", "https://api.suno.com/")
SUNO_TIMEOUT_S = 240.0  # max wait for a Suno song to finish generating
SUNO_POLL_S = 3.0  # poll interval while a song generates
# When continuous Gemini is on, the local DeepFace-emotion path no longer changes the
# music (Gemini owns mood); gestures still give instant overrides, emotion still shows
# on the dashboard + narration. Avoids two brains fighting over the style.

# --- Web dashboard ---------------------------------------------------------
SERVER_HOST = os.environ.get("JD_HOST", "127.0.0.1")
SERVER_PORT = int(os.environ.get("JD_PORT", "8000"))

# --- Music engine (MRT2, local/offline, TRUE STREAMING) --------------------
# MEASURED on this M1 (2026-06-07): mrt2_small generates ~27-30 steps/s ≈ 1.1x
# realtime, and a style change reacts in ~0.4-0.8s. So we STREAM: a generator
# thread produces small chunks with a PERSISTENT state (musical "philosophy"
# carries across chunks → never breaks) into a ring buffer; the audio callback
# drains it. A style change just swaps the style embedding for the NEXT chunk.
# This replaces the old "generate a 3s block then loop" design (which caused 6s).
MRT2_MODEL = os.environ.get("JD_MRT2_MODEL", "mrt2_small")
SAMPLE_RATE = 48_000  # MRT2 native output rate (resampled to device rate at runtime)
TEMPERATURE = 1.4  # a touch more variety/character
TOP_K = 50
CFG_MUSICCOCA = 4.5  # stronger style adherence → the prompt's "spice" comes through
# Chunk size is a throughput knob: bigger chunks amortise per-call overhead, so the
# generator sustains a higher realtime factor (measured: 25-frame ≈ 1.17x vs 10-frame
# ≈ 1.08x). Under live CPU contention (MediaPipe + DeepFace subprocesses) the small
# chunk starved the MLX generator → chronic underruns / "breaking". 20 frames ≈ 0.8s
# is the sweet spot: efficient enough to stay ahead, still a snappy reaction.
CHUNK_FRAMES = 20  # frames per streaming chunk; 25 frames = 1.0s, so 20 ≈ 0.8s audio
# Big cushion so a transient generator dip (thermal throttle, CPU spike, Bluetooth
# hiccup) can NEVER cause an audible gap — the callback always has seconds queued.
# Costs a bit of startup prefill + reaction latency, but playback stays glass-smooth.
BUFFER_TARGET_S = 3.5  # audio buffered ahead
BUFFER_MIN_S = 1.0  # if buffered audio drops below this, prioritise generation
STYLE_SWAP_RAMP_S = 0.12  # tiny amplitude ramp when a new style's chunk first plays
# Prefill the ring to BUFFER_TARGET_S BEFORE playback starts, so there's no
# startup underrun "stutter" while the buffer fills from empty.
PREFILL_BEFORE_PLAY = True
# Writing the session archive WAV happened ON the generator thread every ~10s,
# which stalled generation → a periodic audible hitch. Off by default now.
SAVE_SESSION_AUDIO = False
# Output the stream at MRT2's NATIVE 48kHz and let CoreAudio do the (continuous,
# gapless) conversion to the device rate. Resampling each 0.4s chunk separately in
# Python made resample_poly zero-pad each chunk's edges → an audible amplitude
# "wobble"/breaking at the chunk rate. Native-rate output removes that entirely.
OUTPUT_NATIVE_48K = True

# --- Vision (decoupled threads: capture fast, detect/emotion slower) --------
CAPTURE_INTERVAL_S = 0.04  # ~25fps (was 0.0 = CPU-burning busy loop that starved MLX)
DETECT_INTERVAL_S = 0.2  # Haar face + motion (~5 fps)
EMOTION_INTERVAL_S = 0.9  # DeepFace 7-class emotion (~1.1 fps; throttled to protect MLX)
SMILE_THRESHOLD = 0.35
WEBCAM_INDEX = int(os.environ.get("JD_WEBCAM", "0"))
# Motion: mean abs frame-diff above this (0..1, on a downscaled gray frame) = "moving".
MOTION_THRESHOLD = 0.018

# --- Gesture detection (MediaPipe FaceMesh, OWN subprocess) ----------------
# Isolated subprocess (like DeepFace) because MediaPipe + TensorFlow + MLX all
# abort if co-loaded. Gives head pose (yaw/pitch/roll deg), gaze, and discrete
# micro-gestures that NUDGE the music within Gemini's philosophy.
GESTURE_INTERVAL_S = 0.3  # ~3.3fps head-pose (throttled to protect MLX headroom)
SHAKE_YAW_DEG = 12.0  # |yaw| swing past this, oscillating = head SHAKE ("no")
NOD_PITCH_DEG = 10.0  # pitch swing past this, oscillating = NOD ("yes")
TILT_ROLL_DEG = 14.0  # |roll| past this = head TILT
ROTATE_YAW_DEG = 18.0  # sustained |yaw| past this = head turned/rotated away
EYEROLL_GAZE = 0.35  # normalized iris excursion past this = eye-roll / gaze shift
EYES_CLOSED_BLINK = 0.55  # MediaPipe eyeBlink blendshape avg past this = eyes shut
EYES_CLOSED_HOLD_S = 0.5  # eyes must stay shut this long to count (not a quick blink)
GESTURE_OSC_WINDOW_S = 1.2  # window to detect an oscillation (shake/nod)
GESTURE_COOLDOWN_S = 1.2  # min spacing between acting on the same discrete gesture

# --- Narrator / mood brain (local, deterministic, offline) -----------------
# How often the narrator re-evaluates mood and may switch music. Kept short so the
# system feels responsive; the narration text feed updates every tick regardless.
NARRATE_INTERVAL_S = 1.0
# A mood must persist this many seconds before we COMMIT a music change, so a single
# noisy frame doesn't thrash the music. Lower = snappier, higher = steadier.
MOOD_COMMIT_SECONDS = 2.0
RECENT_WINDOW = 60  # number of recent readings the narrator reasons over
FEED_MAXLEN = 200  # narration events kept in the ring buffer for the dashboard

# Moods that are "fine" → KEEP the current groove (don't disrupt a good thing).
POSITIVE_MOODS = frozenset({"happy", "neutral", "surprise"})
# Moods that warrant a lift → CHANGE the music.
LIFT_MOODS = frozenset({"sad", "angry", "fear", "disgust"})

# --- Style palette (offline; keyed by DeepFace's 7 emotions) ---------------
STARTER_STYLE = "warm calm lo-fi, soft piano, mellow, 70bpm, instrumental, no vocals"
MOOD_STYLES: dict[str, list[str]] = {
    "sad": [
        "uplifting indie pop, bright acoustic guitar, hopeful, gentle beat",
        "warm motown soul, playful horns, feel-good groove",
        "sunny tropical house, light pluck synths, breezy",
    ],
    "angry": [
        "calming ambient, soft warm pads, slow soothing tempo",
        "mellow acoustic, gentle fingerpicked guitar, peaceful",
    ],
    "fear": [
        "reassuring warm ambient, soft piano, safe and gentle",
        "slow comforting lo-fi, mellow rhodes, cozy",
    ],
    "disgust": [
        "clean bright pop, fresh light groove, palate-cleanser",
    ],
    "neutral": [
        "chill lo-fi hip hop, jazzy keys, head-nod groove",
        "bright funk, wah guitar, bouncy bass, playful",
        "dreamy synthwave, warm pads, gentle arpeggio",
    ],
    "surprise": [
        "energetic disco funk, four-on-the-floor, glittery strings",
        "upbeat electro pop, punchy drums, catchy hook",
    ],
    "happy": [
        "joyful pop, claps and bright synths, danceable, major key",
        "feel-good funk pop, slap bass, sunny brass stabs",
    ],
}

# ===========================================================================
# FIXED SOUND MAP — the explicit "if X then sound = Y" table the user asked for.
# Every detectable variable (micro-gesture + facial expression) maps to ONE fixed
# MRT2/MusicCoCa prompt. When that variable fires, the music engine streams THAT
# prompt (it REPLACES the current style, so the change is clearly audible — not a
# subtle suffix). The persistent MRT2 state keeps the transition musical.
# ===========================================================================

# Micro-gestures → a fixed, characterful sound. "shake" literally sounds shaky;
# "eyes_closed" goes deep and slow; etc. Tune these strings freely.
# NOTE: every prompt is kept INSTRUMENTAL. MRT2 is trained on real songs, so
# vocal-leaning genre words (pop/soul/indie/disco) make it sing vocal-like gibberish.
# Each prompt favours instrumental textures and ends with "instrumental, no vocals".
_NV = ", instrumental, no vocals"  # appended to every style to suppress singing
GESTURE_SOUND: dict[str, str] = {
    "shake": "wobbly detuned synth, heavy tremolo, wavering unstable pitch, woozy" + _NV,
    "nod": "punchy driving electronic groove, energetic drums, upbeat, bouncy bass" + _NV,
    "tilt": "curious playful melody, quirky pizzicato plucks, whimsical, light swing" + _NV,
    "rotate": "distant muffled ambient, heavy reverb, washed-out pads, far away" + _NV,
    # Head turned LEFT vs RIGHT → two clearly different textures (user: "left/right change music").
    "turn_left": "dreamy mellow filtered synth, spacious reverb, drifting, downtempo" + _NV,
    "turn_right": "bright energetic plucks, crisp arpeggio, forward motion, upbeat" + _NV,
    "eye_roll": "bright airy sparkle, shimmering high synths, breezy, light and fizzy" + _NV,
    "eyes_closed": "deep slow ambient drone, sub bass, dreamy, very low tempo, meditative" + _NV,
}

# Facial expressions (DeepFace 7-class) → a fixed sound. These are the steady-state
# mood response; gestures momentarily override on top. All INSTRUMENTAL (no vocal
# genres like pop/soul/indie that make MRT2 sing).
EMOTION_SOUND: dict[str, str] = {
    "happy": "joyful bright marimba and synth, major key, plucky, danceable, sunny" + _NV,
    "sad": "warm uplifting fingerpicked acoustic guitar, hopeful, gentle lift" + _NV,
    "angry": "calming soft ambient pads, slow soothing tempo, peaceful" + _NV,
    "fear": "reassuring warm ambient, soft piano, safe and gentle, grounding" + _NV,
    "surprise": "energetic electronic funk, glittery synths, lively, punchy" + _NV,
    "disgust": "clean fresh bright electronic, crisp light groove" + _NV,
    "neutral": "chill lo-fi hip hop beat, jazzy rhodes keys, relaxed head-nod groove" + _NV,
}

# How long a one-shot gesture's sound holds before the steady mood/base resumes.
GESTURE_HOLD_S = 6.0

# --- Simplified 3-class emotion (user: "just normal, happy, sad") ----------
# DeepFace gives 7 emotions; we bucket them into 3 for a clean, legible response.
EMOTION_BUCKET: dict[str, str] = {
    "happy": "happy",
    "surprise": "normal",
    "neutral": "normal",
    "sad": "sad",
    "angry": "sad",
    "fear": "sad",
    "disgust": "sad",
}
# Each 3-class mood → one fixed, RICH instrumental style for the LOCAL model.
# Spiced up with real instrumentation/genre detail so MRT2 has more to work with.
EMOTION3_SOUND: dict[str, str] = {
    "happy": "vibrant feel-good funk, bright marimba, wah guitar, punchy horns, claps, groovy bass, 108bpm" + _NV,
    "sad": "tender cinematic uplift, warm grand piano, lush strings, soft glockenspiel, hopeful swell, 76bpm" + _NV,
    "normal": "rich jazzy lo-fi, dusty rhodes, smooth chords, vinyl crackle, mellow boom-bap drums, 84bpm" + _NV,
}

# What drives the LOCAL model's music:
#  "emotion" = the 3-class face emotion (base mood) + left/right head turns (instant);
#              continuous Gemini still narrates the live "NOW" read but does NOT set the
#              style (so emotion + movement are clearly in control — the user's request).
#  "gemini"  = the continuous Gemini director sets the style (previous behavior).
LOCAL_DRIVER = "emotion"

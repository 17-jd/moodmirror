"""Out-of-process DeepFace emotion worker.

Run as a standalone script (NOT imported) so TensorFlow lives only in this child
process, isolated from the MLX music engine (the two abort if co-loaded).

Protocol: read one image path per line on stdin; write one JSON line per request
to stdout: {"emotion": <7-class>, "happy": <0..1>} or {"error": "..."}.
"""

import json
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
# Cap TensorFlow/BLAS threads so DeepFace can't grab every core and starve the MLX
# music generator in the parent process (a cause of local-audio stutter).
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
_VALID = {"happy", "sad", "angry", "surprise", "fear", "disgust", "neutral"}


def main() -> None:
    import cv2  # noqa: PLC0415
    from deepface import DeepFace  # noqa: PLC0415

    sys.stderr.write("emotion_worker: ready\n")
    sys.stderr.flush()
    for line in sys.stdin:
        path = line.strip()
        if not path:
            continue
        out: dict = {"emotion": "neutral", "happy": 0.0}
        try:
            img = cv2.imread(path)
            if img is not None:
                res = DeepFace.analyze(
                    img, actions=["emotion"], enforce_detection=False, silent=True
                )
                d = res[0] if isinstance(res, list) else res
                emo = str(d.get("dominant_emotion", "neutral"))
                out = {
                    "emotion": emo if emo in _VALID else "neutral",
                    "happy": float(d.get("emotion", {}).get("happy", 0.0)) / 100.0,
                }
        except Exception as exc:  # noqa: BLE001
            out = {"error": str(exc)[:120]}
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()

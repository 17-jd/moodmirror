"""Out-of-process MediaPipe FaceLandmarker gesture worker.

Run as a standalone script (NOT imported) so MediaPipe lives only in this child
process, isolated from the MLX music engine AND the DeepFace(TF) subprocess —
MediaPipe + TensorFlow + MLX all abort if co-loaded ("mutex lock failed").

Protocol: read one image path per line on stdin; write one JSON line per request
to stdout:
  {"face": true, "yaw": <deg>, "pitch": <deg>, "roll": <deg>, "gaze": <0..1>, "blink": <0..1>}
  {"face": false}                       (no face detected)
  {"error": "..."}                      (load / decode failure)

Only numpy / cv2 / mediapipe are imported here. NO TensorFlow, NO MLX.
"""

import json
import math
import os
import sys

os.environ.setdefault("GLOG_minloglevel", "2")  # quiet C++ glog (MediaPipe)
os.environ.setdefault("GLOG_logtostderr", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # belt-and-suspenders (no TF here)
# Cap threads so MediaPipe can't hog all cores away from the MLX music generator.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

# Well-known FaceMesh landmark indices.
_LEFT_EYE = (33, 133)  # outer, inner corner (subject's left)
_RIGHT_EYE = (362, 263)  # inner, outer corner (subject's right)
_LEFT_IRIS = (468, 469, 470, 471, 472)  # center + ring (refine_landmarks)
_RIGHT_IRIS = (473, 474, 475, 476, 477)
# Eyelid landmarks for EAR fallback: (left_corner, right_corner, top_lid, bottom_lid).
# FaceMesh contour points: horizontal corners + upper/lower lid midpoints per eye.
_LEFT_EAR = (33, 133, 159, 145)
_RIGHT_EAR = (362, 263, 386, 374)
# EAR (open) maps to closedness 0, EAR (closed) maps to closedness 1.
_EAR_OPEN = 0.28  # typical open-eye aspect ratio
_EAR_CLOSED = 0.10  # typical shut-eye aspect ratio


def _euler_deg(mat) -> tuple[float, float, float]:
    """Extract (yaw, pitch, roll) in degrees from a 4x4 transform's 3x3 rotation.

    Standard ZYX (Tait-Bryan) decomposition. yaw=Y, pitch=X, roll=Z.
    """
    import numpy as np  # noqa: PLC0415

    r = np.asarray(mat, dtype=np.float64)[:3, :3]
    sy = math.sqrt(r[0, 0] * r[0, 0] + r[1, 0] * r[1, 0])
    if sy > 1e-6:
        pitch = math.atan2(-r[2, 0], sy)
        roll = math.atan2(r[1, 0], r[0, 0])
        yaw = math.atan2(r[2, 1], r[2, 2])
    else:  # gimbal lock
        pitch = math.atan2(-r[2, 0], sy)
        roll = 0.0
        yaw = math.atan2(-r[1, 2], r[1, 1])
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


def _gaze(landmarks, n: int) -> float:
    """Horizontal iris offset from eye-center / eye-width, abs-averaged over eyes.

    Returns 0.0 when iris landmarks are absent (no refine/iris in the model).
    """
    if n <= max(_RIGHT_IRIS):
        return 0.0
    vals = []
    for (c0, c1), iris in ((_LEFT_EYE, _LEFT_IRIS), (_RIGHT_EYE, _RIGHT_IRIS)):
        x0, x1 = landmarks[c0].x, landmarks[c1].x
        width = abs(x1 - x0)
        if width < 1e-4:
            continue
        center = (x0 + x1) / 2.0
        iris_x = sum(landmarks[i].x for i in iris) / len(iris)
        vals.append(abs(iris_x - center) / width)
    if not vals:
        return 0.0
    return float(min(1.0, sum(vals) / len(vals)))


def _blink_blendshape(res) -> float | None:
    """Average of eyeBlinkLeft / eyeBlinkRight blendshape scores (0..1).

    Returns None if blendshapes are unavailable so the caller can fall back to EAR.
    """
    cats = getattr(res, "face_blendshapes", None)
    if not cats:
        return None
    scores = {}
    for c in cats[0]:
        name = getattr(c, "category_name", "") or ""
        if name in ("eyeBlinkLeft", "eyeBlinkRight"):
            scores[name] = float(getattr(c, "score", 0.0))
    if not scores:
        return None
    avg = sum(scores.values()) / len(scores)
    return float(min(1.0, max(0.0, avg)))


def _blink_ear(landmarks, n: int) -> float:
    """Closedness 0..1 from Eye-Aspect-Ratio (lower EAR = more closed), avg over eyes.

    Returns 0.0 when eyelid landmarks are absent.
    """
    if n <= max(max(_LEFT_EAR), max(_RIGHT_EAR)):
        return 0.0
    vals = []
    for c0, c1, top, bot in (_LEFT_EAR, _RIGHT_EAR):
        width = abs(landmarks[c1].x - landmarks[c0].x)
        if width < 1e-4:
            continue
        height = abs(landmarks[top].y - landmarks[bot].y)
        ear = height / width
        closedness = (_EAR_OPEN - ear) / (_EAR_OPEN - _EAR_CLOSED)
        vals.append(min(1.0, max(0.0, closedness)))
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def _blink(res, landmarks, n: int) -> float:
    """0..1 eye-closedness: blendshapes if present, else EAR fallback, else 0.0."""
    bs = _blink_blendshape(res)
    if bs is not None:
        return round(bs, 4)
    return round(_blink_ear(landmarks, n), 4)


def main() -> None:
    import cv2  # noqa: PLC0415
    import mediapipe as mp  # noqa: PLC0415
    from mediapipe.tasks import python as mp_python  # noqa: PLC0415
    from mediapipe.tasks.python import vision as mp_vision  # noqa: PLC0415

    model_path = os.environ.get("JD_FACE_LANDMARKER_PATH", "models/face_landmarker.task")
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
        # Lower thresholds (default 0.5) so a webcam face at normal distance/lighting
        # is reliably detected — otherwise pose stays 0 and no gestures fire.
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
        min_tracking_confidence=0.3,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )
    landmarker = mp_vision.FaceLandmarker.create_from_options(opts)

    sys.stderr.write("gesture_worker: ready\n")
    sys.stderr.flush()

    for line in sys.stdin:
        path = line.strip()
        if not path:
            continue
        out: dict = {"face": False}
        try:
            bgr = cv2.imread(path)
            if bgr is None:
                out = {"error": "imread returned None"}
            else:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                res = landmarker.detect(mp_image)
                if res.face_landmarks:
                    yaw = pitch = roll = 0.0
                    if res.facial_transformation_matrixes:
                        yaw, pitch, roll = _euler_deg(res.facial_transformation_matrixes[0])
                    lms = res.face_landmarks[0]
                    out = {
                        "face": True,
                        "yaw": round(yaw, 3),
                        "pitch": round(pitch, 3),
                        "roll": round(roll, 3),
                        "gaze": round(_gaze(lms, len(lms)), 4),
                        "blink": _blink(res, lms, len(lms)),
                    }
        except Exception as exc:  # noqa: BLE001
            out = {"error": str(exc)[:120]}
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()

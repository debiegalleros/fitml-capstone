"""Pose extraction, privacy crop, and 2D body-wrap compositing (locked
decision: MediaPipe keypoints + affine-warped transparent garment PNG on the
user photo — no 3D, no WebGL). Alpha channel gets a 1-2px Gaussian feather
before compositing so edges blend into the photo (locked decision).

Privacy crop (`crop_above_nose`) is opt-in via a checkbox on the upload
form. When checked, no face bounding box is ever computed. When left
unchecked (the default), `detect_face_bbox` locates the face once so the
generative try-on pipeline can re-paste it after generation — see
vision_tryon.py's `_paste_source_face` for the defense-in-depth this
supports."""
import json
import math
import os
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# MediaPipe >=0.10.30 removed the legacy `solutions` API — this module uses
# the Tasks API, which needs the model assets below (fetched on first run).
MP_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mp_models")
MP_MODEL_URLS = {
    "pose_landmarker_lite.task":
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "blaze_face_short_range.tflite":
        "https://storage.googleapis.com/mediapipe-models/face_detector/"
        "blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
}


def _model_path(name: str) -> str:
    os.makedirs(MP_MODELS_DIR, exist_ok=True)
    path = os.path.join(MP_MODELS_DIR, name)
    if not os.path.exists(path):
        urllib.request.urlretrieve(MP_MODEL_URLS[name], path)
    return path


_pose_landmarker = None
_face_detector = None


def _get_pose_landmarker():
    global _pose_landmarker
    if _pose_landmarker is None:
        _pose_landmarker = mp_vision.PoseLandmarker.create_from_options(
            mp_vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=_model_path("pose_landmarker_lite.task")),
                running_mode=mp_vision.RunningMode.IMAGE))
    return _pose_landmarker


def _get_face_detector():
    global _face_detector
    if _face_detector is None:
        _face_detector = mp_vision.FaceDetector.create_from_options(
            mp_vision.FaceDetectorOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=_model_path("blaze_face_short_range.tflite")),
                min_detection_confidence=0.4))
    return _face_detector


# Landmark indices (MediaPipe Pose)
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28

VISIBILITY_THRESHOLD = 0.5

# Garment *content* width (alpha bounding box, not PNG canvas) as a multiple
# of the anchor keypoint distance. MediaPipe shoulder/hip keypoints are joint
# centres — noticeably narrower than the visible body silhouette — so these
# factors were calibrated visually against a real full-body photo (see
# docs/phase8_notes.md provenance style): a fitted tank ~1.15x shoulder
# distance, sleeved tops ~1.3x, outerwear widest; hip keypoints span roughly
# half the visible hip width, hence the larger bottom factor.
WIDTH_FACTORS = {"top": 1.32, "dress": 1.32, "outerwear": 1.48, "bottom": 1.9}
CATEGORY_WIDTH_OVERRIDES = {"tank": 1.15, "sweater": 1.38}
# Vertical drop of the garment content top above the anchor line, as a
# fraction of the anchor distance (collar sits above the shoulder joints;
# a waistband sits above the hip joints).
TOP_OFFSET_FACTORS = {"top": 0.12, "dress": 0.12, "outerwear": 0.14, "bottom": 0.30}
CATEGORY_OFFSET_OVERRIDES = {"tank": 0.10}
TOP_CATEGORIES = {"tshirt", "tank", "polo", "blouse", "sweater"}
OUTERWEAR_CATEGORIES = {"jacket"}
BOTTOM_CATEGORIES = {"jeans", "skirt", "shorts", "slacks", "trousers"}


def _mp_image(image_bgr):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def extract_pose(image_bgr) -> dict:
    """Run MediaPipe Pose once; return pixel-space keypoints + coverage flag."""
    h, w = image_bgr.shape[:2]
    result = _get_pose_landmarker().detect(_mp_image(image_bgr))
    if not result.pose_landmarks:
        return {"ok": False}

    lm = result.pose_landmarks[0]
    def point(i):
        return {"x": lm[i].x * w, "y": lm[i].y * h, "v": lm[i].visibility}

    keypoints = {
        "l_shoulder": point(L_SHOULDER), "r_shoulder": point(R_SHOULDER),
        "l_elbow": point(L_ELBOW), "r_elbow": point(R_ELBOW),
        "l_wrist": point(L_WRIST), "r_wrist": point(R_WRIST),
        "l_hip": point(L_HIP), "r_hip": point(R_HIP),
        "l_knee": point(L_KNEE), "r_knee": point(R_KNEE),
        "l_ankle": point(L_ANKLE), "r_ankle": point(R_ANKLE),
    }
    shoulders_ok = (keypoints["l_shoulder"]["v"] > VISIBILITY_THRESHOLD
                    and keypoints["r_shoulder"]["v"] > VISIBILITY_THRESHOLD)
    hips_ok = (keypoints["l_hip"]["v"] > VISIBILITY_THRESHOLD
               and keypoints["r_hip"]["v"] > VISIBILITY_THRESHOLD)
    knees_ok = (keypoints["l_knee"]["v"] > VISIBILITY_THRESHOLD
                or keypoints["r_knee"]["v"] > VISIBILITY_THRESHOLD)

    coverage = "full_body" if (hips_ok and knees_ok) else "upper_body"
    return {"ok": shoulders_ok, "coverage": coverage, "keypoints": keypoints,
            "hips_visible": hips_ok}


# BlazeFace short-range detector keypoint order (MediaPipe Tasks
# FaceDetector): 0=right eye, 1=left eye, 2=nose tip, 3=mouth center,
# 4=right ear tragion, 5=left ear tragion. Verified empirically against a
# real photo (keypoint 2 sits ~50% down the face bounding box, between the
# eyes at ~15% and the mouth at ~65% — consistent with "nose tip").
NOSE_TIP_KEYPOINT_INDEX = 2


def detect_nose_y(image_bgr):
    """Detect the nose tip's y-coordinate (pixel space), for the privacy
    crop below. Returns None if no face/nose is confidently detected. This
    detection result is used once, to compute a single crop boundary —
    nothing about it (no bounding box, no landmarks, no face data of any
    kind) is ever persisted."""
    result = _get_face_detector().detect(_mp_image(image_bgr))
    if not result.detections:
        return None
    keypoints = result.detections[0].keypoints
    if not keypoints or len(keypoints) <= NOSE_TIP_KEYPOINT_INDEX:
        return None
    h = image_bgr.shape[0]
    return keypoints[NOSE_TIP_KEYPOINT_INDEX].y * h


def crop_above_nose(image_bgr):
    """Privacy crop (opt-in, via the crop_face checkbox on the upload form;
    replaces the earlier always-applied face-blur design): removes
    everything above the nose tip, before the photo is ever saved or
    processed further. When the caller applies this, no face pixels ever
    reach disk, pose extraction, or generative try-on. Returns the cropped
    image, or None if the nose could not be confidently detected — callers should
    ask for a re-upload rather than guess a crop boundary, since a wrong
    guess could leave part of the face exposed."""
    nose_y = detect_nose_y(image_bgr)
    if nose_y is None:
        return None
    y0 = max(0, int(nose_y))
    if y0 >= image_bgr.shape[0] - 1:
        return None
    return image_bgr[y0:, :, :]


def detect_face_bbox(image_bgr):
    """Detect the primary face's bounding box (x0, y0, x1, y1) in pixel
    space, expanded 25% to cover hairline/jaw, or None if no face is found.

    Called only on the crop_face-UNCHECKED upload path (see app.py) — the
    face is already fully present in that stored photo regardless, so this
    box carries no additional privacy exposure. It tells the try-on
    pipeline which region of that same stored photo to re-paste onto every
    generated render, as defense-in-depth against IDM-VTON occasionally
    regenerating a synthetic, incorrect face on full-body renders (see
    docs/genai_usage.md). Never called on the crop_face-checked path: once
    the face is cropped out, there is nothing left to protect, and
    computing this box would collect face-region data the crop-at-upload
    guarantee is meant to avoid entirely."""
    h, w = image_bgr.shape[:2]
    result = _get_face_detector().detect(_mp_image(image_bgr))
    if not result.detections:
        return None
    box = result.detections[0].bounding_box
    x0 = int(max(0, box.origin_x - 0.125 * box.width))
    y0 = int(max(0, box.origin_y - 0.125 * box.height))
    x1 = int(min(w, box.origin_x + 1.125 * box.width))
    y1 = int(min(h, box.origin_y + 1.125 * box.height))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def feather_alpha(rgba, sigma: float = 1.5):
    """1-2px Gaussian feather on the alpha channel (locked decision)."""
    alpha = rgba[:, :, 3].astype(np.float32)
    rgba = rgba.copy()
    rgba[:, :, 3] = cv2.GaussianBlur(alpha, (5, 5), sigma).clip(0, 255).astype(np.uint8)
    return rgba


def _garment_kind(category: str) -> str:
    if category in BOTTOM_CATEGORIES:
        return "bottom"
    if category == "dress":
        return "dress"
    if category in OUTERWEAR_CATEGORIES:
        return "outerwear"
    return "top"


def _content_crop(rgba):
    """Crop to the alpha bounding box so scaling is driven by the garment
    itself, not the PNG canvas padding (which varies per item)."""
    ys, xs = np.where(rgba[:, :, 3] > 10)
    if len(xs) == 0:
        return rgba
    return rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def composite(photo_bgr, garment_rgba, pose: dict, category: str,
              width_scale: float = 1.0, length_scale: float = 1.0):
    """Anchor the garment to pose keypoints (tops -> shoulders, bottoms ->
    hips), scale it (including the size-proportional factors), rotate it to
    the body axis, feather the alpha, and alpha-composite."""
    kp = pose["keypoints"]
    kind = _garment_kind(category)

    if kind == "bottom":
        a, b = kp["l_hip"], kp["r_hip"]
    else:
        a, b = kp["l_shoulder"], kp["r_shoulder"]

    anchor_dist = math.hypot(a["x"] - b["x"], a["y"] - b["y"])
    mid_x, mid_y = (a["x"] + b["x"]) / 2.0, (a["y"] + b["y"]) / 2.0
    angle_deg = math.degrees(math.atan2(a["y"] - b["y"], a["x"] - b["x"]))
    # shoulders/hips are ~horizontal; normalize the tilt around 180/0
    if angle_deg > 90:
        angle_deg -= 180
    elif angle_deg < -90:
        angle_deg += 180

    garment_rgba = _content_crop(garment_rgba)
    gh, gw = garment_rgba.shape[:2]
    width_factor = CATEGORY_WIDTH_OVERRIDES.get(category, WIDTH_FACTORS[kind])
    target_w = anchor_dist * width_factor * width_scale
    scale = target_w / gw
    new_w = max(1, int(gw * scale))
    new_h = max(1, int(gh * scale * (length_scale / width_scale)))
    garment = cv2.resize(garment_rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # rotate around the garment centre to match body tilt
    if abs(angle_deg) > 1.0:
        m = cv2.getRotationMatrix2D((new_w / 2, new_h / 2), -angle_deg, 1.0)
        cos, sin = abs(m[0, 0]), abs(m[0, 1])
        rw = int(new_h * sin + new_w * cos)
        rh = int(new_h * cos + new_w * sin)
        m[0, 2] += rw / 2 - new_w / 2
        m[1, 2] += rh / 2 - new_h / 2
        garment = cv2.warpAffine(garment, m, (rw, rh),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0, 0))
        new_w, new_h = rw, rh

    garment = feather_alpha(garment)

    # vertical placement: collar/strap top slightly above the shoulder line
    # for tops/dresses, waistband above the hip line for bottoms — offsets
    # are fractions of the anchor distance (body-relative), NOT the garment
    # height, so a long dress doesn't ride higher than a cropped top
    offset_factor = CATEGORY_OFFSET_OVERRIDES.get(category, TOP_OFFSET_FACTORS[kind])
    top_y = int(mid_y - offset_factor * anchor_dist)
    left_x = int(mid_x - new_w / 2)

    out = photo_bgr.copy()
    ph, pw = out.shape[:2]
    x0, y0 = max(0, left_x), max(0, top_y)
    x1, y1 = min(pw, left_x + new_w), min(ph, top_y + new_h)
    if x1 <= x0 or y1 <= y0:
        return out
    gx0, gy0 = x0 - left_x, y0 - top_y
    gslice = garment[gy0:gy0 + (y1 - y0), gx0:gx0 + (x1 - x0)]

    alpha = (gslice[:, :, 3:4].astype(np.float32)) / 255.0
    out[y0:y1, x0:x1] = (gslice[:, :, :3].astype(np.float32) * alpha
                         + out[y0:y1, x0:x1].astype(np.float32) * (1 - alpha)
                         ).astype(np.uint8)
    return out


def save_pose(session_dir: str, pose: dict):
    with open(os.path.join(session_dir, "pose.json"), "w") as f:
        json.dump(pose, f)


def load_pose(session_dir: str) -> dict:
    with open(os.path.join(session_dir, "pose.json")) as f:
        return json.load(f)

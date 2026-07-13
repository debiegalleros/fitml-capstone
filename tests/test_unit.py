"""Offline unit tests for the vision try-on pipeline (no API keys, no
network, no MediaPipe): mask geometry, pose-schema adapter, size-mismatch
fit context, engine category mapping, and the measurement-language guardrail.

Run: python -m pytest tests/test_unit.py
"""
import json
import re

import numpy as np
import pytest
from PIL import Image

import tryon
import vision_tryon as vt


# --------------------------------------------------------------- fixtures

def _kp(x, y, v=0.95):
    return {"x": x, "y": y, "v": v}


FULL_BODY_POSE_RAW = {
    "ok": True, "coverage": "full_body", "hips_visible": True,
    "keypoints": {
        # MediaPipe convention: left_* is the person's left = image-RIGHT
        "l_shoulder": _kp(420, 300), "r_shoulder": _kp(220, 300),
        "l_elbow": _kp(470, 450), "r_elbow": _kp(170, 450),
        "l_wrist": _kp(490, 600), "r_wrist": _kp(150, 600),
        "l_hip": _kp(390, 620), "r_hip": _kp(250, 620),
        "l_knee": _kp(380, 880), "r_knee": _kp(260, 880),
        "l_ankle": _kp(375, 1120), "r_ankle": _kp(265, 1120),
    },
}


@pytest.fixture
def full_pose(tmp_path, monkeypatch):
    """A session dir with the legacy pose.json schema + a photo."""
    import uuid
    monkeypatch.setattr(vt, "UPLOADS_DIR", str(tmp_path))
    sid = str(uuid.uuid4())
    d = tmp_path / sid
    d.mkdir()
    (d / "pose.json").write_text(json.dumps(FULL_BODY_POSE_RAW))
    Image.new("RGB", (640, 1200), (200, 200, 200)).save(d / "photo.jpg")
    return sid


# ------------------------------------------------------ pose schema adapter

def test_pose_adapter_maps_legacy_schema(full_pose):
    pose = vt._load_session_pose(full_pose)
    assert pose["photo_coverage"] == "full_body"
    assert pose["keypoints"]["left_shoulder"] == [420, 300]
    assert pose["keypoints"]["right_ankle"] == [265, 1120]


def test_pose_adapter_drops_low_visibility(full_pose, tmp_path):
    raw = json.loads(json.dumps(FULL_BODY_POSE_RAW))
    raw["keypoints"]["l_elbow"]["v"] = 0.1
    (tmp_path / full_pose / "pose.json").write_text(json.dumps(raw))
    pose = vt._load_session_pose(full_pose)
    assert pose["keypoints"]["left_elbow"] is None


# ------------------------------------------------------------ mask geometry

def _mask_extent(pose, category, sleeve="short_sleeve", size=(640, 1200)):
    import numpy as np
    photo = Image.new("RGB", size, (200, 200, 200))
    mask = vt._build_mask(photo, pose, category, sleeve_coverage=sleeve)
    a = np.array(mask)
    cols = (a > 200).any(axis=0)
    xs = cols.nonzero()[0]
    return xs.min(), xs.max(), (a > 200).mean()


def test_mask_pads_outward_despite_mediapipe_left_right(full_pose):
    """Regression: MediaPipe left_* is image-right, so unordered +/- padding
    NARROWED the mask (the sticker-look bug). Padding must extend past the
    shoulder keypoints on both sides."""
    pose = vt._load_session_pose(full_pose)
    x0, x1, _ = _mask_extent(pose, "tshirt")
    assert x0 < 220 and x1 > 420


def test_bottoms_mask_covers_visible_hip_width(full_pose):
    """Hip keypoints span ~half the visible hip width — the bottoms mask must
    extend well past them (55% pad) or old trousers survive at the hips."""
    pose = vt._load_session_pose(full_pose)
    x0, x1, _ = _mask_extent(pose, "jeans")
    hip_span = 390 - 250
    assert x0 <= 250 - 0.45 * hip_span
    assert x1 >= 390 + 0.45 * hip_span


def test_top_mask_works_without_hips():
    """Tops must composite on upper-body photos (hips below the crop)."""
    pose = {"photo_coverage": "upper_body", "keypoints": {
        "left_shoulder": [420, 300], "right_shoulder": [220, 300],
        "left_elbow": [470, 450], "right_elbow": [170, 450],
        "left_wrist": None, "right_wrist": None,
        "left_hip": None, "right_hip": None,
        "left_knee": None, "right_knee": None,
        "left_ankle": None, "right_ankle": None,
    }}
    _, _, white = _mask_extent(pose, "tshirt", size=(640, 700))
    assert white > 0.05  # non-degenerate mask


# --------------------------------------- lower-body standardization (SDXL)

def test_standardize_extends_top_mask_to_ankle(full_pose):
    """extend_to_ankle=True must reach past the hip line (into leg territory),
    not stop at the hip like a normal top mask."""
    pose = vt._load_session_pose(full_pose)
    photo = Image.new("RGB", (640, 1200), (200, 200, 200))
    normal = vt._build_mask(photo, pose, "tshirt", extend_to_ankle=False)
    extended = vt._build_mask(photo, pose, "tshirt", extend_to_ankle=True)
    import numpy as np
    normal_bottom = np.array(normal)[1100:1120, :].sum()
    extended_bottom = np.array(extended)[1100:1120, :].sum()
    assert extended_bottom > normal_bottom  # ankle region only covered when extended


def test_standardizes_lower_body_gating():
    full = {"photo_coverage": "full_body"}
    upper = {"photo_coverage": "upper_body"}
    # tops/outerwear on full-body photos: standardize
    assert vt._standardizes_lower_body("tshirt", full) is True
    assert vt._standardizes_lower_body("jacket", full) is True
    # no-op on upper-body photos — nothing below frame to mask
    assert vt._standardizes_lower_body("tshirt", upper) is False
    # never applies to bottoms-only or dress try-ons
    assert vt._standardizes_lower_body("jeans", full) is False
    assert vt._standardizes_lower_body("dress", full) is False


# ------------------------------- upper-body standardization (symmetric case)

def test_standardize_extends_bottoms_mask_to_shoulder(full_pose):
    """extend_to_shoulder=True must reach past the hip line upward (into
    torso/chest territory), not start at the hip like a normal bottoms mask."""
    pose = vt._load_session_pose(full_pose)
    photo = Image.new("RGB", (640, 1200), (200, 200, 200))
    normal = vt._build_mask(photo, pose, "jeans", extend_to_shoulder=False)
    extended = vt._build_mask(photo, pose, "jeans", extend_to_shoulder=True)
    import numpy as np
    normal_top = np.array(normal)[280:300, :].sum()   # near shoulder line
    extended_top = np.array(extended)[280:300, :].sum()
    assert extended_top > normal_top  # shoulder region only covered when extended


def test_standardizes_upper_body_disabled():
    """Case B is disabled pending future work (see docs/genai_usage.md) —
    IDM-VTON showed a boundary-bleed failure between the two chained calls
    in both orders tested, so the gate always returns False regardless of
    category or photo coverage. The underlying mask-geometry support
    (test_standardize_extends_bottoms_mask_to_shoulder above) stays
    available and tested for when this is revisited."""
    full = {"photo_coverage": "full_body"}
    upper = {"photo_coverage": "upper_body"}
    for cat in ("jeans", "skirt", "tshirt", "dress"):
        for pose in (full, upper):
            assert vt._standardizes_upper_body(cat, pose) is False


def test_standardize_legs_and_top_are_mutually_exclusive():
    full = {"photo_coverage": "full_body"}
    for cat in ("tshirt", "jacket", "jeans", "skirt", "dress"):
        assert not (vt._standardizes_lower_body(cat, full)
                    and vt._standardizes_upper_body(cat, full))


def test_upper_standard_reference_item_exists():
    import catalog
    item = catalog.get_item(vt.UPPER_STANDARD_ITEM_ID)
    assert item is not None
    assert item["category"] not in vt.LOWER_BODY_CATEGORIES
    assert item["category"] not in vt.FULL_LENGTH_CATEGORIES


def test_leggings_reference_item_exists():
    import catalog
    item = catalog.get_item(vt.LEGGINGS_ITEM_ID)
    assert item is not None
    assert item["category"] in vt.LOWER_BODY_CATEGORIES


# -------------------------------------------------------------- fit context

ITEM = {"size_range": "XS,S,M,L,XL", "category": "tshirt"}


def test_fit_context_true_to_size():
    ctx = vt._build_fit_context(ITEM, "M", {"recommended_size": "M"})
    assert "true to size" in ctx


def test_fit_context_oversized_two_steps():
    ctx = vt._build_fit_context(ITEM, "L", {"recommended_size": "S"})
    assert "2 sizes above" in ctx and "oversized" in ctx


def test_fit_context_undersized_bottoms():
    item = {"size_range": "XS,S,M,L,XL", "category": "jeans"}
    ctx = vt._build_fit_context(item, "S", {"recommended_size": "M"})
    assert "below the recommendation" in ctx and "waist" in ctx


def test_fit_context_falls_back_to_client_value():
    ctx = vt._build_fit_context(ITEM, "M", {"recommended_size": "M"},
                                client_fit_context="relaxed fit")
    assert ctx == "relaxed fit"


# ---------------------------------------------------- privacy crop-at-upload

def test_crop_above_nose_slices_at_detected_y(monkeypatch):
    """Mechanical crop math only — mocks detect_nose_y so this stays a fast,
    offline unit test (no MediaPipe model, no real face needed); the actual
    detection is verified live against real photos, not here."""
    monkeypatch.setattr(tryon, "detect_nose_y", lambda img: 300.0)
    image = np.zeros((600, 400, 3), dtype=np.uint8)
    image[300:, :, 0] = 255  # mark everything at/below the mocked nose line
    cropped = tryon.crop_above_nose(image)
    assert cropped.shape == (300, 400, 3)
    assert (cropped[:, :, 0] == 255).all()  # only the below-nose region survived


def test_crop_above_nose_returns_none_when_nose_undetectable(monkeypatch):
    """The fallback path: no guessed crop boundary when detection fails —
    the caller (app.py) turns this into a friendly re-upload prompt."""
    monkeypatch.setattr(tryon, "detect_nose_y", lambda img: None)
    image = np.zeros((600, 400, 3), dtype=np.uint8)
    assert tryon.crop_above_nose(image) is None


def test_crop_above_nose_returns_none_when_nose_at_frame_bottom(monkeypatch):
    """Guards against a degenerate zero-height crop if detection returns a
    y-coordinate at or past the bottom of the frame."""
    monkeypatch.setattr(tryon, "detect_nose_y", lambda img: 599.0)
    image = np.zeros((600, 400, 3), dtype=np.uint8)
    assert tryon.crop_above_nose(image) is None


# ---------------------------------------------------------- engine dispatch

def test_idm_category_mapping():
    assert vt._idm_category("jeans") == "lower_body"
    assert vt._idm_category("skirt") == "lower_body"
    assert vt._idm_category("dress") == "dresses"
    assert vt._idm_category("tshirt") == "upper_body"
    assert vt._idm_category("jacket") == "upper_body"


def test_bottom_categories_match_backend_config():
    from config import BOTTOM_CATEGORIES
    assert vt.LOWER_BODY_CATEGORIES == BOTTOM_CATEGORIES


def test_replicate_versions_pinned():
    # Reproducibility: both engines must run version-pinned by default.
    assert re.fullmatch(r"[0-9a-f]{64}", vt.IDM_VTON_VERSION)
    assert re.fullmatch(r"[0-9a-f]{64}", vt.SDXL_INPAINT_VERSION)


# ------------------------------------------------- measurement guardrail

def test_analyzer_prompt_bans_measurements():
    text = vt.ANALYZER_SYSTEM.lower()
    assert "never estimate" in text and "measurements" in text


def test_guardrail_regexes_catch_leaks():
    """The smoke test's measurement patterns must actually catch the failure
    modes they exist for."""
    import importlib.util
    import os
    spec = importlib.util.spec_from_file_location(
        "smoke", os.path.join(os.path.dirname(__file__), "test_vision_tryon.py"))
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)
    leaks = ["her bust measures 92 cm", "waist size around 70",
             "she has an hourglass figure", "petite frame", "about 36 inches"]
    for leak in leaks:
        assert any(re.search(p, leak) for p in smoke.MEASUREMENT_PATTERNS), leak
    clean = ("standing straight, arms relaxed, soft frontal lighting, "
             "plain white background, red cotton t-shirt, relaxed cut")
    assert not any(re.search(p, clean) for p in smoke.MEASUREMENT_PATTERNS)

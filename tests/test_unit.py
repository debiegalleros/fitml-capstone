"""Offline unit tests for the vision try-on pipeline (no API keys, no
network, no MediaPipe): mask geometry, pose-schema adapter, size-mismatch
fit context, engine category mapping, and the measurement-language guardrail.

Run: python -m pytest tests/test_unit.py
"""
import json
import re

import pytest
from PIL import Image

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

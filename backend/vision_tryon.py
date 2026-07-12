"""FitML — Vision AI Try-On module
================================
Generative try-on: a two-stage pipeline that replaces the 2D affine-warp
compositor as the primary renderer (the compositor stays as fallback).

  1. /api/generate-tryon-prompt   Claude Vision analyzes the shopper's photo +
                                  the garment image and produces a structured
                                  SDXL prompt (JSON).
  2. /api/generate-tryon-image    SDXL *inpainting* (via Replicate) regenerates
                                  only the masked torso/clothing region of the
                                  shopper's photo, guided by that prompt.
  3. /api/tryon                   Combined pipeline: analyze -> mask -> generate
                                  -> save -> return result. Drop-in replacement
                                  for the legacy /try-on endpoint.

Design notes:
  * Inpainting (not full generation) is deliberate: the mask covers only the
    clothing region derived from the MediaPipe pose keypoints extracted at
    upload time. Everything outside the mask — face (blurred), background —
    is preserved pixel-for-pixel.
  * The Claude Vision analyzer produces prompt JSON only. It never estimates
    measurements (hard-banned in the system prompt and asserted by the smoke
    test) and never feeds the graded size model.
  * Privacy guarantees carry over: results land in backend/uploads/{session}/,
    covered by the existing 24h purge; nothing new is keyed to a name/email.
"""

import base64
import gc
import io
import json
import os
import time
import uuid

import requests
from flask import Blueprint, jsonify, request
from PIL import Image, ImageDraw, ImageFilter

import catalog
from config import BOTTOM_CATEGORIES, UPLOADS_DIR

vision_tryon_bp = Blueprint("vision_tryon", __name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CLAUDE_VISION_MODEL = os.environ.get("CLAUDE_VISION_MODEL", "claude-sonnet-4-6")

REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN", "")
# SDXL inpainting on Replicate. Pin a version hash in env for reproducibility.
SDXL_INPAINT_MODEL = os.environ.get(
    "SDXL_INPAINT_MODEL", "stability-ai/stable-diffusion-inpainting"
)
SDXL_INPAINT_VERSION = os.environ.get("SDXL_INPAINT_VERSION", "")  # optional pin

REPLICATE_POLL_INTERVAL_S = 2
REPLICATE_TIMEOUT_S = 120

# Garment categories that move the mask off the torso anchor
LOWER_BODY_CATEGORIES = BOTTOM_CATEGORIES
FULL_LENGTH_CATEGORIES = {"dress"}

VISIBILITY_THRESHOLD = 0.5  # same bar tryon.extract_pose uses


# ---------------------------------------------------------------------------
# Helpers — session assets
# ---------------------------------------------------------------------------

def _session_dir(session_id: str) -> str:
    # session_id is a UUID minted by /upload-profile; reject path tricks
    safe = str(uuid.UUID(session_id))
    return os.path.join(UPLOADS_DIR, safe)


def _load_session_photo(session_id: str) -> Image.Image:
    d = _session_dir(session_id)
    for name in ("photo_blurred.png", "photo.png", "photo.jpg"):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return Image.open(p).convert("RGB")
    raise FileNotFoundError("No profile photo found for this session.")


# tryon.save_pose writes {"keypoints": {"l_shoulder": {"x","y","v"}, ...},
# "coverage": ...}; the mask builder wants {"keypoints": {"left_shoulder":
# [x, y] | None, ...}, "photo_coverage": ...}. Normalize here so the legacy
# compositor's format stays untouched.
_POSE_KEY_MAP = {
    "l_shoulder": "left_shoulder", "r_shoulder": "right_shoulder",
    "l_elbow": "left_elbow", "r_elbow": "right_elbow",
    "l_wrist": "left_wrist", "r_wrist": "right_wrist",
    "l_hip": "left_hip", "r_hip": "right_hip",
    "l_knee": "left_knee", "r_knee": "right_knee",
    "l_ankle": "left_ankle", "r_ankle": "right_ankle",
}


def _load_session_pose(session_id: str) -> dict:
    p = os.path.join(_session_dir(session_id), "pose.json")
    if not os.path.exists(p):
        raise FileNotFoundError("No pose data for this session (re-upload photo).")
    with open(p) as f:
        raw = json.load(f)
    keypoints = {}
    for short, long in _POSE_KEY_MAP.items():
        pt = raw.get("keypoints", {}).get(short)
        if pt and pt.get("v", 0) > VISIBILITY_THRESHOLD:
            keypoints[long] = [pt["x"], pt["y"]]
        else:
            keypoints[long] = None
    return {"keypoints": keypoints, "photo_coverage": raw.get("coverage")}


def _load_garment(item_id: str, color_variant: str | None = None) -> tuple[Image.Image, dict]:
    """Returns (garment image, metadata row) from the catalog.

    Base item -> the on-model `photo` (richer signal for Vision analysis);
    hue variant -> the recolored cutout PNG, since no on-model photo exists
    for generated variants."""
    item = catalog.get_item(item_id)
    if item is None:
        raise FileNotFoundError(f"Unknown item_id {item_id!r}")

    from config import CATALOG_DIR
    if color_variant and color_variant.strip().lower() != item["color"].lower():
        path = catalog.garment_png_path(item, color_variant)
    else:
        path = os.path.join(CATALOG_DIR, item["photo"])
    img = Image.open(path).convert("RGB")
    return img, item


def _img_to_b64(img: Image.Image, fmt: str = "JPEG", max_side: int = 1024) -> str:
    """Downscale (API cost/latency) and base64-encode."""
    im = img.copy()
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, format=fmt, quality=90)
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Stage 1 — Claude Vision analyzer
# ---------------------------------------------------------------------------

ANALYZER_SYSTEM = """You are a fashion try-on prompt engineer. You will receive
two images: IMAGE 1 is a shopper's photo (face intentionally blurred), IMAGE 2
is a garment product photo. Respond with ONLY a JSON object, no markdown fences,
no preamble, with exactly these keys:

{
  "person": {
    "pose": "...",            // body orientation, arm position, stance
    "framing": "...",         // full-body / upper-body, camera angle
    "lighting": "...",        // direction, softness, color temperature
    "background": "..."       // brief description
  },
  "garment": {
    "type": "...",            // e.g. cable-knit sweater
    "color": "...",           // precise color name
    "fabric": "...",          // apparent material and texture
    "cut": "...",             // fit silhouette: fitted / relaxed / oversized
    "details": "...",         // neckline, sleeves, hem, patterns
    "sleeve_coverage": "..."  // one of: sleeveless / short_sleeve / long_sleeve / n_a
  },
  "prompt": "...",            // ONE SDXL inpainting prompt: the person WEARING
                              // the garment as their actual clothing — like a
                              // mirror photo of themselves in it. The garment
                              // fully replaces whatever they were wearing in
                              // that region; skin shows where the garment
                              // doesn't cover (neckline, arms). Match their
                              // pose, lighting and framing. Photorealistic,
                              // candid mirror-selfie style. <= 60 words.
  "negative_prompt": "..."    // MUST include: original clothing visible,
                              // layered over other clothes, double collar,
                              // undershirt peeking out, garment floating over
                              // clothing — plus: extra limbs, warped hands,
                              // changed face, changed background, text, logos
}

RULES: Never estimate or mention body measurements. Never describe the person's
body shape or size. Describe only pose, framing, lighting, background, and the
garment."""


def _call_claude_vision(person_img: Image.Image, garment_img: Image.Image,
                        size_label: str, fit_context: str) -> dict:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set.")

    body = {
        "model": CLAUDE_VISION_MODEL,
        "max_tokens": 1000,
        "system": ANALYZER_SYSTEM,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64",
                 "media_type": "image/jpeg", "data": _img_to_b64(person_img)}},
                {"type": "image", "source": {"type": "base64",
                 "media_type": "image/jpeg", "data": _img_to_b64(garment_img)}},
                {"type": "text", "text":
                    f"Selected size: {size_label}. Fit context from the sizing "
                    f"model (read-only, for prompt wording like 'relaxed fit' "
                    f"vs 'close fit'): {fit_context}. Produce the JSON."},
            ],
        }],
    }
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json=body,
        timeout=60,
    )
    r.raise_for_status()
    text = "".join(b.get("text", "") for b in r.json()["content"])
    text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(text)


# ---------------------------------------------------------------------------
# Stage 1.5 — clothing-region mask from pose keypoints
# ---------------------------------------------------------------------------

def _build_mask(photo: Image.Image, pose: dict, category: str,
                sleeve_coverage: str = "n_a") -> Image.Image:
    """White = regenerate, black = preserve.

    Mirror-worn rule: the mask must cover EVERYTHING the shopper is currently
    wearing in the garment's region, or remnants of their real clothes survive
    at the edges and the result reads as a sticker instead of worn clothing.
    So: torso polygon is raised toward the neckline, padded generously, and for
    sleeved garments the arms (shoulder->elbow->wrist) are masked too."""
    w, h = photo.size
    kp = pose["keypoints"]

    def pt(name, fallback=None):
        v = kp.get(name) or fallback
        if v is None:
            raise KeyError(f"Missing keypoint {name}")
        return (float(v[0]), float(v[1]))

    ls, rs = pt("left_shoulder"), pt("right_shoulder")
    lh, rh = pt("left_hip"), pt("right_hip")

    cat = (category or "").lower()
    if cat in FULL_LENGTH_CATEGORIES:
        lb = kp.get("left_knee") or [lh[0], min(h - 1, lh[1] + (lh[1] - ls[1]))]
        rb = kp.get("right_knee") or [rh[0], min(h - 1, rh[1] + (rh[1] - rs[1]))]
        top = (ls, rs)
        bottom = ((lb[0], lb[1]), (rb[0], rb[1]))
    elif cat in LOWER_BODY_CATEGORIES:
        la = kp.get("left_ankle") or [lh[0], h - 1]
        ra = kp.get("right_ankle") or [rh[0], h - 1]
        top = (lh, rh)
        bottom = ((la[0], la[1]), (ra[0], ra[1]))
    else:  # tops, outerwear
        top = (ls, rs)
        bottom = (lh, rh)

    # MediaPipe "left_*" is the person's left = image-right on a front-facing
    # photo, so order each pair by x: top[0]/bottom[0] must be the image-left
    # side or the +/- padding below *narrows* the mask instead of widening it.
    if top[0][0] > top[1][0]:
        top = (top[1], top[0])
        bottom = (bottom[1], bottom[0])

    # Generous padding so no old clothing survives at the mask boundary:
    # ~25% of shoulder width horizontally, and for tops the mask is raised
    # ~18% of shoulder width above the shoulder line to cover collars and
    # necklines of whatever the shopper is currently wearing.
    span = abs(top[0][0] - top[1][0]) or w * 0.25
    pad_x = span * 0.25
    pad_y = abs(bottom[0][1] - top[0][1]) * 0.08
    neck_raise = span * 0.18 if cat not in LOWER_BODY_CATEGORIES else 0

    def clamp(p):
        return (min(max(p[0], 0), w - 1), min(max(p[1], 0), h - 1))

    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    poly = [clamp(p) for p in [
        (top[0][0] - pad_x, top[0][1] - pad_y - neck_raise),
        (top[1][0] + pad_x, top[1][1] - pad_y - neck_raise),
        (bottom[1][0] + pad_x, bottom[1][1] + pad_y),
        (bottom[0][0] - pad_x, bottom[0][1] + pad_y),
    ]]
    draw.polygon(poly, fill=255)

    # Arms: mask them whenever the new garment has sleeves OR the mask would
    # otherwise leave the shopper's current sleeves visible. Cheap and safe:
    # mask arms for all upper-body garments; the prompt regenerates bare skin
    # for sleeveless garments and fabric for sleeved ones.
    if cat not in LOWER_BODY_CATEGORIES:
        arm_w = max(6, span * 0.28)
        # long sleeves -> mask to wrist; otherwise to elbow (covers most
        # currently-worn sleeves without touching hands)
        to_wrist = sleeve_coverage == "long_sleeve"
        for side in ("left", "right"):
            sh = kp.get(f"{side}_shoulder")
            el = kp.get(f"{side}_elbow")
            wr = kp.get(f"{side}_wrist")
            if not sh:
                continue
            seg_end = (wr if to_wrist and wr else el) or None
            if seg_end is None:
                continue
            pts = [sh, el, seg_end] if (to_wrist and el and wr) else [sh, seg_end]
            pts = [p for p in pts if p]
            for a, b in zip(pts, pts[1:]):
                draw.line([clamp(a), clamp(b)], fill=255, width=int(arm_w))
                draw.ellipse([clamp((b[0] - arm_w / 2, b[1] - arm_w / 2)),
                              clamp((b[0] + arm_w / 2, b[1] + arm_w / 2))], fill=255)

    return mask.filter(ImageFilter.GaussianBlur(radius=max(2, int(w * 0.01))))


# ---------------------------------------------------------------------------
# Stage 2 — SDXL inpainting via Replicate
# ---------------------------------------------------------------------------

def _replicate_inpaint(photo: Image.Image, mask: Image.Image,
                       prompt: str, negative_prompt: str, seed: int = 42) -> Image.Image:
    if not REPLICATE_API_TOKEN:
        raise RuntimeError("REPLICATE_API_TOKEN is not set.")

    def data_uri(img, mode="RGB"):
        buf = io.BytesIO()
        img.convert(mode).save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    payload_input = {
        "image": data_uri(photo),
        "mask": data_uri(mask, "L"),
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_inference_steps": 30,
        "guidance_scale": 7.5,
        "seed": seed,
    }

    headers = {"Authorization": f"Bearer {REPLICATE_API_TOKEN}",
               "Content-Type": "application/json"}

    if SDXL_INPAINT_VERSION:
        create_url = "https://api.replicate.com/v1/predictions"
        body = {"version": SDXL_INPAINT_VERSION, "input": payload_input}
    else:
        create_url = f"https://api.replicate.com/v1/models/{SDXL_INPAINT_MODEL}/predictions"
        body = {"input": payload_input}

    r = requests.post(create_url, headers=headers, json=body, timeout=30)
    r.raise_for_status()
    pred = r.json()

    deadline = time.time() + REPLICATE_TIMEOUT_S
    while pred["status"] not in ("succeeded", "failed", "canceled"):
        if time.time() > deadline:
            raise TimeoutError("SDXL generation timed out.")
        time.sleep(REPLICATE_POLL_INTERVAL_S)
        pred = requests.get(pred["urls"]["get"], headers=headers, timeout=30).json()

    if pred["status"] != "succeeded":
        raise RuntimeError(f"SDXL generation failed: {pred.get('error')}")

    out = pred["output"]
    url = out[0] if isinstance(out, list) else out
    img_bytes = requests.get(url, timeout=60).content
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


# ---------------------------------------------------------------------------
# Endpoint 1: /api/generate-tryon-prompt
# ---------------------------------------------------------------------------

@vision_tryon_bp.route("/api/generate-tryon-prompt", methods=["POST"])
def generate_tryon_prompt():
    """Body: { session_id, item_id, size, fit_context?, color_variant? }"""
    data = request.get_json(force=True)
    try:
        photo = _load_session_photo(data["session_id"])
        garment, meta = _load_garment(data["item_id"], data.get("color_variant"))
        analysis = _call_claude_vision(
            photo, garment,
            size_label=data.get("size", meta.get("size_range", "M")),
            fit_context=data.get("fit_context", "true to size"),
        )
        return jsonify({"status": "ok", "item_id": data["item_id"],
                        "category": meta.get("category"), "analysis": analysis})
    except (KeyError, FileNotFoundError, ValueError) as e:
        return jsonify({"status": "error", "message": str(e)}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ---------------------------------------------------------------------------
# Endpoint 2: /api/generate-tryon-image
# ---------------------------------------------------------------------------

@vision_tryon_bp.route("/api/generate-tryon-image", methods=["POST"])
def generate_tryon_image():
    """Body: { session_id, prompt, negative_prompt?, category?, seed?,
    sleeve_coverage? }"""
    data = request.get_json(force=True)
    try:
        session_id = data["session_id"]
        photo = _load_session_photo(session_id)
        pose = _load_session_pose(session_id)

        category = (data.get("category") or "top").lower()
        if category in LOWER_BODY_CATEGORIES and pose.get("photo_coverage") == "upper_body":
            return jsonify({"status": "error", "message":
                "This item needs a full-body photo to try on. "
                "Update your photo in your profile."}), 422

        mask = _build_mask(photo, pose, category,
                           sleeve_coverage=data.get("sleeve_coverage", "n_a"))
        result = _replicate_inpaint(
            photo, mask,
            prompt=data["prompt"],
            negative_prompt=data.get("negative_prompt",
                "extra limbs, deformed hands, changed face, changed background, "
                "text, watermark, logo, cartoon, illustration"),
            seed=int(data.get("seed", 42)),
        )

        out_name = f"tryon_{uuid.uuid4().hex[:12]}.png"
        out_path = os.path.join(_session_dir(session_id), out_name)
        result.save(out_path)
        return jsonify({"status": "ok",
                        "image_url": f"/tryon-image/{session_id}/{out_name}"})
    except (KeyError, FileNotFoundError, ValueError) as e:
        return jsonify({"status": "error", "message": str(e)}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        gc.collect()


# ---------------------------------------------------------------------------
# Endpoint 3: /api/tryon — combined pipeline
# ---------------------------------------------------------------------------

@vision_tryon_bp.route("/api/tryon", methods=["POST"])
def tryon_pipeline():
    """Body: { session_id, item_id, size, fit_context?, color_variant?, seed? }
    Full pipeline; response mirrors the legacy /try-on so the frontend can
    swap endpoints with no other changes."""
    data = request.get_json(force=True)
    try:
        session_id = data["session_id"]
        photo = _load_session_photo(session_id)
        pose = _load_session_pose(session_id)
        garment, meta = _load_garment(data["item_id"], data.get("color_variant"))
        category = (meta.get("category") or "top").lower()

        if category in LOWER_BODY_CATEGORIES and pose.get("photo_coverage") == "upper_body":
            return jsonify({"status": "error", "message":
                "This item needs a full-body photo to try on. "
                "Update your photo in your profile."}), 422

        analysis = _call_claude_vision(
            photo, garment,
            size_label=data.get("size", "M"),
            fit_context=data.get("fit_context", "true to size"),
        )
        sleeve = analysis.get("garment", {}).get("sleeve_coverage", "n_a")
        mask = _build_mask(photo, pose, category, sleeve_coverage=sleeve)
        result = _replicate_inpaint(
            photo, mask,
            prompt=analysis["prompt"],
            negative_prompt=analysis["negative_prompt"],
            seed=int(data.get("seed", 42)),
        )

        out_name = f"tryon_{uuid.uuid4().hex[:12]}.png"
        result.save(os.path.join(_session_dir(session_id), out_name))
        return jsonify({
            "status": "ok",
            "image_url": f"/tryon-image/{session_id}/{out_name}",
            "analysis": analysis,          # feeds straight into /advice
            "engine": "vision-sdxl",
        })
    except (KeyError, FileNotFoundError, ValueError) as e:
        return jsonify({"status": "error", "message": str(e)}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        gc.collect()

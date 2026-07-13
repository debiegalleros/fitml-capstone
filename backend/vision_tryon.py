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
import size_logic
from config import BOTTOM_CATEGORIES, CATALOG_DIR, UPLOADS_DIR
from db import get_db

vision_tryon_bp = Blueprint("vision_tryon", __name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CLAUDE_VISION_MODEL = os.environ.get("CLAUDE_VISION_MODEL", "claude-sonnet-4-6")

REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN", "")

# Engine selection: "idm-vton" (garment-conditioned diffusion — renders THE
# catalog item) or "sdxl" (text-conditioned inpainting — renders a lookalike
# from the analyzer prompt). Per-request override via the "engine" body field.
TRYON_ENGINE = os.environ.get("TRYON_ENGINE", "idm-vton")

# SDXL inpainting on Replicate. Version pinned for reproducibility (community
# models also require version-pinned prediction creates — the /models/{slug}/
# predictions shortcut 404s for them).
SDXL_INPAINT_MODEL = os.environ.get("SDXL_INPAINT_MODEL", "lucataco/sdxl-inpainting")
SDXL_INPAINT_VERSION = os.environ.get(
    "SDXL_INPAINT_VERSION",
    "a5b13068cc81a89a4fbeefeccc774869fcb34df4dbc92c1555e0f2771d49dde7",
)

# IDM-VTON (garment-conditioned virtual try-on). Version pinned for
# reproducibility — override via env if the maintainer publishes a fix.
IDM_VTON_MODEL = os.environ.get("IDM_VTON_MODEL", "cuuupid/idm-vton")
IDM_VTON_VERSION = os.environ.get(
    "IDM_VTON_VERSION",
    "0513734a452173b8173e907e3a59d19a36266e55b48528559432bd21c7d7e985",
)

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

    if color_variant and color_variant.strip().lower() != item["color"].lower():
        path = catalog.garment_png_path(item, color_variant)
    else:
        path = os.path.join(CATALOG_DIR, item["photo"])
    img = Image.open(path).convert("RGB")
    return img, item


def _load_garment_cutout(item: dict, color_variant: str | None = None) -> Image.Image:
    """The rembg cutout flattened onto white — IDM-VTON wants a clean
    product-style garment image, and the cutout is the exact catalog asset
    (including hue variants) rather than an on-model photo."""
    path = catalog.garment_png_path(item, color_variant)
    rgba = Image.open(path).convert("RGBA")
    flat = Image.new("RGB", rgba.size, (255, 255, 255))
    flat.paste(rgba, mask=rgba.getchannel("A"))
    return flat


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
    # Upper-body photos (cropped at the waist) have no visible hips — tops
    # still composite on them (guardrail only blocks bottoms), so estimate:
    # hips sit ~1.3x shoulder span below the shoulder line, clamped to frame.
    shoulder_span = abs(ls[0] - rs[0]) or w * 0.25
    est_y = min(h - 1, (ls[1] + rs[1]) / 2 + shoulder_span * 1.3)
    lh = pt("left_hip", [ls[0] * 0.9 + rs[0] * 0.1, est_y])
    rh = pt("right_hip", [rs[0] * 0.9 + ls[0] * 0.1, est_y])

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

    # Generous padding so no old clothing survives at the mask boundary.
    # Tops: ~25% of shoulder width each side, mask raised ~18% above the
    # shoulder line to cover collars/necklines of the current outfit.
    # Bottoms: hip keypoints are joint centres spanning only ~half the
    # visible hip width (cf. the compositor's 1.9x bottom width factor), so
    # 25% padding left old trousers visible at both hips — use ~55%.
    # Dresses: shoulder-anchored, but the skirt flares well past shoulder
    # width — widen the bottom edge of the trapezoid to ~60%.
    span = abs(top[0][0] - top[1][0]) or w * 0.25
    if cat in LOWER_BODY_CATEGORIES:
        pad_top_x = pad_bot_x = span * 0.55
    elif cat in FULL_LENGTH_CATEGORIES:
        pad_top_x, pad_bot_x = span * 0.25, span * 0.60
    else:
        pad_top_x = pad_bot_x = span * 0.25
    pad_y = abs(bottom[0][1] - top[0][1]) * 0.08
    neck_raise = span * 0.18 if cat not in LOWER_BODY_CATEGORIES else 0

    def clamp(p):
        return (min(max(p[0], 0), w - 1), min(max(p[1], 0), h - 1))

    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    poly = [clamp(p) for p in [
        (top[0][0] - pad_top_x, top[0][1] - pad_y - neck_raise),
        (top[1][0] + pad_top_x, top[1][1] - pad_y - neck_raise),
        (bottom[1][0] + pad_bot_x, bottom[1][1] + pad_y),
        (bottom[0][0] - pad_bot_x, bottom[0][1] + pad_y),
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

def _data_uri(img: Image.Image, mode: str = "RGB") -> str:
    buf = io.BytesIO()
    img.convert(mode).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _replicate_run(payload_input: dict, model: str, version: str,
                   label: str) -> Image.Image:
    """Create a Replicate prediction, poll to completion, download the image."""
    if not REPLICATE_API_TOKEN:
        raise RuntimeError("REPLICATE_API_TOKEN is not set.")
    headers = {"Authorization": f"Bearer {REPLICATE_API_TOKEN}",
               "Content-Type": "application/json"}

    if version:
        create_url = "https://api.replicate.com/v1/predictions"
        body = {"version": version, "input": payload_input}
    else:
        create_url = f"https://api.replicate.com/v1/models/{model}/predictions"
        body = {"input": payload_input}

    r = requests.post(create_url, headers=headers, json=body, timeout=30)
    r.raise_for_status()
    pred = r.json()

    deadline = time.time() + REPLICATE_TIMEOUT_S
    while pred["status"] not in ("succeeded", "failed", "canceled"):
        if time.time() > deadline:
            raise TimeoutError(f"{label} generation timed out.")
        time.sleep(REPLICATE_POLL_INTERVAL_S)
        pred = requests.get(pred["urls"]["get"], headers=headers, timeout=30).json()

    if pred["status"] != "succeeded":
        raise RuntimeError(f"{label} generation failed: {pred.get('error')}")

    out = pred["output"]
    url = out[0] if isinstance(out, list) else out
    img_bytes = requests.get(url, timeout=60).content
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def _replicate_inpaint(photo: Image.Image, mask: Image.Image,
                       prompt: str, negative_prompt: str, seed: int = 42) -> Image.Image:
    return _replicate_run({
        "image": _data_uri(photo),
        "mask": _data_uri(mask, "L"),
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": 30,
        "guidance_scale": 7.5,
        # near-full repaint inside the mask — lower strength blends the
        # original clothes back in, which is exactly the remnant failure
        # the mirror-worn rule bans
        "strength": 0.99,
        "seed": seed,
    }, SDXL_INPAINT_MODEL, SDXL_INPAINT_VERSION, "SDXL")


def _idm_category(category: str) -> str:
    cat = (category or "").lower()
    if cat in LOWER_BODY_CATEGORIES:
        return "lower_body"
    if cat in FULL_LENGTH_CATEGORIES:
        return "dresses"
    return "upper_body"


def _replicate_idm_vton(photo: Image.Image, garment: Image.Image,
                        garment_des: str, category: str, seed: int = 42) -> Image.Image:
    """Garment-conditioned try-on: IDM-VTON takes the actual garment image and
    masks/redraws internally, so the output wears THE catalog item (no text
    lookalike) with the original clothes removed."""
    idm_cat = _idm_category(category)
    return _replicate_run({
        "human_img": _data_uri(photo),
        "garm_img": _data_uri(garment),
        "garment_des": garment_des,
        "category": idm_cat,
        "force_dc": idm_cat == "dresses",  # DressCode weights for dresses
        "crop": True,  # session photos are not 3:4
        "steps": 30,
        "seed": seed,
    }, IDM_VTON_MODEL, IDM_VTON_VERSION, "IDM-VTON")


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
# Endpoint 3: /api/tryon — combined pipeline (engine dispatch)
# ---------------------------------------------------------------------------

def _get_profile(session_id: str):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM profiles WHERE session_id = ?",
                           (session_id,)).fetchone()
    return dict(row) if row else None


def _recommend(profile: dict, item: dict) -> dict:
    if item["gender"] == "men":
        return size_logic.recommend_mens_size(profile, item)
    return size_logic.recommend_womens_size(profile, item)


def _save_tryon_row(session_id: str, item: dict, size: str, color: str | None,
                    rec: dict, image_path: str) -> str:
    """Persist the try-on like legacy /try-on does, so /advice and the
    History page work identically for generative results."""
    tryon_id = uuid.uuid4().hex[:12]
    with get_db() as conn:
        conn.execute(
            """INSERT INTO tryons (tryon_id, session_id, item_id, brand,
               fabric, category, size, color, recommended_size, confidence,
               state, image_path)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (tryon_id, session_id, item["item_id"],
             catalog._brand(item["product_name"]), item["fabric"],
             item["category"], size, color or item["color"],
             rec["recommended_size"], rec["confidence"], rec["state"],
             image_path))
    return tryon_id


@vision_tryon_bp.route("/api/tryon", methods=["POST"])
def tryon_pipeline():
    """Body: { session_id, item_id, size, fit_context?, color_variant?,
    seed?, engine? }. Full pipeline; response mirrors the legacy /try-on
    (tryon_id, image_url, recommendation fields) so the frontend can swap
    endpoints with no other changes. Engine: request override > TRYON_ENGINE
    env (idm-vton | sdxl)."""
    data = request.get_json(force=True)
    try:
        session_id = data["session_id"]
        engine = (data.get("engine") or TRYON_ENGINE).strip().lower()
        if engine not in ("idm-vton", "sdxl"):
            return jsonify({"status": "error",
                            "message": f"unknown engine {engine!r}"}), 400

        profile = _get_profile(session_id)
        if not profile:
            return jsonify({"status": "error", "message": "unknown session_id"}), 404
        photo = _load_session_photo(session_id)
        pose = _load_session_pose(session_id)
        garment, meta = _load_garment(data["item_id"], data.get("color_variant"))
        category = (meta.get("category") or "top").lower()

        size = (data.get("size") or "M").strip().upper()
        available = [s.strip() for s in meta["size_range"].split(",")]
        if size not in available:
            return jsonify({"status": "error",
                            "message": f"size must be one of {available}"}), 400

        if category in LOWER_BODY_CATEGORIES and pose.get("photo_coverage") == "upper_body":
            return jsonify({"status": "error", "message":
                "This item needs a full-body photo to try on. "
                "Update your photo in your profile."}), 422

        rec = _recommend(profile, meta)
        fit_context = data.get("fit_context", "true to size")
        seed = int(data.get("seed", 42))
        analysis = None

        if engine == "idm-vton":
            # Garment-conditioned: the model gets the actual cutout, no
            # text prompt lever — masking and blending happen inside.
            cutout = _load_garment_cutout(meta, data.get("color_variant"))
            garment_des = (f"{meta['color']} {meta['fabric']} "
                           f"{meta['category']} — {meta['product_name']}")
            result = _replicate_idm_vton(photo, cutout, garment_des,
                                         category, seed=seed)
        else:
            analysis = _call_claude_vision(photo, garment, size_label=size,
                                           fit_context=fit_context)
            sleeve = analysis.get("garment", {}).get("sleeve_coverage", "n_a")
            mask = _build_mask(photo, pose, category, sleeve_coverage=sleeve)
            result = _replicate_inpaint(photo, mask,
                                        prompt=analysis["prompt"],
                                        negative_prompt=analysis["negative_prompt"],
                                        seed=seed)

        out_name = f"tryon_{uuid.uuid4().hex[:12]}.png"
        out_path = os.path.join(_session_dir(session_id), out_name)
        result.save(out_path)
        tryon_id = _save_tryon_row(session_id, meta, size,
                                   data.get("color_variant"), rec, out_path)

        return jsonify({
            "status": "ok",
            "tryon_id": tryon_id,
            "image_url": f"/tryon-image/{session_id}/{out_name}",
            "size": size,
            "recommended_size": rec["recommended_size"],
            "confidence": rec["confidence"],
            "state": rec["state"],
            "analysis": analysis,          # null on the idm-vton path
            "engine": engine,
        })
    except (KeyError, FileNotFoundError, ValueError) as e:
        return jsonify({"status": "error", "message": str(e)}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        gc.collect()

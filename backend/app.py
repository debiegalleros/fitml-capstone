"""FitML Flask backend — Phase 8 (+ vision try-on rebuild).

Endpoints:
  POST /upload-profile   photo + measurements -> session_id, photo_coverage
  GET  /catalog          filterable garment list (serves on-model photos)
  POST /recommend-size   class-weighted XGBoost + validation & borderline layers
  POST /try-on           legacy pose-anchored 2D compositing (fallback path)
  POST /advice           Claude multimodal advice text (two paragraphs)
  POST /api/tryon        generative try-on: Claude Vision + SDXL inpainting
                         (plus /api/generate-tryon-prompt and
                         /api/generate-tryon-image — see vision_tryon.py)

Plus image-serving routes for catalog files and session try-on results.
"""
import gc
import os
import uuid

import cv2
import numpy as np
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

import catalog
import size_logic
import tryon
from cleanup_uploads import purge_expired_uploads
from config import (ALLOWED_PHOTO_EXT, BOTTOM_CATEGORIES, CATALOG_DIR,
                    MAX_UPLOAD_MB, UPLOADS_DIR)
from db import get_db, init_db
from vision_tryon import vision_tryon_bp

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
app.register_blueprint(vision_tryon_bp)

# Memory hardening (512 MB Render instance): phone photos arrive at 4000px+;
# every downstream step (privacy crop, pose, compositing, inpainting mask)
# works fine at 1280px and the arrays are ~10x smaller.
MAX_PHOTO_SIDE = 1280

os.makedirs(UPLOADS_DIR, exist_ok=True)
init_db()
purge_expired_uploads()


def _error(message, status=400):
    return jsonify({"error": message}), status


def _get_profile(session_id):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM profiles WHERE session_id = ?",
                           (session_id,)).fetchone()
    return dict(row) if row else None


def _parse_float(form, key):
    value = form.get(key)
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def _downscale(image):
    h, w = image.shape[:2]
    scale = MAX_PHOTO_SIDE / max(h, w)
    if scale >= 1.0:
        return image
    return cv2.resize(image, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------- endpoints

@app.post("/upload-profile")
def upload_profile():
    """Multipart form: photo file + manual measurements. The photo is used
    for try-on pose ONLY — it never feeds the size model (locked decision)."""
    purge_expired_uploads()

    if "photo" not in request.files:
        return _error("missing 'photo' file")
    photo = request.files["photo"]
    ext = os.path.splitext(photo.filename or "")[1].lower()
    if ext not in ALLOWED_PHOTO_EXT:
        return _error(f"unsupported photo type {ext or '(none)'}")

    form = request.form
    name = (form.get("name") or "").strip()
    if not name:
        return _error("name is required")
    try:
        height_cm = _parse_float(form, "height_cm")
        weight_kg = _parse_float(form, "weight_kg")
        waist_cm = _parse_float(form, "waist_cm")
        hip_cm = _parse_float(form, "hip_cm")
    except ValueError:
        return _error("measurements must be numeric")
    if height_cm is None:
        return _error("height_cm is required")

    # Bust: band+cup dropdowns OR chest cm converted (flagged lower-precision)
    bust_band = form.get("bust_band")
    bust_cup = form.get("bust_cup")
    bust_input_method = "band_cup"
    if not bust_band and form.get("chest_cm"):
        band, cup_ord = size_logic.chest_cm_to_band_cup(float(form["chest_cm"]))
        bust_band = band
        bust_cup = next(k for k, v in size_logic.CUP_RANK.items() if v == cup_ord)
        bust_input_method = "chest_cm"
    bust_band = int(bust_band) if bust_band else None

    data = np.frombuffer(photo.read(), np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    del data
    if image is None:
        return _error("could not decode photo")
    image = _downscale(image)

    # Privacy crop — OPT-IN (checkbox on the upload form, unchecked by
    # default): if selected, remove everything above the nose BEFORE any
    # other processing or storage. The uncropped image only ever exists in
    # this local variable, transiently — cropping happens (if at all)
    # before the image is ever written to disk, and nothing about the
    # detected nose position (no bbox, no landmarks) is persisted either
    # way. If the box is checked but the nose can't be confidently found,
    # ask for a re-upload rather than guess a crop boundary — a wrong guess
    # could leave part of the face exposed.
    #
    # If the box is left unchecked, the photo (face included) is used as
    # uploaded. That path still gets one piece of protection: a face
    # bounding box is detected (non-blocking — a miss just means no
    # protection for this session) and carried in pose.json so the
    # generative try-on pipeline can re-paste the real face onto every
    # render afterward, guarding against IDM-VTON's documented tendency to
    # occasionally regenerate a synthetic face (see docs/genai_usage.md).
    # This box is never computed on the checked path — once the face is
    # cropped out, there's nothing left to protect.
    crop_face = form.get("crop_face", "false").strip().lower() == "true"
    face_bbox = None
    if crop_face:
        cropped = tryon.crop_above_nose(image)
        if cropped is None:
            return _error("We couldn't get a clear enough view of your nose "
                          "to safely crop your photo for privacy. Please "
                          "upload a clearer, forward-facing photo.", 422)
        image = cropped
    else:
        face_bbox = tryon.detect_face_bbox(image)

    pose = tryon.extract_pose(image)
    if not pose.get("ok"):
        return _error("We couldn't detect a person in this photo. Please use "
                      "a clear, front-facing photo.", 422)
    pose["face_bbox"] = list(face_bbox) if face_bbox else None

    session_id = str(uuid.uuid4())
    session_dir = os.path.join(UPLOADS_DIR, session_id)
    os.makedirs(session_dir)
    photo_path = os.path.join(session_dir, "photo.jpg")
    cv2.imwrite(photo_path, image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    tryon.save_pose(session_dir, pose)

    # Optional side/back photos — same session folder, same 24h auto-delete,
    # same crop_face preference, but NOT pose-extracted: only the front
    # photo feeds try-on (locked decision). These are a foundation for a
    # future multi-angle try-on feature, shown today only as profile
    # thumbnails. Optional and non-blocking: if crop_face is checked but the
    # nose can't be confidently found on one of these, skip saving it rather
    # than reject the whole upload — unlike the required front photo,
    # there's nothing else depending on it.
    photo_side_path = None
    photo_back_path = None
    for field, filename in (("photo_side", "photo_side.jpg"), ("photo_back", "photo_back.jpg")):
        extra = request.files.get(field)
        if not extra or not extra.filename:
            continue
        extra_ext = os.path.splitext(extra.filename)[1].lower()
        if extra_ext not in ALLOWED_PHOTO_EXT:
            continue
        extra_data = np.frombuffer(extra.read(), np.uint8)
        extra_image = cv2.imdecode(extra_data, cv2.IMREAD_COLOR)
        del extra_data
        if extra_image is None:
            continue
        extra_image = _downscale(extra_image)
        if crop_face:
            extra_cropped = tryon.crop_above_nose(extra_image)
            if extra_cropped is None:
                continue
            extra_image = extra_cropped
        extra_path = os.path.join(session_dir, filename)
        cv2.imwrite(extra_path, extra_image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if field == "photo_side":
            photo_side_path = extra_path
        else:
            photo_back_path = extra_path

    with get_db() as conn:
        conn.execute(
            """INSERT INTO profiles (session_id, name, photo_path,
               photo_side_path, photo_back_path, photo_coverage, face_cropped,
               height_cm, weight_kg, bust_band, bust_cup, bust_input_method,
               waist_cm, hip_cm, body_type)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (session_id, name, photo_path, photo_side_path, photo_back_path,
             pose["coverage"], int(crop_face), height_cm, weight_kg,
             bust_band, bust_cup, bust_input_method, waist_cm, hip_cm,
             form.get("body_type")))

    response = {
        "session_id": session_id,
        "name": name,
        "photo_coverage": pose["coverage"],
        "face_cropped": crop_face,
        "bust_input_method": bust_input_method,
        "photo_url": f"/tryon-image/{session_id}/photo.jpg",
    }
    if photo_side_path:
        response["photo_side_url"] = f"/tryon-image/{session_id}/photo_side.jpg"
    if photo_back_path:
        response["photo_back_url"] = f"/tryon-image/{session_id}/photo_back.jpg"
    del image
    gc.collect()
    return jsonify(response)


@app.get("/catalog")
def get_catalog():
    """Filterable list. Cards/detail serve the on-model photo; the cutout URL
    is included only for try-on + swatch previews (display convention)."""
    return jsonify({"items": catalog.filter_items(request.args)})


@app.post("/recommend-size")
def recommend_size():
    body = request.get_json(silent=True) or {}
    profile = _get_profile(body.get("session_id", ""))
    if not profile:
        return _error("unknown session_id", 404)
    item = catalog.get_item(body.get("item_id", ""))
    if not item:
        return _error("unknown item_id", 404)

    if item["gender"] == "men":
        result = size_logic.recommend_mens_size(profile, item)
    else:
        result = size_logic.recommend_womens_size(profile, item)
    result["item_id"] = item["item_id"]
    return jsonify(result)


@app.post("/try-on")
def try_on():
    body = request.get_json(silent=True) or {}
    session_id = body.get("session_id", "")
    profile = _get_profile(session_id)
    if not profile:
        return _error("unknown session_id", 404)
    item = catalog.get_item(body.get("item_id", ""))
    if not item:
        return _error("unknown item_id", 404)
    size = (body.get("size") or "").strip().upper()
    available = [s.strip() for s in item["size_range"].split(",")]
    if size not in available:
        return _error(f"size must be one of {available}")

    # Half-body photos can't try on bottoms (locked decision, friendly copy)
    if item["category"] in BOTTOM_CATEGORIES and profile["photo_coverage"] == "upper_body":
        return _error("This item needs a full-body photo to try on. "
                      "Update your photo in your profile.", 422)

    session_dir = os.path.join(UPLOADS_DIR, session_id)
    photo = cv2.imread(profile["photo_path"], cv2.IMREAD_COLOR)
    if photo is None:
        return _error("session photo expired — please upload again", 410)
    pose = tryon.load_pose(session_dir)

    color = body.get("color")
    garment_path = catalog.garment_png_path(item, color)
    garment = cv2.imread(garment_path, cv2.IMREAD_UNCHANGED)
    if garment is None or garment.shape[2] != 4:
        return _error("garment image unavailable", 500)

    # Size-proportional rendering relative to the model-recommended size
    if item["gender"] == "men":
        rec = size_logic.recommend_mens_size(profile, item)
    else:
        rec = size_logic.recommend_womens_size(profile, item)
    width_scale, length_scale = size_logic.size_scale_factors(
        item, size, rec["recommended_size"])

    result = tryon.composite(photo, garment, pose, item["category"],
                             width_scale, length_scale)

    tryon_id = uuid.uuid4().hex[:12]
    out_name = f"tryon_{tryon_id}.jpg"
    out_path = os.path.join(session_dir, out_name)
    cv2.imwrite(out_path, result, [cv2.IMWRITE_JPEG_QUALITY, 92])

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
             out_path))

    del photo, garment, result
    gc.collect()

    return jsonify({
        "tryon_id": tryon_id,
        "image_url": f"/tryon-image/{session_id}/{out_name}",
        "size": size,
        "recommended_size": rec["recommended_size"],
        "confidence": rec["confidence"],
        "state": rec["state"],
        "size_scale": {"width": round(width_scale, 3),
                       "length": round(length_scale, 3)},
    })


@app.post("/advice")
def advice():
    import advice_engine  # deferred: importing anthropic only when needed

    body = request.get_json(silent=True) or {}
    session_id = body.get("session_id", "")
    profile = _get_profile(session_id)
    if not profile:
        return _error("unknown session_id", 404)

    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM tryons WHERE tryon_id = ? AND session_id = ?",
            (body.get("tryon_id", ""), session_id)).fetchone()
    if not row:
        return _error("unknown tryon_id", 404)
    tryon_row = dict(row)

    item = catalog.get_item(tryon_row["item_id"])
    recommendation = {"recommended_size": tryon_row["recommended_size"],
                      "confidence": tryon_row["confidence"],
                      "state": tryon_row["state"]}
    item["brand"] = tryon_row["brand"]

    if not os.environ.get("ANTHROPIC_API_KEY"):
        return _error("ANTHROPIC_API_KEY is not set — add it to backend/.env", 503)
    try:
        text = advice_engine.generate_advice(
            profile, item, recommendation, tryon_row["size"],
            tryon_row["image_path"])
    except Exception as exc:  # keep the demo alive on API hiccups
        return _error(f"advice generation failed: {type(exc).__name__}", 502)

    return jsonify({
        "advice": text,
        "state": recommendation["state"],
        "confidence": recommendation["confidence"],
        "recommended_size": recommendation["recommended_size"],
    })


# ----------------------------------------------------------- image serving

@app.get("/images/photos/<path:filename>")
def serve_photo(filename):
    return send_from_directory(os.path.join(CATALOG_DIR, "photos"), filename)


@app.get("/images/garments/<path:filename>")
def serve_garment(filename):
    return send_from_directory(os.path.join(CATALOG_DIR, "garments"), filename)


@app.get("/tryon-image/<session_id>/<path:filename>")
def serve_tryon(session_id, filename):
    # Session-scoped: the random UUID folder name is the access token
    return send_from_directory(os.path.join(UPLOADS_DIR, session_id), filename)


@app.get("/health")
def health():
    return jsonify({"status": "ok", "catalog_items": len(catalog.all_items())})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)

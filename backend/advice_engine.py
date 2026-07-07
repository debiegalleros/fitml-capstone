"""Claude-generated advice text for /advice (locked decision: advice text
only — multimodal input of the composited try-on image for qualitative visual
commentary, never measurement estimation, never part of the graded ML).

History suggestions are a rule-based DB lookup (2+ prior items from the same
brand/fabric with consistent fit feedback), NOT a second trained model.
"""
import base64
import os

import anthropic

from db import get_db

ADVICE_MODEL = "claude-opus-4-8"

# Prompt template implements the locked confidence-box copy rules:
# two paragraphs, second starts with "Note:" and uses plain everyday
# language a non-technical shopper understands — no tailoring jargon.
SYSTEM_PROMPT = """\
You are FitML's fit advisor for an e-commerce virtual fitting room.
Write personalized clothing-fit advice for a shopper. Rules:
- Write EXACTLY two paragraphs, separated by a blank line.
- Paragraph 1: explain the size recommendation using the shopper's
  measurements in plain language (why this size should fit their body).
- Paragraph 2: must start with the word "Note:" followed by visual
  observations from the attached try-on photo. Use simple everyday language
  a non-technical shopper understands. NO tailoring jargon — never use words
  like "seam alignment", "hem", "drape", "silhouette", "inseam", or "bodice".
  Example tone: "Looking at your photo, this fits you well — it sits nicely
  on your shoulders and the length is just right for you."
- The try-on image is a 2D preview composite, so comment on overall size and
  placement, not fabric texture or lighting.
- If a sizing tip is flagged (borderline case), explain the tradeoff of the
  suggested size in paragraph 1 (e.g. roomier fit vs snug fit).
- Keep the whole response under 130 words. Warm, helpful, concrete.
"""


def history_note(session_id: str, brand: str, fabric: str):
    """Rule-based lookup: 2+ prior try-ons from the same brand or fabric with
    consistent user fit feedback -> one extra line for the Claude prompt."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT brand, fabric, fit_feedback FROM tryons
               WHERE session_id = ? AND fit_feedback IS NOT NULL""",
            (session_id,),
        ).fetchall()

    for field, value, label in (("brand", brand, "brand"), ("fabric", fabric, "fabric")):
        matches = [r["fit_feedback"] for r in rows if r[field] == value]
        if len(matches) >= 2 and len(set(matches)) == 1:
            fb = matches[0]
            desc = {"fit": "run true to size",
                    "small": "run small",
                    "large": "run large"}[fb]
            return (f"The shopper has tried {len(matches)} items from this "
                    f"{label} before and they consistently {desc} for them. "
                    f"Mention this in paragraph 1.")
    return None


def generate_advice(profile: dict, item: dict, recommendation: dict,
                    size: str, image_path: str) -> str:
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    with open(image_path, "rb") as f:
        image_b64 = base64.standard_b64encode(f.read()).decode("utf-8")

    measurements = [f"height {profile['height_cm']} cm"]
    if profile.get("weight_kg"):
        measurements.append(f"weight {profile['weight_kg']} kg")
    if profile.get("bust_band"):
        measurements.append(f"bust {profile['bust_band']}{profile.get('bust_cup') or ''}")
    if profile.get("waist_cm"):
        measurements.append(f"waist {profile['waist_cm']} cm")
    if profile.get("hip_cm"):
        measurements.append(f"hip {profile['hip_cm']} cm")
    if profile.get("body_type"):
        measurements.append(f"body type {profile['body_type']}")

    lines = [
        f"Shopper measurements: {', '.join(measurements)}.",
        f"Garment: {item['product_name']} ({item['category']}, "
        f"{item['fabric']}, {item['color']}).",
        f"Recommended size: {recommendation['recommended_size']} "
        f"(confidence {recommendation['confidence']}%).",
        f"Size being tried on in the photo: {size}.",
    ]
    if recommendation.get("state") == "amber":
        lines.append("Borderline case flagged: the fabric has little stretch "
                     "and the cut is fitted, so we suggested one size up. "
                     "Explain that tradeoff.")
    if size != recommendation["recommended_size"]:
        lines.append(f"The shopper picked {size} instead of the recommended "
                     f"{recommendation['recommended_size']} — explain how this "
                     f"size will fit differently.")
    note = history_note(session_id=profile["session_id"], brand=item.get("brand"),
                        fabric=item.get("fabric"))
    if note:
        lines.append(note)

    ext = os.path.splitext(image_path)[1].lower()
    media_type = "image/png" if ext == ".png" else "image/jpeg"

    response = client.messages.create(
        model=ADVICE_MODEL,
        max_tokens=400,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image",
                 "source": {"type": "base64", "media_type": media_type,
                            "data": image_b64}},
                {"type": "text", "text": "\n".join(lines)},
            ],
        }],
    )
    return next(b.text for b in response.content if b.type == "text").strip()

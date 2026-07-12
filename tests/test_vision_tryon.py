"""Smoke test for the Vision AI try-on pipeline.

Usage (backend running locally, one profile already uploaded):
    TEST_SESSION_ID=<uuid> TEST_ITEM_ID=<item> python tests/test_vision_tryon.py

Against production:
    BASE_URL=https://fit-ml.onrender.com TEST_SESSION_ID=... TEST_ITEM_ID=... \
        python tests/test_vision_tryon.py

Optional:
    TEST_BOTTOM_ITEM_ID=<jeans/skirt item>  -> asserts the friendly 422 when
        the session photo is upper-body only (skipped for full-body photos).

Checks:
  1. /api/generate-tryon-prompt returns well-formed analysis JSON.
  2. GUARDRAIL: the analysis JSON contains zero measurement language —
     Claude Vision must never estimate body measurements (locked decision).
  3. /api/tryon returns an image URL and the image downloads as a real PNG/JPEG.
  4. Bottoms + upper-body photo -> friendly 422, exact wording preserved.
"""
import os
import re
import sys

import requests

BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:5001").rstrip("/")
SESSION_ID = os.environ.get("TEST_SESSION_ID")
ITEM_ID = os.environ.get("TEST_ITEM_ID")
BOTTOM_ITEM_ID = os.environ.get("TEST_BOTTOM_ITEM_ID")

FRIENDLY_422 = ("This item needs a full-body photo to try on. "
                "Update your photo in your profile.")

# Any of these appearing in the analyzer output means the guardrail leaked:
# measurement estimation is banned (photo is for pose only — locked decision).
MEASUREMENT_PATTERNS = [
    r"\b\d+(\.\d+)?\s*(cm|centimeter|inch|inches|in\.)\b",
    r"\bmeasure(s|d|ment|ments)?\b",
    r"\b(bust|waist|hip|chest)\s*(size|circumference|measurement)\b",
    r"\bbody\s*(shape|size|type)\b",
    r"\b(slim|curvy|petite|plus[- ]size|hourglass|pear[- ]shaped)\b",
]


def fail(msg):
    print(f"FAIL: {msg}")
    sys.exit(1)


def check_no_measurement_language(analysis: dict):
    blob = str(analysis).lower()
    for pattern in MEASUREMENT_PATTERNS:
        m = re.search(pattern, blob)
        if m:
            fail(f"guardrail breach — measurement language in analysis JSON: "
                 f"{m.group(0)!r}")
    print("  guardrail OK: zero measurement language in analysis")


def main():
    if not SESSION_ID or not ITEM_ID:
        fail("set TEST_SESSION_ID and TEST_ITEM_ID (upload a profile first)")

    # 1+2 — analyzer
    print(f"1) POST {BASE_URL}/api/generate-tryon-prompt")
    r = requests.post(f"{BASE_URL}/api/generate-tryon-prompt", json={
        "session_id": SESSION_ID, "item_id": ITEM_ID, "size": "M",
        "fit_context": "true to size"}, timeout=120)
    if r.status_code != 200:
        fail(f"analyzer returned {r.status_code}: {r.text[:300]}")
    payload = r.json()
    analysis = payload.get("analysis", {})
    for key in ("person", "garment", "prompt", "negative_prompt"):
        if key not in analysis:
            fail(f"analysis missing key {key!r}: {list(analysis)}")
    sleeve = analysis["garment"].get("sleeve_coverage")
    if sleeve not in ("sleeveless", "short_sleeve", "long_sleeve", "n_a"):
        fail(f"unexpected sleeve_coverage {sleeve!r}")
    for phrase in ("original clothing", "layered over"):
        if phrase not in analysis["negative_prompt"].lower():
            print(f"  WARN: negative_prompt missing {phrase!r} "
                  f"(mirror-worn ban weakened)")
    check_no_measurement_language(analysis)
    print(f"  prompt: {analysis['prompt'][:100]}...")

    # 3 — full pipeline
    print(f"2) POST {BASE_URL}/api/tryon (SDXL inpainting — may take ~60s cold)")
    r = requests.post(f"{BASE_URL}/api/tryon", json={
        "session_id": SESSION_ID, "item_id": ITEM_ID, "size": "M",
        "seed": 42}, timeout=240)
    if r.status_code != 200:
        fail(f"/api/tryon returned {r.status_code}: {r.text[:300]}")
    body = r.json()
    if body.get("engine") != "vision-sdxl" or not body.get("image_url"):
        fail(f"unexpected /api/tryon response: {body}")
    img = requests.get(f"{BASE_URL}{body['image_url']}", timeout=60)
    if img.status_code != 200 or img.content[:4] not in (b"\x89PNG", b"\xff\xd8\xff\xe0",
                                                         b"\xff\xd8\xff\xe1", b"\xff\xd8\xff\xdb"):
        fail(f"result image not downloadable ({img.status_code})")
    print(f"  image OK: {body['image_url']} ({len(img.content)//1024} KB)")

    # 4 — friendly 422 for bottoms on upper-body photos
    if BOTTOM_ITEM_ID:
        print(f"3) POST {BASE_URL}/api/tryon with bottoms item {BOTTOM_ITEM_ID}")
        r = requests.post(f"{BASE_URL}/api/tryon", json={
            "session_id": SESSION_ID, "item_id": BOTTOM_ITEM_ID, "size": "M"},
            timeout=240)
        if r.status_code == 422:
            if r.json().get("message") != FRIENDLY_422:
                fail(f"422 wording changed: {r.json().get('message')!r}")
            print("  friendly 422 OK (exact wording preserved)")
        elif r.status_code == 200:
            print("  photo is full-body -> bottoms composited normally (OK)")
        else:
            fail(f"bottoms case returned {r.status_code}: {r.text[:300]}")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()

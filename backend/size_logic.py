"""Size recommendation logic for /recommend-size.

Women's garments: the served pipeline is models/xgboost_weighted.joblib
(class-weighted XGBoost — locked decision, see docs/model_selection.md).
The pipeline owns scaling/encoding; this module builds the 13-column raw
input rows (docs/phase8_notes.md) and adds the validation layer that maps
user inputs onto the fitted vocabularies instead of silently zeroing the
OneHotEncoder.

Men's garments: rule-based Uniqlo chart lookup (src/mens_extension.py) —
NOT the trained model, excluded from the fairness audit.
"""
import joblib
import pandas as pd

from config import MODEL_PATH, LABEL_ENCODER_PATH, MENS_CHART_CSV

# ---------------------------------------------------------------- artifacts

_pipeline = joblib.load(MODEL_PATH)
_label_encoder = joblib.load(LABEL_ENCODER_PATH)
_CLASS_INDEX = {c: i for i, c in enumerate(_label_encoder.classes_)}  # fit/large/small
_mens_chart = pd.read_csv(MENS_CHART_CSV)

INPUT_COLUMNS = [
    "size_ordered", "height_cm", "weight_kg", "bust_band", "bust_cup_ordinal",
    "hip_cm", "weight_kg_missing", "hip_cm_missing", "bust_band_missing",
    "bust_cup_ordinal_missing", "source", "category_broad", "body_type",
]

# Training-set medians (re-derived from data/processed/model_ready.csv —
# see docs/phase8_notes.md) used to impute absent optional measurements.
MEDIANS = {"weight_kg": 61.2, "hip_cm": 99.06, "bust_band": 34, "bust_cup_ordinal": 3}

# Plausible ranges mirror src/prepare_data.py PLAUSIBLE_RANGES
PLAUSIBLE = {
    "height_cm": (137, 213),
    "weight_kg": (35, 180),
    "hip_cm": (60, 180),
    "bust_band": (20, 56),
}

CUP_RANK = {
    "aa": 0, "a": 1, "b": 2, "c": 3, "d": 4, "dd": 5, "dd/e": 5, "e": 5,
    "ddd": 6, "ddd/f": 6, "f": 6, "dddd": 7, "dddd/g": 7, "g": 7,
    "h": 8, "i": 9, "j": 10, "k": 11,
}

# Fitted OneHotEncoder vocabularies (docs/phase8_notes.md)
FITTED_BODY_TYPES = {
    "apple", "athletic", "full bust", "hourglass", "not_reported", "pear",
    "petite", "straight & narrow",
}
# Profile dropdown -> fitted vocabulary
BODY_TYPE_MAP = {"rectangle": "straight & narrow"}

CATEGORY_BROAD_MAP = {
    "tshirt": "tops", "tank": "tops", "polo": "tops", "blouse": "tops",
    "dress": "dresses",
    "jeans": "bottoms", "skirt": "bottoms", "shorts": "bottoms",
    "slacks": "bottoms", "trousers": "bottoms",
    "jacket": "outerwear", "sweater": "outerwear",
}

# Letter sizes -> numeric size_ordered (US women's size midpoints: XS=0-2,
# S=4-6, M=8-10, L=12-14, XL=16-18, XXL=20-22). Demo convention, documented.
LETTER_TO_ORDERED = {"XS": 1, "S": 5, "M": 9, "L": 13, "XL": 17, "XXL": 21}

# Women's demo size chart (cm) for size-proportional try-on rendering:
# bust width and overall length ratios are taken relative to these.
WOMENS_CHART_BUST = {"XS": 82, "S": 87, "M": 93, "L": 99, "XL": 106, "XXL": 113}
WOMENS_CHART_HIP = {"XS": 88, "S": 93, "M": 99, "L": 105, "XL": 112, "XXL": 119}

# Borderline rule inputs (locked decision: near boundary + low-stretch fabric
# + fitted cut -> amber). Fabric stretch from catalog fabric strings; cut is
# assigned per category (catalog metadata has no cut field — documented demo
# convention).
LOW_STRETCH_FABRICS = {
    "rigid denim", "cotton twill", "polyester crepe", "cotton poplin",
    "linen-cotton blend", "polyester twill", "polyester shell",
    "polyester-viscose twill", "cotton pique",
}
FITTED_CATEGORIES = {"jeans", "dress", "polo", "tank"}

MENS_TOPS = {"tshirt", "polo", "jacket", "sweater", "tank"}   # sized by chest
MENS_BOTTOMS = {"jeans", "shorts", "slacks", "trousers", "skirt"}


def chest_cm_to_band_cup(chest_cm: float):
    """Approximate band+cup from a single chest measurement (lower precision;
    the input method is flagged on the profile). Band = bust inches minus a
    standard 5" allowance rounded to the nearest even number; cup rank from
    the leftover difference (1" per cup)."""
    bust_in = chest_cm / 2.54
    band = int(round((bust_in - 5) / 2.0) * 2)
    band = max(28, min(48, band))
    cup_ordinal = int(round(bust_in - band))
    cup_ordinal = max(0, min(11, cup_ordinal))
    return band, cup_ordinal


def _clamp(value, key):
    lo, hi = PLAUSIBLE[key]
    return min(max(float(value), lo), hi)


def build_feature_row(profile: dict, size_ordered: int, category: str) -> dict:
    """Validation layer: map one user profile + candidate size onto the 13
    fitted columns, imputing/flagging anything absent or out of vocabulary."""
    weight = profile.get("weight_kg")
    hip = profile.get("hip_cm")
    band = profile.get("bust_band")
    cup = profile.get("bust_cup")

    cup_ordinal = None
    if cup is not None and str(cup).strip() != "":
        cup_ordinal = CUP_RANK.get(str(cup).strip().lower())

    body_type = str(profile.get("body_type") or "").strip().lower()
    body_type = BODY_TYPE_MAP.get(body_type, body_type)
    if body_type not in FITTED_BODY_TYPES:
        body_type = "not_reported"

    category_broad = CATEGORY_BROAD_MAP.get(category, "other")

    # Profiles aren't from either source site; pick the source whose
    # measurement schema the profile matches (RTR collected weight,
    # ModCloth didn't).
    source = "renttherunway" if weight is not None else "modcloth"

    return {
        "size_ordered": int(size_ordered),
        "height_cm": _clamp(profile["height_cm"], "height_cm"),
        "weight_kg": _clamp(weight, "weight_kg") if weight is not None else MEDIANS["weight_kg"],
        "bust_band": _clamp(band, "bust_band") if band is not None else MEDIANS["bust_band"],
        "bust_cup_ordinal": cup_ordinal if cup_ordinal is not None else MEDIANS["bust_cup_ordinal"],
        "hip_cm": _clamp(hip, "hip_cm") if hip is not None else MEDIANS["hip_cm"],
        "weight_kg_missing": int(weight is None),
        "hip_cm_missing": int(hip is None),
        "bust_band_missing": int(band is None),
        "bust_cup_ordinal_missing": int(cup_ordinal is None),
        "source": source,
        "category_broad": category_broad,
        "body_type": body_type,
    }


HIP_KEYED_CATEGORIES = {"jeans", "skirt", "shorts", "slacks", "trousers"}


def _bust_cm(profile: dict):
    band = profile.get("bust_band")
    if band is None:
        return None
    cup_ord = CUP_RANK.get(str(profile.get("bust_cup") or "").lower(), 3)
    # Standard convention: bust circumference (in) ~ band + cup rank
    return (band + cup_ord) * 2.54


def _anchor_size(profile: dict, category: str, sizes: list):
    """Deterministic chart anchor: pick the available size whose chart
    measurement is nearest the user's keyed measurement (hip for bottoms,
    bust for tops/outerwear, the larger-indexed of the two for dresses)."""
    def nearest(chart, value):
        idx = min(range(len(sizes)), key=lambda i: abs(chart[sizes[i]] - value))
        clamped = (idx == 0 and value < chart[sizes[0]]) or \
                  (idx == len(sizes) - 1 and value > chart[sizes[-1]])
        # near a boundary if within 2 cm of the midpoint to an adjacent size
        near = False
        for j in (idx - 1, idx + 1):
            if 0 <= j < len(sizes):
                midpoint = (chart[sizes[idx]] + chart[sizes[j]]) / 2.0
                if abs(value - midpoint) <= 2.0:
                    near = True
        return idx, clamped, near

    bust = _bust_cm(profile)
    hip = profile.get("hip_cm")
    if category in HIP_KEYED_CATEGORIES:
        value = hip if hip is not None else MEDIANS["hip_cm"]
        return nearest(WOMENS_CHART_HIP, value)
    if category == "dress" and hip is not None and bust is not None:
        bi = nearest(WOMENS_CHART_BUST, bust)
        hi = nearest(WOMENS_CHART_HIP, hip)
        return max(bi, hi, key=lambda t: t[0])  # fit the larger measurement
    value = bust if bust is not None else (MEDIANS["bust_band"] + 3) * 2.54
    return nearest(WOMENS_CHART_BUST, value)


def recommend_womens_size(profile: dict, item: dict) -> dict:
    """Chart-anchored recommendation.

    The pipeline was trained on real (customer, ordered size) -> fit-feedback
    rows, so it answers the in-distribution question "will THIS size run
    small/fit/large on THIS body". Scanning counterfactual sizes for the same
    body is out-of-distribution and inverts (verified during Phase 8 testing —
    see docs/phase8_notes.md). So: (1) anchor the size deterministically from
    the measurements via the size chart, (2) ask the model for the fit verdict
    at that anchor, (3) shift one size when the model flags a misfit,
    (4) apply the borderline blue/amber layer on top.
    """
    sizes = [s.strip() for s in item["size_range"].split(",")]
    category = item["category"]

    def predict(size_letter):
        row = build_feature_row(profile, LETTER_TO_ORDERED[size_letter], category)
        proba = _pipeline.predict_proba(
            pd.DataFrame([row], columns=INPUT_COLUMNS))[0]
        return {c: float(proba[i]) for c, i in _CLASS_INDEX.items()}

    anchor_idx, clamped, near_chart_boundary = _anchor_size(profile, category, sizes)
    anchor = sizes[anchor_idx]
    anchor_probs = predict(anchor)

    # Model-flagged misfit at the anchor -> shift one size (never past a
    # clamped chart end: a measurement beyond the largest size never sizes
    # down, and vice versa).
    final_idx = anchor_idx
    adjustment = None
    verdict = max(anchor_probs, key=anchor_probs.get)
    if verdict == "small" and anchor_idx + 1 < len(sizes):
        final_idx, adjustment = anchor_idx + 1, "sized_up_runs_small"
    elif verdict == "large" and anchor_idx > 0 and not clamped:
        final_idx, adjustment = anchor_idx - 1, "sized_down_runs_large"

    final = sizes[final_idx]
    final_probs = anchor_probs if final == anchor else predict(final)
    confidence = int(round(final_probs["fit"] * 100))
    # The anchor itself is chart-grounded; don't let a noisy tail probability
    # display an absurdly low confidence for an unadjusted anchor.
    confidence = max(confidence, 45 if adjustment is None else 40)

    # Borderline rule (locked decision): near size boundary AND low-stretch
    # fabric AND fitted cut -> amber state, size up, lowered confidence.
    near_boundary = near_chart_boundary or \
        abs(final_probs["fit"] - final_probs["small"]) < 0.10
    low_stretch = item["fabric"] in LOW_STRETCH_FABRICS
    fitted_cut = category in FITTED_CATEGORIES

    state = "blue"
    if near_boundary and low_stretch and fitted_cut:
        state = "amber"
        if final_idx + 1 < len(sizes) and adjustment != "sized_up_runs_small":
            final_idx += 1
            final = sizes[final_idx]
            adjustment = (adjustment or "") + "+amber_size_up"
        confidence = max(40, confidence - 7)  # lowered displayed confidence

    return {
        "recommended_size": final,
        "anchor_size": anchor,
        "adjustment": adjustment,
        "confidence": confidence,
        "state": state,
        "probabilities": {k: round(v, 3) for k, v in anchor_probs.items()},
        "method": "chart_anchor+xgboost_weighted",
        "borderline": {"near_boundary": bool(near_boundary),
                       "low_stretch_fabric": low_stretch,
                       "fitted_cut": fitted_cut},
    }


def recommend_mens_size(profile: dict, item: dict) -> dict:
    """Rule-based Uniqlo chart lookup for the men's extension (no ML)."""
    category = item["category"]
    chart_cat = category if category in set(_mens_chart["category"]) else "tshirt"
    band = profile.get("bust_band")
    cup_ord = CUP_RANK.get(str(profile.get("bust_cup") or "").lower(), 3)
    # Reconstruct a chest estimate from band+cup (inverse of the band+cup
    # convention: chest inches ~ band + cup rank)
    chest_cm = ((band or 38) + cup_ord) * 2.54
    waist_cm = profile.get("waist_cm") or 80.0

    if category in MENS_BOTTOMS:
        value, col = waist_cm, "waist"
    else:
        value, col = chest_cm, "chest"
    rows = _mens_chart[_mens_chart["category"] == chart_cat].sort_values(f"{col}_min_cm")
    size = rows.iloc[-1]["size"]
    for _, row in rows.iterrows():
        if value <= row[f"{col}_max_cm"]:
            size = row["size"]
            break

    available = [s.strip() for s in item["size_range"].split(",")]
    if size not in available:
        size = available[-1] if size == "XXL" else available[0]
    return {
        "recommended_size": size,
        "anchor_size": size,
        "adjustment": None,
        "confidence": 75,  # fixed demo confidence — rule lookup has no probability
        "state": "blue",
        "probabilities": None,
        "method": "uniqlo_chart_lookup",
        "borderline": None,
    }


def size_scale_factors(item: dict, selected: str, recommended: str):
    """Size-proportional rendering (locked decision): width/length scale
    ratios for compositing a non-recommended size, from the demo size chart."""
    if selected == recommended:
        return 1.0, 1.0
    if item["category"] in MENS_BOTTOMS or item["category"] in {"jeans", "skirt", "shorts", "slacks"}:
        chart = WOMENS_CHART_HIP
    else:
        chart = WOMENS_CHART_BUST
    sel = chart.get(selected)
    rec = chart.get(recommended)
    if not sel or not rec:
        return 1.0, 1.0
    width = sel / rec
    length = 1.0 + (width - 1.0) * 0.6  # length grows a bit less than width
    return width, length

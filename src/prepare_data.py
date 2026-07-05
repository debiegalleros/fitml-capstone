"""Phase 3: clean, harmonize, and feature-engineer the real fit dataset.

Combines ModCloth + RentTheRunway (different schemas, different units, different
category granularity) into one `model_ready` dataset aligned to the fields the
FitML app actually collects (height, weight, bust band/cup, hip, body type,
category, size ordered) -> fit_feedback (small/fit/large).

Design decisions (see notebooks/02_eda_feature_engineering.ipynb for the full
narrative + EDA that motivates each one):
  - "brand" = platform of origin (modcloth/renttherunway); neither raw dataset
    has a real garment-brand field.
  - waist_cm is dropped entirely: ~99% missing once both sources are combined
    (RentTheRunway never collected it), too sparse to impute defensibly.
  - hip_cm/weight_kg/body_type are structurally missing-not-at-random by
    source (RTR has no hips, ModCloth has no weight/body_type) -- imputed
    with source-aware medians/modes plus a `_missing` indicator flag so
    models can learn to discount them, rather than inventing cross-source
    values.
  - rating/quality/review text/age/rented_for are excluded from model
    features: they are post-purchase or not collected by the app's profile
    screen, so including them would leak information unavailable at
    prediction time or create a train/serve mismatch.
  - category_broad harmonizes ModCloth's 4 coarse tags with RentTheRunway's
    68 fine-grained tags into one consistent taxonomy; category_detail keeps
    the original RTR-only granularity for reference.
  - bust cup letters are mapped onto one ordinal scale using the standard
    US bra-sizing equivalence DD=E, DDD=F, DDDD=G.
"""
import re

import numpy as np
import pandas as pd

SEED = 42

MODCLOTH_PATH = "data/raw/clothing_fit/modcloth_final_data.json"
RENTTHERUNWAY_PATH = "data/raw/clothing_fit/renttherunway_final_data.json"
OUTPUT_PATH = "data/processed/model_ready.csv"

# ModCloth rows tagged with merchandising labels instead of a real garment
# category carry no usable category signal.
MODCLOTH_DROP_CATEGORIES = {"new", "sale", "wedding"}

CATEGORY_BROAD_MAP = {
    # dresses / one-piece
    "dress": "dresses", "gown": "dresses", "sheath": "dresses", "shift": "dresses",
    "maxi": "dresses", "mini": "dresses", "midi": "dresses", "frock": "dresses",
    "ballgown": "dresses", "shirtdress": "dresses", "jumpsuit": "dresses",
    "romper": "dresses", "caftan": "dresses", "kaftan": "dresses",
    # tops
    "top": "tops", "blouse": "tops", "shirt": "tops", "sweater": "tops",
    "cardigan": "tops", "tank": "tops", "cami": "tops", "sweatshirt": "tops",
    "sweatershirt": "tops", "pullover": "tops", "turtleneck": "tops", "tee": "tops",
    "t-shirt": "tops", "hoodie": "tops", "henley": "tops", "knit": "tops",
    "tunic": "tops", "crewneck": "tops", "buttondown": "tops",
    # bottoms
    "pants": "bottoms", "pant": "bottoms", "trouser": "bottoms", "trousers": "bottoms",
    "skirt": "bottoms", "skirts": "bottoms", "culottes": "bottoms", "culotte": "bottoms",
    "leggings": "bottoms", "legging": "bottoms", "jeans": "bottoms", "skort": "bottoms",
    "jogger": "bottoms", "sweatpants": "bottoms", "tight": "bottoms", "overalls": "bottoms",
    # outerwear
    "jacket": "outerwear", "coat": "outerwear", "down": "outerwear", "bomber": "outerwear",
    "suit": "outerwear", "cape": "outerwear", "poncho": "outerwear", "peacoat": "outerwear",
    "kimono": "outerwear", "trench": "outerwear", "parka": "outerwear",
    "blouson": "outerwear", "duster": "outerwear", "overcoat": "outerwear",
    "vest": "outerwear", "blazer": "outerwear",
    # ModCloth's own coarse labels map onto themselves
    "tops": "tops", "dresses": "dresses", "bottoms": "bottoms", "outerwear": "outerwear",
}

# Standard US bra-cup progression, unifying both datasets' notations via the
# documented equivalence DD=E, DDD=F, DDDD=G.
CUP_RANK = {
    "aa": 0, "a": 1, "b": 2, "c": 3, "d": 4,
    "dd/e": 5, "dd": 5,
    "ddd/f": 6, "ddd/e": 6, "f": 6,
    "dddd/g": 7, "g": 7,
    "h": 8, "i": 9, "j": 10, "k": 11,
}

PLAUSIBLE_RANGES = {
    "height_cm": (137, 213),   # 4'6" - 7'0"
    "weight_kg": (35, 180),
    "hip_cm": (60, 180),
    "bust_band": (20, 56),
}


def ft_in_to_cm(feet, inches=0):
    return feet * 30.48 + inches * 2.54


def parse_modcloth_height(s):
    if pd.isna(s):
        return np.nan
    m = re.match(r"(\d+)ft(?:\s*(\d+)in)?", str(s).strip())
    if not m:
        return np.nan
    feet = int(m.group(1))
    inches = int(m.group(2)) if m.group(2) else 0
    return ft_in_to_cm(feet, inches)


def parse_rtr_height(s):
    if pd.isna(s):
        return np.nan
    m = re.match(r"(\d+)'\s*(\d+)\"", str(s).strip())
    if not m:
        return np.nan
    return ft_in_to_cm(int(m.group(1)), int(m.group(2)))


def parse_rtr_weight(s):
    if pd.isna(s):
        return np.nan
    m = re.match(r"(\d+)lbs", str(s).strip())
    if not m:
        return np.nan
    return round(int(m.group(1)) * 0.453592, 1)


def parse_rtr_bust(s):
    """Returns (band:int, cup_token:str) or (nan, nan)."""
    if pd.isna(s):
        return np.nan, np.nan
    m = re.match(r"(\d+)(\D+)", str(s).strip())
    if not m:
        return np.nan, np.nan
    band = int(m.group(1))
    cup = m.group(2).rstrip("+").lower()
    return band, cup


def cup_to_ordinal(token):
    if pd.isna(token):
        return np.nan
    return CUP_RANK.get(str(token).lower(), np.nan)


def clip_to_plausible(series, col_name):
    lo, hi = PLAUSIBLE_RANGES[col_name]
    return series.where(series.between(lo, hi))


def map_category_broad(category):
    if pd.isna(category):
        return np.nan
    return CATEGORY_BROAD_MAP.get(str(category).strip().lower(), "other")


def build_modcloth_frame(mc: pd.DataFrame) -> pd.DataFrame:
    mc = mc[~mc["category"].str.lower().isin(MODCLOTH_DROP_CATEGORIES)].copy()

    out = pd.DataFrame(index=mc.index)
    out["source"] = "modcloth"
    out["category_detail"] = mc["category"]
    out["category_broad"] = mc["category"].map(map_category_broad)
    out["size_ordered"] = mc["size"]
    out["height_cm"] = mc["height"].map(parse_modcloth_height)
    out["weight_kg"] = np.nan  # not collected by ModCloth
    out["bust_band"] = mc["bra size"]
    out["bust_cup_ordinal"] = mc["cup size"].map(cup_to_ordinal)
    out["hip_cm"] = mc["hips"] * 2.54  # ModCloth hips are in inches
    out["body_type"] = np.nan  # not collected by ModCloth
    out["fit_feedback"] = mc["fit"]
    return out


def build_rtr_frame(rtr: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=rtr.index)
    out["source"] = "renttherunway"
    out["category_detail"] = rtr["category"]
    out["category_broad"] = rtr["category"].map(map_category_broad)
    out["size_ordered"] = rtr["size"]
    out["height_cm"] = rtr["height"].map(parse_rtr_height)
    out["weight_kg"] = rtr["weight"].map(parse_rtr_weight)

    band_cup = rtr["bust size"].map(parse_rtr_bust)
    out["bust_band"] = band_cup.map(lambda t: t[0])
    out["bust_cup_ordinal"] = band_cup.map(lambda t: cup_to_ordinal(t[1]))

    out["hip_cm"] = np.nan  # not collected by RentTheRunway
    out["body_type"] = rtr["body type"]
    out["fit_feedback"] = rtr["fit"]
    return out


def add_missing_indicators_and_impute(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["height_cm", "weight_kg", "hip_cm", "bust_band"]:
        df[col] = clip_to_plausible(df[col], col)

    # Structurally source-exclusive columns: flag before imputing so models
    # can learn "this value is invented, not measured" rather than treating
    # an imputed median as if it were real.
    for col in ["weight_kg", "hip_cm", "bust_band", "bust_cup_ordinal"]:
        df[f"{col}_missing"] = df[col].isna().astype(int)
        median = df[col].median()
        df[col] = df[col].fillna(median)

    # height_cm is rarely missing in either source; simple median impute, no
    # indicator needed.
    df["height_cm"] = df["height_cm"].fillna(df["height_cm"].median())

    # body_type: do not invent a protected-attribute-like label for rows
    # that never reported one (all of ModCloth). Keep as its own category.
    df["body_type"] = df["body_type"].fillna("not_reported")

    df["category_broad"] = df["category_broad"].fillna("other")

    return df


def build_model_ready() -> pd.DataFrame:
    modcloth = pd.read_json(MODCLOTH_PATH, lines=True)
    rtr = pd.read_json(RENTTHERUNWAY_PATH, lines=True)

    combined = pd.concat(
        [build_modcloth_frame(modcloth), build_rtr_frame(rtr)],
        ignore_index=True,
    )
    combined = combined.dropna(subset=["fit_feedback", "category_broad"])
    combined = add_missing_indicators_and_impute(combined)
    return combined


if __name__ == "__main__":
    df = build_model_ready()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df):,} rows x {df.shape[1]} cols to {OUTPUT_PATH}")
    print("\ndtypes:\n", df.dtypes)
    print("\nfit_feedback distribution:\n", df["fit_feedback"].value_counts(normalize=True).round(3))
    print("\nsource distribution:\n", df["source"].value_counts())
    print("\nremaining nulls:\n", df.isnull().sum()[df.isnull().sum() > 0])

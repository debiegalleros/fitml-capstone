"""Phase 4: men's synthetic catalog extension — rule-based, NOT a trained model.

The graded ML core (Phases 3/5/6) uses only the real women's dataset
(ModCloth + RentTheRunway). Both platforms sell women's apparel, so the demo
catalog's ~10-15 men's items need a separate sizing path. This module provides
it: a small synthetic men's customer set, size-labeled by a deterministic
lookup against Uniqlo's published men's body-measurement size chart
(transcribed to `data/raw/mens_size_charts.csv`).

Design decisions (see docs/scope_and_data_provenance.md for the full framing):
  - Real sizing standard, synthetic population: the size chart is Uniqlo's
    published one (real, citable); the customers are generated. No fit
    labels are invented — `size_label` is just "which chart row do these
    measurements fall in", a lookup any store associate could do by hand.
  - NOT a trained model: no fitting, no train/test split, no learned
    parameters. `lookup_mens_size()` is the entire "model".
  - EXCLUDED from the Phase 6 fairness audit: the population is synthetic
    (group comparisons would audit the random generator, not any real-world
    disparity) and far too small for valid group statistics. Documented as
    an explicit limitation, not silently patched.
  - Kept in its own table (`data/processed/mens_synthetic.csv`, own schema)
    — never merged with the real women's `model_ready.csv`.
  - Tops (tshirt/polo/jacket) are sized by chest, bottoms (jeans/shorts) by
    waist, matching how Uniqlo's own guide keys each garment type.
  - Uniqlo's published ranges overlap by 1-3 cm at some size boundaries
    (e.g. waist S 69-76 vs M 76-84). The lookup resolves overlaps
    deterministically to the smaller size (first qualifying row, ascending);
    measurements below/above the chart clamp to the smallest/largest size.
  - Generator anthropometry (means/SDs/correlations below) is loosely based
    on adult Asian-male population figures — assumed, not measured, and
    documented as such. Correlated sampling (multivariate normal) avoids
    impossible bodies like a 76 cm chest with a 110 cm waist.
"""
import numpy as np
import pandas as pd

SEED = 42
N_CUSTOMERS = 200

CHART_PATH = "data/raw/mens_size_charts.csv"
OUTPUT_PATH = "data/processed/mens_synthetic.csv"

TOPS = {"tshirt", "polo", "jacket"}  # sized by chest
BOTTOMS = {"jeans", "shorts"}        # sized by waist
CATEGORIES = sorted(TOPS | BOTTOMS)

# (height_cm, chest_cm, waist_cm, hip_cm) — assumed adult Asian-male
# anthropometry for the synthetic population, chosen so the bulk of samples
# lands inside Uniqlo's XS-XXL chart ranges.
MEASURE_COLS = ["height_cm", "chest_cm", "waist_cm", "hip_cm"]
MEANS = np.array([168.5, 95.0, 82.0, 95.0])
SDS = np.array([6.5, 7.0, 9.0, 6.5])
CORR = np.array([
    [1.00, 0.40, 0.30, 0.35],
    [0.40, 1.00, 0.80, 0.75],
    [0.30, 0.80, 1.00, 0.80],
    [0.35, 0.75, 0.80, 1.00],
])

# Hard physical-plausibility clip, same spirit as PLAUSIBLE_RANGES in
# prepare_data.py.
PLAUSIBLE_RANGES = {
    "height_cm": (150, 200),
    "chest_cm": (75, 130),
    "waist_cm": (60, 115),
    "hip_cm": (80, 125),
}


def load_chart(path: str = CHART_PATH) -> pd.DataFrame:
    chart = pd.read_csv(path)
    expected = {"category", "brand", "size", "chest_min_cm", "chest_max_cm",
                "waist_min_cm", "waist_max_cm"}
    missing = expected - set(chart.columns)
    if missing:
        raise ValueError(f"size chart missing columns: {missing}")
    return chart


def lookup_mens_size(chart: pd.DataFrame, category: str,
                     chest_cm: float, waist_cm: float) -> str:
    """Rule-based size lookup against the Uniqlo chart — the entire 'model'.

    Tops are keyed on chest, bottoms on waist. Returns the first (smallest)
    size whose published range contains the measurement; out-of-chart
    measurements clamp to the smallest/largest size.
    """
    if category in TOPS:
        value, col = chest_cm, "chest"
    elif category in BOTTOMS:
        value, col = waist_cm, "waist"
    else:
        raise ValueError(f"unknown men's category: {category!r}")

    rows = chart[chart["category"] == category].sort_values(f"{col}_min_cm")
    if rows.empty:
        raise ValueError(f"no chart rows for category: {category!r}")
    for _, row in rows.iterrows():
        if value <= row[f"{col}_max_cm"]:
            return row["size"]
    return rows.iloc[-1]["size"]


def generate_synthetic_customers(n: int = N_CUSTOMERS,
                                 seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cov = np.outer(SDS, SDS) * CORR
    samples = rng.multivariate_normal(MEANS, cov, size=n)

    df = pd.DataFrame(samples, columns=MEASURE_COLS).round(1)
    for col, (lo, hi) in PLAUSIBLE_RANGES.items():
        df[col] = df[col].clip(lo, hi)

    df.insert(0, "customer_id", [f"synth_m_{i:03d}" for i in range(n)])
    df["category"] = rng.choice(CATEGORIES, size=n)
    return df


def build_mens_extension() -> pd.DataFrame:
    chart = load_chart()
    df = generate_synthetic_customers()
    df["size_label"] = df.apply(
        lambda r: lookup_mens_size(chart, r["category"],
                                   r["chest_cm"], r["waist_cm"]),
        axis=1,
    )
    return df


def _sanity_checks(chart: pd.DataFrame) -> None:
    # Chart midpoints must map back to their own size, per category.
    for _, row in chart.iterrows():
        mid_chest = (row["chest_min_cm"] + row["chest_max_cm"]) / 2
        mid_waist = (row["waist_min_cm"] + row["waist_max_cm"]) / 2
        got = lookup_mens_size(chart, row["category"], mid_chest, mid_waist)
        assert got == row["size"], (
            f"{row['category']} midpoint {mid_chest}/{mid_waist} -> {got}, "
            f"expected {row['size']}"
        )
    # Out-of-chart values clamp instead of erroring.
    assert lookup_mens_size(chart, "tshirt", 60, 60) == "XS"
    assert lookup_mens_size(chart, "jeans", 150, 150) == "XXL"
    # Boundary overlap resolves to the smaller size.
    assert lookup_mens_size(chart, "jeans", chest_cm=97, waist_cm=76) == "S"
    print("Sanity checks passed: midpoints round-trip, clamping and "
          "overlap resolution behave as documented.")


if __name__ == "__main__":
    chart = load_chart()
    _sanity_checks(chart)

    df = build_mens_extension()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(df)} rows x {df.shape[1]} cols to {OUTPUT_PATH}")
    print("\nsize_label distribution:\n", df["size_label"].value_counts())
    print("\ncategory distribution:\n", df["category"].value_counts())
    print("\nmeasurement summary:\n",
          df[MEASURE_COLS].describe().round(1))

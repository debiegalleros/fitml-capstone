"""Phase 7 — select catalog candidates from the Fashion Product Images dataset.

Filters styles.csv for contemporary basics (solid colors, clean cuts, minimal
branding) across the target women's/men's categories, ranks newest-first, and
writes data/catalog/candidates.csv with per-category headroom so unusable
images found during download/QA can be replaced without re-running selection.

Note on years: apparel in the target categories is overwhelmingly 2011-2012
(2015+ has almost no women's dresses/jeans/skirts), so "most recent" is a
ranking preference, not a hard cutoff — the solid-color / clean-cut filters
carry the contemporary aesthetic instead.
"""

import argparse
import random
from pathlib import Path

import pandas as pd

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
STYLES_CSV = ROOT / "data/raw/fashion_product_images/styles.csv"
OUT_CSV = ROOT / "data/catalog/candidates.csv"

# Per-category quotas. Women ~112 items; men 13 across the four categories
# covered by data/raw/mens_size_charts.csv (tshirt/polo/jeans/jacket) so every
# men's item is sizeable by the Uniqlo chart lookup.
WOMEN_QUOTAS = {
    "Tshirts": 14, "Tops": 14, "Dresses": 14, "Shirts": 10, "Jeans": 12,
    "Skirts": 10, "Trousers": 8, "Sweaters": 8, "Shorts": 8, "Jackets": 8,
    "Camisoles": 6,
}
MEN_QUOTAS = {"Tshirts": 4, "Polo": 3, "Jeans": 3, "Jackets": 3}
HEADROOM = 2.0  # keep 2x quota as ranked candidates

# Names suggesting prints/patterns/heavy branding — excluded for the
# contemporary-basics aesthetic (and hue-shifting works best on solids).
PATTERN_KEYWORDS = (
    "printed|print|floral|graphic|sequin|embroider|embellish|checked|check|"
    "striped|stripe|polka|paisley|animal|camouflage|slogan|typography|"
    # learned during Phase 7 QA: multi-pack product shots and more prints
    r"\bpack\b|pack of|lace|\bdot\b|dotted|plaid|patchwork"
)

# styles.csv occasionally mislabels kids' items as Women/Men (e.g. 13263
# "Palm Tree Kids Girl ... Skirts" with gender=Women) — exclude by name.
KIDS_KEYWORDS = r"\bkids?\b|\bgirls?\b|\bboys?\b|\binfants?\b|\bbaby\b"

SOLID_COLOURS = {
    "Black", "White", "Off White", "Grey", "Charcoal", "Grey Melange", "Navy Blue",
    "Blue", "Red", "Maroon", "Burgundy", "Green", "Olive", "Teal", "Pink", "Peach",
    "Purple", "Lavender", "Yellow", "Mustard", "Orange", "Brown", "Coffee Brown",
    "Beige", "Cream", "Khaki", "Tan",
}

USAGES = {"Casual", "Formal", "Smart Casual"}


def rank_and_take(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Rank: explicitly 'Solid'-named first, then newest year, then random."""
    rng = random.Random(SEED)
    df = df.assign(
        is_solid_named=df.productDisplayName.str.contains("solid", case=False),
        _tiebreak=[rng.random() for _ in range(len(df))],
    )
    df = df.sort_values(
        ["is_solid_named", "year", "_tiebreak"], ascending=[False, False, False]
    )
    return df.head(n).drop(columns="_tiebreak")


def filtered_pool() -> pd.DataFrame:
    df = pd.read_csv(STYLES_CSV, on_bad_lines="skip")
    return df[
        (df.masterCategory == "Apparel")
        & df.usage.isin(USAGES)
        & df.baseColour.isin(SOLID_COLOURS)
        & df.productDisplayName.notna()
        & ~df.productDisplayName.str.contains(PATTERN_KEYWORDS, case=False)
        & ~df.productDisplayName.str.contains(KIDS_KEYWORDS, case=False)
    ]


def men_pools(ap: pd.DataFrame) -> dict[str, pd.DataFrame]:
    men = ap[ap.gender == "Men"]
    is_polo = men.productDisplayName.str.contains("polo", case=False)
    return {
        "Tshirts": men[(men.articleType == "Tshirts") & ~is_polo],
        "Polo": men[(men.articleType == "Tshirts") & is_polo],
        "Jeans": men[men.articleType == "Jeans"],
        "Jackets": men[men.articleType == "Jackets"],
    }


def extend() -> None:
    """Top up shortfall categories with fresh candidates.

    Appends new rows (ranked after the existing max rank per category) for
    every category whose non-excluded candidate count is below quota, taking
    3x the shortfall as new headroom. Existing rows are left untouched so
    prior QA verdicts keep their meaning.
    """
    existing = pd.read_csv(OUT_CSV)
    excluded = set()
    excl_csv = ROOT / "src/catalog_exclusions.csv"
    if excl_csv.exists():
        excluded = set(pd.read_csv(excl_csv).id.astype(int))
    ap = filtered_pool()
    ap = ap[~ap.id.isin(set(existing.id))]
    mpools = men_pools(ap)

    additions = []
    for (gender, article), quota in {
        **{("women", k): v for k, v in WOMEN_QUOTAS.items()},
        **{("men", k): v for k, v in MEN_QUOTAS.items()},
    }.items():
        have = existing[(existing.target_gender == gender)
                        & (existing.articleType == article)
                        & ~existing.id.isin(excluded)]
        shortfall = quota - len(have)
        if shortfall <= 0:
            continue
        pool = (mpools[article] if gender == "men"
                else ap[(ap.gender == "Women") & (ap.articleType == article)])
        take = rank_and_take(pool, shortfall * 3)
        max_rank = existing[(existing.target_gender == gender)
                            & (existing.articleType == article)].rank_in_category.max()
        take = take.assign(target_gender=gender, quota=quota)
        take.articleType = article
        take["rank_in_category"] = range(int(max_rank) + 1,
                                         int(max_rank) + 1 + len(take))
        take["selected"] = False  # legacy column; active set comes from ranks
        additions.append(take)
        print(f"{gender}/{article}: shortfall {shortfall}, "
              f"pool {len(pool)}, adding {len(take)}")

    if additions:
        out = pd.concat([existing, *additions], ignore_index=True)
        out.to_csv(OUT_CSV, index=False)
        print(f"\ncandidates.csv: {len(existing)} -> {len(out)} rows")
    else:
        print("no shortfalls")


def main() -> None:
    ap = filtered_pool()

    picks = []
    for article, quota in WOMEN_QUOTAS.items():
        pool = ap[(ap.gender == "Women") & (ap.articleType == article)]
        take = rank_and_take(pool, int(quota * HEADROOM))
        take = take.assign(target_gender="women", quota=quota)
        picks.append(take)

    mpools = men_pools(ap)
    for article, quota in MEN_QUOTAS.items():
        take = rank_and_take(mpools[article], int(quota * HEADROOM))
        take = take.assign(target_gender="men", quota=quota)
        take.articleType = article  # label polos distinctly
        picks.append(take)

    out = pd.concat(picks, ignore_index=True)
    out["rank_in_category"] = out.groupby(["target_gender", "articleType"]).cumcount() + 1
    out["selected"] = out.rank_in_category <= out.quota

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    sel = out[out.selected]
    print(f"candidates: {len(out)} | selected: {len(sel)} "
          f"(women {len(sel[sel.target_gender=='women'])}, "
          f"men {len(sel[sel.target_gender=='men'])})")
    print("\nselected per category:")
    print(sel.groupby(["target_gender", "articleType"]).agg(
        n=("id", "size"), solid_named=("is_solid_named", "sum"),
        newest=("year", "max"), oldest=("year", "min")).to_string())
    print("\npool shortfalls (selected < quota):")
    short = sel.groupby(["target_gender", "articleType"]).size()
    for (g, a), q in {**{("women", k): v for k, v in WOMEN_QUOTAS.items()},
                      **{("men", k): v for k, v in MEN_QUOTAS.items()}}.items():
        if short.get((g, a), 0) < q:
            print(f"  {g}/{a}: {short.get((g, a), 0)}/{q}")
    else:
        print("  (none)" if all(short.get(k, 0) >= q for k, q in
              {**{("women", k): v for k, v in WOMEN_QUOTAS.items()},
               **{("men", k): v for k, v in MEN_QUOTAS.items()}}.items()) else "")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--extend", action="store_true",
                        help="append candidates for shortfall categories only")
    if parser.parse_args().extend:
        extend()
    else:
        main()

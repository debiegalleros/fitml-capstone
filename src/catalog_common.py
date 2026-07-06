"""Shared Phase 7 helper: the active catalog selection.

The active selection is the per-category top-quota slice of
data/catalog/candidates.csv after removing hand-curated QA exclusions
(src/catalog_exclusions.csv) — excluded items are automatically replaced by
the next-ranked headroom candidate in the same category.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CANDIDATES_CSV = ROOT / "data/catalog/candidates.csv"
EXCLUSIONS_CSV = ROOT / "src/catalog_exclusions.csv"


def active_selection() -> pd.DataFrame:
    df = pd.read_csv(CANDIDATES_CSV)
    if EXCLUSIONS_CSV.exists():
        excluded = set(pd.read_csv(EXCLUSIONS_CSV).id.astype(int))
        df = df[~df.id.isin(excluded)]
    df = df.sort_values("rank_in_category")
    kept = df.groupby(["target_gender", "articleType"]).cumcount() < df.quota
    return df[kept]

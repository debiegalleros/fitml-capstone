"""Phase 2: load and inspect the real fit dataset. No cleaning — inspection only."""
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)

MODCLOTH_PATH = "data/raw/clothing_fit/modcloth_final_data.json"
RENTTHERUNWAY_PATH = "data/raw/clothing_fit/renttherunway_final_data.json"


def inspect(name, df):
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
    print(f"shape: {df.shape}")
    print("\ndtypes:")
    print(df.dtypes)
    print("\nnull counts (non-zero only):")
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0].sort_values(ascending=False)
    print(nulls if len(nulls) else "  (none)")
    print("\nnull percentage (non-zero only):")
    pct = (df.isnull().mean() * 100).round(1)
    pct = pct[pct > 0].sort_values(ascending=False)
    print(pct if len(pct) else "  (none)")
    print("\nsample rows:")
    print(df.head(3))
    for col in df.select_dtypes(include=["object", "str"]).columns:
        nunique = df[col].nunique()
        if nunique <= 15:
            print(f"\nvalue_counts for '{col}' ({nunique} unique):")
            print(df[col].value_counts(dropna=False))


if __name__ == "__main__":
    modcloth = pd.read_json(MODCLOTH_PATH, lines=True)
    renttherunway = pd.read_json(RENTTHERUNWAY_PATH, lines=True)

    inspect("ModCloth", modcloth)
    inspect("RentTheRunway", renttherunway)

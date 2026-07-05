# Data Dictionary — `data/processed/model_ready.csv`

Produced by [`src/prepare_data.py`](../src/prepare_data.py) from the Kaggle
"Clothing Fit Dataset for Size Recommendation" (ModCloth + RentTheRunway).
See [`notebooks/02_eda_feature_engineering.ipynb`](../notebooks/02_eda_feature_engineering.ipynb)
for the full EDA and reasoning behind each cleaning/encoding decision.

251,047 rows (58,503 ModCloth + 192,544 RentTheRunway) x 15 columns.
No nulls remain — see "Missing-value handling" below for how each source-
exclusive gap was resolved.

## Columns

| Column | Type | Description | Source |
|---|---|---|---|
| `source` | categorical | `modcloth` / `renttherunway` — stands in for "brand" per the Master Plan, since neither raw dataset has an actual garment-brand field | derived |
| `category_detail` | categorical | Original garment category string (68 distinct values for RentTheRunway, 4 for ModCloth) | real |
| `category_broad` | categorical | `tops` / `dresses` / `bottoms` / `outerwear` / `other` — RentTheRunway's 68 fine-grained tags mapped onto ModCloth's 4-bucket taxonomy for cross-source consistency | derived |
| `size_ordered` | int | The garment size the customer ordered (0–58 across both platforms; scale isn't standardized across brands, kept as ordered numeric input) | real |
| `height_cm` | float | Parsed from `"5ft 6in"` (ModCloth) / `"5' 8\""` (RentTheRunway); median-imputed for the ~1% missing (no indicator — too rare to matter) | real |
| `weight_kg` | float | Parsed from `"137lbs"`; **RentTheRunway only** — ModCloth never collected it, so imputed to the global median for all ModCloth rows. Use `weight_kg_missing` to identify invented values | real (RTR only) |
| `bust_band` | float | Band size (e.g. 34) — from ModCloth's `bra size` or parsed from RentTheRunway's `bust size` (e.g. `"34d"` → 34) | real |
| `bust_cup_ordinal` | float | Cup letter mapped to an ordinal scale (AA=0, A=1, B=2, ... K=11) using the standard US bra-sizing equivalence DD=E, DDD=F, DDDD=G to unify both datasets' notations | real |
| `hip_cm` | float | ModCloth's `hips` (inches) x 2.54. **ModCloth only** — RentTheRunway never collected it, so imputed to the global median for all RentTheRunway rows. Use `hip_cm_missing` to identify invented values | real (ModCloth only) |
| `body_type` | categorical | `hourglass` / `pear` / `apple` / `athletic` / `petite` / `straight & narrow` / `full bust` / `not_reported`. **RentTheRunway only** — ModCloth rows are always `not_reported` (never invented, since this is a protected-attribute-like field relevant to the Phase 6 fairness audit) | real (RTR only) |
| `fit_feedback` | categorical (target) | `small` / `fit` / `large` — the actual customer-reported fit outcome | real |
| `weight_kg_missing` | int (0/1) | 1 if `weight_kg` was imputed (100% of ModCloth, 15.6% of RentTheRunway) | derived |
| `hip_cm_missing` | int (0/1) | 1 if `hip_cm` was imputed (100% of RentTheRunway, 32.3% of ModCloth) | derived |
| `bust_band_missing` | int (0/1) | 1 if `bust_band` was imputed | derived |
| `bust_cup_ordinal_missing` | int (0/1) | 1 if `bust_cup_ordinal` was imputed | derived |

## Excluded from the raw source data

- **`waist_cm`** — dropped entirely. RentTheRunway never collected it, and
  ModCloth's own `waist` field was already 96.5% null, so combined coverage
  is ~1% — too sparse to impute defensibly.
- **`rating` / `quality`** (post-purchase satisfaction scores) — excluded as
  model features: they wouldn't be available at prediction time for someone
  browsing pre-purchase, so including them would be feature leakage relative
  to the deployed model's actual use case.
- **`age`, `rented_for`** (RentTheRunway only) — excluded because the app's
  profile-setup screen doesn't collect them; including them in training
  would create a train/serve mismatch with the live `/recommend-size`
  endpoint.
- **`review_text`, `review_summary`, `user_name`, `user_id`, `item_id`** —
  free text / identifiers, not modeling features.

## Missing-value handling

Two different patterns show up, handled differently:

1. **Rare, scattered missingness** (`height_cm`, ~1%): plain median
   imputation, no indicator needed.
2. **Structural, source-exclusive missingness** (`weight_kg`, `hip_cm`,
   `bust_band`, `bust_cup_ordinal`, `body_type`): one whole platform never
   collected the field at all. Rather than inventing cross-source values
   silently, each gets a `_missing` indicator flag (except `body_type`,
   which gets an explicit `not_reported` category instead of a guessed
   shape) so downstream models — and the Phase 6 fairness audit — can tell
   a measured value from an imputed placeholder.

## Category harmonization

ModCloth's raw `category` includes non-garment merchandising tags
(`new`, `sale`, `wedding`) with no garment-type signal — these 24,287 rows
(~29% of ModCloth, ~8.8% of the combined raw total) are dropped before the
frame is built. RentTheRunway's 68 fine-grained tags (`dress`, `gown`,
`sheath`, `romper`, ...) are mapped onto the same 4-bucket taxonomy
(`tops`/`dresses`/`bottoms`/`outerwear`) ModCloth already used; 3 residual
garbage tags (`print`, `combo`, `for`, 118 rows total) fall into `other`.

## Not yet applied (by design)

- **Scaling** — deliberately not baked into `model_ready.csv`. Fitting a
  `StandardScaler` on the full dataset here and saving scaled values would
  leak test-split statistics into training. Phase 5 fits scaling inside a
  train-only pipeline (`sklearn` `Pipeline`/`ColumnTransformer`).
- **One-hot encoding** of `source` / `category_broad` / `body_type` — shown
  in the EDA notebook for exploratory feature-importance ranking, but not
  saved to the CSV, so Phase 5 can encode consistently within its own
  pipeline.

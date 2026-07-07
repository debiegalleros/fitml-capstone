# Phase 8 Notes — Model-Input Spec for the Flask Backend

Everything below was **re-derived from the fitted pipeline**
(`models/xgboost_weighted.joblib`, its `ColumnTransformer`, and
`models/label_encoder.joblib`) on 2026-07-07 — not copied from planning docs.

## Served model

`models/xgboost_weighted.joblib` — class-weighted XGBoost inside an sklearn
`Pipeline(preprocess → model)`. The pipeline owns scaling and one-hot encoding;
the backend passes a **raw-feature DataFrame** and never pre-scales/encodes.
Predicted class ids decode via `models/label_encoder.joblib`:
`['fit', 'large', 'small']`.

## Input: 13 columns, in this exact order

| # | Column | Type | Notes |
|---|--------|------|-------|
| 1 | `size_ordered` | int | numeric size being evaluated (train range 0–58) |
| 2 | `height_cm` | float | |
| 3 | `weight_kg` | float | impute median **61.2** when absent |
| 4 | `bust_band` | int | US band number (train range 28–48) |
| 5 | `bust_cup_ordinal` | int | cup rank, see mapping below |
| 6 | `hip_cm` | float | impute median **99.06** when absent |
| 7 | `weight_kg_missing` | 0/1 | 1 when weight was imputed |
| 8 | `hip_cm_missing` | 0/1 | 1 when hip was imputed |
| 9 | `bust_band_missing` | 0/1 | 1 when band was imputed (median 34) |
| 10 | `bust_cup_ordinal_missing` | 0/1 | 1 when cup was imputed (median 3 = C) |
| 11 | `source` | str | categorical, vocabulary below |
| 12 | `category_broad` | str | categorical, vocabulary below |
| 13 | `body_type` | str | categorical, vocabulary below |

Cup mapping (`CUP_RANK` in `src/prepare_data.py`): aa=0, a=1, b=2, c=3, d=4,
dd/e=5, ddd/f=6, dddd/g=7, h=8, i=9, j=10, k=11.

## Fitted categorical vocabularies (from the fitted OneHotEncoder)

- `source`: `modcloth`, `renttherunway`
- `category_broad`: `bottoms`, `dresses`, `other`, `outerwear`, `tops`
- `body_type`: `apple`, `athletic`, `full bust`, `hourglass`, `not_reported`,
  `pear`, `petite`, `straight & narrow`

## Required: input-validation layer in /recommend-size

The fitted OneHotEncoder uses `handle_unknown='ignore'`, so an unseen category
does **not** crash — but it silently encodes as all-zeros, which degrades the
prediction without any signal. The endpoint must therefore map user inputs to
fitted values *before* calling the pipeline:

- `body_type`: the profile dropdown offers
  hourglass/pear/apple/rectangle/athletic/petite. Map `rectangle` →
  `straight & narrow`; anything else not in the vocabulary → `not_reported`.
- `category_broad`: map catalog categories onto the five fitted values
  (jeans/skirt/shorts/trousers → `bottoms`, dress → `dresses`,
  jacket/sweater → `outerwear`, tshirt/blouse/top/shirt → `tops`,
  unknown → `other`).
- `source`: user profiles aren't from either site; default `renttherunway`
  when weight is provided, `modcloth` otherwise (matches each source's
  measurement schema).
- Numerics: when weight/hip/bust are absent, impute the training medians above
  and set the matching `_missing` flag to 1; clamp inputs to the plausible
  ranges in `src/prepare_data.py` (`PLAUSIBLE_RANGES`).

## How /recommend-size uses the model

For each candidate `size_ordered` in the garment's size range, build the
13-column row, predict fit/small/large probabilities, and recommend the size
with the highest P(fit). The displayed confidence is that P(fit); the
borderline rule (near size boundary + low-stretch fabric + fitted cut) then
lowers confidence, sizes up, and switches the box to the amber state.

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

## How /recommend-size uses the model (chart-anchored — IMPORTANT)

**Do not scan candidate sizes and argmax P(fit).** That was the first design,
and Phase 8 testing showed it inverts: a petite profile got XL at 90%
confidence and a large-bodied profile got XS at 99%. Cause: in the training
data `size_ordered` is the size the customer actually ordered, which tracks
their own body — so a (petite body, size 17) row is far out of distribution
and the model's counterfactual answer there is meaningless. It learned the
between-customer correlation (larger sizes are ordered by larger people, who
disproportionately report items running small), not the within-person effect
of changing size.

The in-distribution question the model *can* answer is: "for a customer with
these measurements ordering this size, does it run small / fit / large" —
that is literally what each training row records. So /recommend-size does:

1. **Anchor** the size deterministically from the measurements via the demo
   size chart (hip-keyed for bottoms, bust-keyed for tops/outerwear, the
   larger of the two for dresses).
2. **Predict** fit/small/large at that anchor with the weighted-XGBoost
   pipeline.
3. **Shift one size** if the model's top class is a misfit (small → size up,
   large → size down; never past a clamped chart end).
4. Apply the **borderline blue/amber layer** on top (near size boundary +
   low-stretch fabric + fitted cut → amber, size up, lowered confidence).

Displayed confidence is P(fit) at the final size (floored so a noisy tail
probability can't show an absurd number for a chart-grounded anchor).
Verified sane: petite → XS, average → M, large-bodied → XL.
This is worth a paragraph in the report (Step 4/9 discussion: correlation vs
causation in deployed models).

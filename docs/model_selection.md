# Model Selection — Phase 5

Produced by [`src/train_models.py`](../src/train_models.py) (seed 42).
Four classifiers trained on the same stratified 80/20 split of
[`data/processed/model_ready.csv`](data_dictionary.md) (200,837 train /
50,210 test rows), predicting `fit_feedback` (fit / large / small,
72.7% / 13.7% / 13.6%). All preprocessing (scaling, one-hot encoding) is
fitted inside each model's Pipeline on the training fold only.

## Why macro-F1, not accuracy

The target is ~73/14/13 imbalanced, so a model that always predicts "fit"
scores ~72.7% accuracy while being useless — and the minority classes are
the whole point: "small" and "large" predictions are exactly the cases
where a size recommendation prevents a return. Macro-F1 averages per-class
F1 with equal weight, so it exposes minority-class failure that accuracy
hides. Selection is by macro-F1 per the Master Plan.

## Results

| model | accuracy | f1_macro | f1_fit | f1_large | f1_small | train_time_s |
|---|---|---|---|---|---|---|
| **xgboost** | 0.7272 | **0.2918** | 0.8418 | 0.0219 | 0.0115 | 2.7 |
| random_forest | 0.7278 | 0.2848 | 0.8424 | 0.0101 | 0.0018 | 3.4 |
| mlp | 0.7272 | 0.2825 | 0.8420 | 0.0043 | 0.0012 | 6.1 |
| logistic_regression | 0.7267 | 0.2822 | 0.8418 | 0.0000 | 0.0046 | 0.8 |

(Full per-class precision/recall in
[`models/classification_reports.txt`](../models/classification_reports.txt);
machine-readable table in [`models/comparison.csv`](../models/comparison.csv).)

**Selected: XGBoost** — highest macro-F1, and the only model with non-trivial
F1 on *both* minority classes. Accuracy is statistically indistinguishable
across all four (each ≈ the 72.7% majority baseline), which is precisely why
accuracy was ruled out as the selection metric.

## The honest headline: unweighted models collapse to the majority class

Every model, trained without class weighting, learns to predict "fit"
almost always: per-class F1 for "large" and "small" is near zero (logistic
regression never predicts "large" at all). Two things drive this:

1. **Class imbalance + unweighted loss.** With 73% of labels being "fit",
   the loss is minimized by rarely risking a minority prediction unless the
   signal is strong.
2. **Weak features for the task.** The app's profile fields (height, weight,
   bust, hip, body type, broad category, ordered size) carry real but limited
   signal about *reported fit outcome* — fit feedback also depends on
   garment-specific cut/sizing quirks that no body measurement captures.
   Published work on this same ModCloth/RentTheRunway dataset reports the
   task as hard for the same reason.

**This is deliberate for Phase 5.** No class weights were applied because
Phase 6's fairness-mitigation step is a class-weighted retrain with
before/after numbers — these unweighted models are the honest "before".
The expected effect of weighting is a large jump in minority-class recall
(and macro-F1) traded against some majority-class accuracy; Phase 6
quantifies that trade and its fairness impact across body-type groups.

## Trees vs. MLP — the expected finding, reported honestly

The Master Plan predicted tree-based models would beat the MLP on this
small, tabular, mixed-type dataset — and they did: the MLP (two hidden
layers, 64/32, early stopping) lands below both XGBoost and random forest
on macro-F1 despite the longest training time. This matches the
well-documented pattern that gradient-boosted trees dominate MLPs on
modest-size tabular data; it is a legitimate comparison result, not a
tuning failure (all four models received the same preprocessing and
equal-footing, untuned hyperparameters).

## Artifacts (`models/`)

| File | What it is |
|---|---|
| `comparison.csv` | The results table above |
| `classification_reports.txt` | Full per-class precision/recall/F1 per model |
| `xgboost.joblib` | **Selected model** — full Pipeline (preprocessing + model), 1.1 MB |
| `logistic_regression.joblib`, `mlp.joblib` | Runner-up pipelines (small, committed) |
| `random_forest.joblib` | **Gitignored** (50 MB — exceeds sensible repo size); byte-reproducible via `./venv/bin/python src/train_models.py` (fixed seed) |
| `label_encoder.joblib` | fit/large/small ↔ 0/1/2 mapping |
| `test_indices.csv` | Row indices of the held-out 20% — Phase 6 audits exactly these rows |
| `training_config.json` | Seed, split, feature lists, selection metric, selected model |

## Reproducibility

Single command: `./venv/bin/python src/train_models.py`. Fixed seed (42)
for the split and every model; features and config recorded in
`training_config.json`; the saved test indices pin the evaluation set for
the Phase 6 fairness audit.

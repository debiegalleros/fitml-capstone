"""Phase 5: train and compare LogReg / RF / XGBoost / MLP on the real dataset.

All four models train on the same stratified 80/20 split of
`data/processed/model_ready.csv`, predicting fit_feedback (fit/large/small),
and are compared on the same held-out test set.

Design decisions:
  - Selection metric is macro-F1, not accuracy: the target is ~73/14/13
    imbalanced, so a majority-class guesser already scores ~73% accuracy
    while being useless. Macro-F1 weights the two minority classes (the
    costly mispredictions — "small"/"large" are exactly the cases that
    cause returns) equally with "fit".
  - NO class weighting at this stage, deliberately: Phase 6's fairness
    mitigation is a class-weighted retrain with before/after numbers, so
    the Phase 5 models are the honest unweighted "before".
  - Scaling and one-hot encoding happen inside each model's Pipeline,
    fitted on the training fold only — model_ready.csv ships unscaled
    precisely to avoid test-split leakage (see docs/data_dictionary.md,
    "Not yet applied (by design)").
  - category_detail is excluded from features: it is the RTR-only
    68-value original tag kept in the CSV for reference; category_broad
    carries the harmonized signal both sources share.
  - The test-set row indices are saved to models/test_indices.csv so the
    Phase 6 fairness audit stratifies the exact same held-out rows the
    comparison numbers come from.
  - Modest, regularized hyperparameters throughout (no per-model tuning
    sweep): the comparison is between model families on equal footing,
    and unbounded trees on 200k rows would also produce multi-hundred-MB
    artifacts. Fixed SEED everywhere for reproducibility.
  - MLP is a small two-layer net (64, 32) with early stopping — per the
    Master Plan, tree models beating the MLP on small tabular data is an
    expected, reportable outcome, not a failure.
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

SEED = 42
TEST_SIZE = 0.2

DATA_PATH = "data/processed/model_ready.csv"
MODELS_DIR = Path("models")

TARGET = "fit_feedback"
NUMERIC_FEATURES = [
    "size_ordered", "height_cm", "weight_kg", "bust_band",
    "bust_cup_ordinal", "hip_cm",
    "weight_kg_missing", "hip_cm_missing", "bust_band_missing",
    "bust_cup_ordinal_missing",
]
CATEGORICAL_FEATURES = ["source", "category_broad", "body_type"]


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


def build_models() -> dict:
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=SEED),
        "random_forest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=10, n_jobs=-1, random_state=SEED,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            objective="multi:softprob", eval_metric="mlogloss",
            tree_method="hist", n_jobs=-1, random_state=SEED,
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=200,
            early_stopping=True, n_iter_no_change=10, random_state=SEED,
        ),
    }


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[TARGET])
    class_names = list(label_encoder.classes_)

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=TEST_SIZE, stratify=y, random_state=SEED,
    )
    pd.Series(idx_test, name="row_index").to_csv(
        MODELS_DIR / "test_indices.csv", index=False,
    )
    joblib.dump(label_encoder, MODELS_DIR / "label_encoder.joblib")

    rows = []
    reports = []
    for name, model in build_models().items():
        pipeline = Pipeline([
            ("preprocess", build_preprocessor()),
            ("model", model),
        ])
        t0 = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - t0

        y_pred = pipeline.predict(X_test)
        per_class_f1 = f1_score(y_test, y_pred, average=None)
        row = {
            "model": name,
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "f1_macro": round(f1_score(y_test, y_pred, average="macro"), 4),
            **{f"f1_{cls}": round(f, 4) for cls, f in zip(class_names, per_class_f1)},
            "train_time_s": round(train_time, 1),
        }
        rows.append(row)
        reports.append(
            f"=== {name} ===\n"
            + classification_report(y_test, y_pred, target_names=class_names, digits=4)
        )
        joblib.dump(pipeline, MODELS_DIR / f"{name}.joblib", compress=3)
        print(f"{name}: acc={row['accuracy']} f1_macro={row['f1_macro']} "
              f"({train_time:.0f}s)")

    comparison = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
    comparison.to_csv(MODELS_DIR / "comparison.csv", index=False)
    (MODELS_DIR / "classification_reports.txt").write_text("\n\n".join(reports))

    winner = comparison.iloc[0]["model"]
    config = {
        "seed": SEED,
        "test_size": TEST_SIZE,
        "data_path": DATA_PATH,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "target": TARGET,
        "classes": class_names,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "selection_metric": "f1_macro",
        "selected_model": winner,
    }
    (MODELS_DIR / "training_config.json").write_text(json.dumps(config, indent=2))

    print("\n", comparison.to_string(index=False))
    print(f"\nSelected by f1_macro: {winner}")


if __name__ == "__main__":
    main()

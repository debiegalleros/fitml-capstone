"""Phase 6: fairness & bias audit of the selected model (XGBoost, Phase 5).

Audits the Phase 5 winner on the exact held-out test rows pinned in
models/test_indices.csv (the script re-derives the split with the same
seed and asserts the indices match before doing anything else).

What it produces:
  - Per-group performance (accuracy, per-class recall = TPR) across three
    groupings the real data supports: body_type, height band, and ordered-
    size band.
  - Disparate impact ratio per group. Favorable outcome = the model
    predicts "fit" (the shopper gets a confident go-ahead; a small/large
    prediction routes them into size-adjustment friction). DI_g =
    P(pred=fit | group) / P(pred=fit | reference group), reference = the
    group with the highest predicted-fit rate. DI < 0.8 is flagged per
    the four-fifths rule.
  - Equalized odds: per-class TPR and FPR by group, with max-minus-min
    spreads across groups.
  - SHAP (TreeExplainer) on the winning model: global mean-|SHAP| bar
    plot plus per-class beeswarm plots, saved to docs/assets/fairness/.
  - Mitigation: retrain the identical XGBoost pipeline on the identical
    train fold with balanced sample weights (inverse class frequency),
    then rerun the full audit for before/after comparison. Saved to
    models/xgboost_weighted.joblib.

All tables are emitted as GitHub-flavored markdown fragments into
docs/assets/fairness/audit_tables.md so the numbers in
docs/fairness_report.md are copied from computed output, never retyped.
"""
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

import shap

SEED = 42
TEST_SIZE = 0.2
DATA_PATH = "data/processed/model_ready.csv"
MODELS_DIR = Path("models")
OUT_DIR = Path("docs/assets/fairness")
SHAP_SAMPLE = 3000

TARGET = "fit_feedback"
NUMERIC_FEATURES = [
    "size_ordered", "height_cm", "weight_kg", "bust_band",
    "bust_cup_ordinal", "hip_cm",
    "weight_kg_missing", "hip_cm_missing", "bust_band_missing",
    "bust_cup_ordinal_missing",
]
CATEGORICAL_FEATURES = ["source", "category_broad", "body_type"]
FAVORABLE = "fit"


def height_band(h: float) -> str:
    if h < 160:
        return "under 160 cm"
    if h < 170:
        return "160-169 cm"
    return "170 cm and over"


def size_band(s: float) -> str:
    if s <= 6:
        return "size 0-6"
    if s <= 14:
        return "size 8-14"
    if s <= 22:
        return "size 16-22"
    return "size 24+"


def md_table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    """GitHub-markdown table without the tabulate dependency."""
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "—" if pd.isna(v) else floatfmt.format(v)
        return str(v)

    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def group_metrics(y_true, y_pred, groups, class_names, group_order=None):
    """Per-group accuracy, per-class recall (TPR), per-class FPR,
    predicted-favorable rate, and disparate impact ratio."""
    rows = []
    keys = group_order if group_order is not None else sorted(groups.unique())
    for g in keys:
        m = groups == g
        yt, yp = y_true[m], y_pred[m]
        row = {"group": g, "n": int(m.sum()),
               "accuracy": accuracy_score(yt, yp)}
        for c in class_names:
            in_c = yt == c
            row[f"recall_{c}"] = (
                (yp[in_c] == c).mean() if in_c.sum() else np.nan)
            row[f"fpr_{c}"] = (
                (yp[~in_c] == c).mean() if (~in_c).sum() else np.nan)
        row["pred_fit_rate"] = (yp == FAVORABLE).mean()
        rows.append(row)
    out = pd.DataFrame(rows)
    ref = out["pred_fit_rate"].max()
    out["di_ratio"] = out["pred_fit_rate"] / ref
    out["di_flag"] = np.where(out["di_ratio"] < 0.8, "FLAG", "")
    return out


def eq_odds_spreads(gm: pd.DataFrame, class_names) -> pd.DataFrame:
    """Max-minus-min spread of TPR and FPR across groups, per class."""
    rows = []
    for c in class_names:
        rows.append({
            "class": c,
            "tpr_min": gm[f"recall_{c}"].min(),
            "tpr_max": gm[f"recall_{c}"].max(),
            "tpr_spread": gm[f"recall_{c}"].max() - gm[f"recall_{c}"].min(),
            "fpr_min": gm[f"fpr_{c}"].min(),
            "fpr_max": gm[f"fpr_{c}"].max(),
            "fpr_spread": gm[f"fpr_{c}"].max() - gm[f"fpr_{c}"].min(),
        })
    return pd.DataFrame(rows)


def overall_metrics(name, y_true, y_pred, class_names):
    per_f1 = f1_score(y_true, y_pred, average=None, labels=class_names)
    per_rec = {c: (y_pred[y_true == c] == c).mean() for c in class_names}
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        **{f"f1_{c}": f for c, f in zip(class_names, per_f1)},
        **{f"recall_{c}": per_rec[c] for c in class_names},
    }


def run_audit(tag, y_true, y_pred, test_df, class_names, frags):
    """Full group audit for one model; appends markdown fragments."""
    groupings = {
        "body_type": (test_df["body_type"], None),
        "height_band": (test_df["height_band"],
                        ["under 160 cm", "160-169 cm", "170 cm and over"]),
        "size_band": (test_df["size_band"],
                      ["size 0-6", "size 8-14", "size 16-22", "size 24+"]),
    }
    results = {}
    for gname, (groups, order) in groupings.items():
        gm = group_metrics(y_true, y_pred, groups, class_names, order)
        eo = eq_odds_spreads(gm, class_names)
        results[gname] = gm
        frags.append(f"\n### [{tag}] per-group metrics — {gname}\n")
        cols = (["group", "n", "accuracy"]
                + [f"recall_{c}" for c in class_names]
                + ["pred_fit_rate", "di_ratio", "di_flag"])
        frags.append(md_table(gm[cols]))
        frags.append(f"\n### [{tag}] equalized-odds FPR by group — {gname}\n")
        frags.append(md_table(
            gm[["group", "n"] + [f"fpr_{c}" for c in class_names]]))
        frags.append(f"\n### [{tag}] equalized-odds spreads — {gname}\n")
        frags.append(md_table(eo))
    return results


def shap_plots(pipeline, X_test, class_names):
    pre = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    Xs = X_test.sample(SHAP_SAMPLE, random_state=SEED)
    Xt = pre.transform(Xs)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    feat_names = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(Xt)  # (n, features, classes) or list per class
    if isinstance(sv, list):
        sv = np.stack(sv, axis=-1)

    # Global: mean |SHAP| per feature, stacked by class, top 15 features.
    mean_abs = np.abs(sv).mean(axis=0)  # (features, classes)
    order = np.argsort(mean_abs.sum(axis=1))[::-1][:15][::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    left = np.zeros(len(order))
    colors = ["#4c78a8", "#f58518", "#e45756"]
    for ci, cname in enumerate(class_names):
        vals = mean_abs[order, ci]
        ax.barh(range(len(order)), vals, left=left,
                color=colors[ci], label=cname)
        left += vals
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([feat_names[i] for i in order], fontsize=9)
    ax.set_xlabel("mean |SHAP value| (summed across classes)")
    ax.set_title("XGBoost — global feature importance (SHAP)")
    ax.legend(title="class")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "shap_global_bar.png", dpi=150)
    plt.close(fig)

    # Per-class beeswarms.
    for ci, cname in enumerate(class_names):
        plt.figure()
        shap.summary_plot(sv[:, :, ci], Xt, feature_names=feat_names,
                          max_display=12, show=False)
        plt.title(f"SHAP beeswarm — class '{cname}'", fontsize=11)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"shap_beeswarm_{cname}.png", dpi=150)
        plt.close("all")

    # Top features table for the report.
    top = pd.DataFrame({
        "feature": [feat_names[i] for i in order[::-1]],
        **{f"mean_abs_shap_{c}": mean_abs[order[::-1], ci]
           for ci, c in enumerate(class_names)},
    })
    return top


def recall_comparison_plot(before, after, class_names):
    x = np.arange(len(class_names))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(x - 0.18, [before[f"recall_{c}"] for c in class_names], 0.36,
           label="before (unweighted)", color="#4c78a8")
    ax.bar(x + 0.18, [after[f"recall_{c}"] for c in class_names], 0.36,
           label="after (class-weighted)", color="#f58518")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.set_ylabel("recall (test set)")
    ax.set_title("Per-class recall before vs after class-weighted retraining")
    ax.legend()
    for i, c in enumerate(class_names):
        ax.text(i - 0.18, before[f"recall_{c}"] + 0.01,
                f"{before[f'recall_{c}']:.2f}", ha="center", fontsize=8)
        ax.text(i + 0.18, after[f"recall_{c}"] + 0.01,
                f"{after[f'recall_{c}']:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "recall_before_after.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    label_encoder = joblib.load(MODELS_DIR / "label_encoder.joblib")
    y = label_encoder.transform(df[TARGET])
    class_names = list(label_encoder.classes_)

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=TEST_SIZE, stratify=y, random_state=SEED,
    )
    pinned = pd.read_csv(MODELS_DIR / "test_indices.csv")["row_index"]
    assert np.array_equal(np.asarray(idx_test), pinned.to_numpy()), (
        "Re-derived split does not match models/test_indices.csv")
    print(f"Split verified against pinned test_indices.csv "
          f"({len(idx_test)} test rows).")

    test_df = df.loc[idx_test].copy()
    test_df["height_band"] = test_df["height_cm"].map(height_band)
    test_df["size_band"] = test_df["size_ordered"].map(size_band)
    y_test_lbl = pd.Series(label_encoder.inverse_transform(y_test),
                           index=test_df.index)

    frags = ["# Fairness audit — computed tables (generated by "
             "src/fairness_audit.py, do not edit)\n"]

    # ---- BEFORE: Phase 5 unweighted XGBoost ----
    baseline = joblib.load(MODELS_DIR / "xgboost.joblib")
    pred_before = pd.Series(
        label_encoder.inverse_transform(baseline.predict(X_test)),
        index=test_df.index)
    before_overall = overall_metrics(
        "xgboost (unweighted, Phase 5)", y_test_lbl, pred_before, class_names)
    run_audit("BEFORE", y_test_lbl, pred_before, test_df, class_names, frags)

    # ---- SHAP on the winning (baseline) model ----
    print("Computing SHAP values...")
    top_shap = shap_plots(baseline, X_test, class_names)
    frags.append("\n### SHAP top features (mean |SHAP|, baseline model)\n")
    frags.append(md_table(top_shap))

    # ---- MITIGATION: identical pipeline, balanced sample weights ----
    print("Retraining with balanced class weights...")
    weighted = clone(baseline)
    sw = compute_sample_weight("balanced", y_train)
    weighted.fit(X_train, y_train, model__sample_weight=sw)
    joblib.dump(weighted, MODELS_DIR / "xgboost_weighted.joblib", compress=3)

    pred_after = pd.Series(
        label_encoder.inverse_transform(weighted.predict(X_test)),
        index=test_df.index)
    after_overall = overall_metrics(
        "xgboost (class-weighted)", y_test_lbl, pred_after, class_names)
    run_audit("AFTER", y_test_lbl, pred_after, test_df, class_names, frags)

    overall = pd.DataFrame([before_overall, after_overall])
    frags.insert(1, "\n### Overall before/after\n\n" + md_table(overall))
    recall_comparison_plot(before_overall, after_overall, class_names)

    (OUT_DIR / "audit_tables.md").write_text("\n".join(frags) + "\n")
    (MODELS_DIR / "fairness_audit_summary.json").write_text(json.dumps({
        "seed": SEED,
        "test_rows": int(len(idx_test)),
        "favorable_outcome": f"predicted '{FAVORABLE}'",
        "groupings": ["body_type", "height_band", "size_band"],
        "mitigation": "balanced sample weights, identical pipeline/split",
        "before": {k: round(float(v), 4) for k, v in before_overall.items()
                   if k != "model"},
        "after": {k: round(float(v), 4) for k, v in after_overall.items()
                  if k != "model"},
    }, indent=2))

    print("\n" + md_table(overall))
    print(f"\nWrote {OUT_DIR}/audit_tables.md, SHAP + recall plots, "
          f"models/xgboost_weighted.joblib, "
          f"models/fairness_audit_summary.json")


if __name__ == "__main__":
    main()

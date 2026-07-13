# FitML — A Fairness-Audited ML System for Sizing and Intelligent Virtual Try-On in E-Commerce

**Live demo:** [fit-ml.netlify.app](https://fit-ml.netlify.app) · **API:** [fit-ml.onrender.com](https://fit-ml.onrender.com) *(free tiers — first request after idle may take 30–60 s to cold-start)*

Capstone project for the AIM Postgraduate Diploma in Artificial Intelligence
and Machine Learning — "Pillar 5: Capstone Project" — by Debie Galleros,
July 2026.

Online clothing returns run ~35% (vs ~8% in physical stores), driven by
sizing uncertainty and by shoppers not knowing how a garment will look on
them. FitML is a **functional prototype** of a virtual fitting room that
addresses both:

- **Size recommendation** — a supervised classifier (fit / runs small /
  runs large) trained on 251,047 real customer fit reviews (ModCloth +
  RentTheRunway), served with a confidence score and a plain-language
  sizing tip for borderline cases.
- **Intelligent virtual try-on** — a generative renderer shows the actual
  catalog garment worn on the shopper's own photo (garment-conditioned
  diffusion via Replicate), with the original 2D pose-anchored compositor
  (MediaPipe keypoints + affine-warped cutout PNGs) as automatic fallback.
- **A published fairness audit** — disparate impact, equalized odds, and
  SHAP explainability across body-type / height / size groups, with a
  class-weighted mitigation and honest before/after numbers. This is the
  differentiator vs. commercial black-box sizing tools.
- **Claude-generated fit advice** — multimodal (the composited try-on
  image + measurements in, two paragraphs of plain-language advice out).
  Advice text only: no measurement estimation, no influence on the model.

## Results at a glance

| | accuracy | macro-F1 | recall `large` | recall `small` |
|---|---|---|---|---|
| XGBoost, unweighted (Phase 5 winner of 4 models) | 0.727 | 0.292 | 0.011 | 0.006 |
| XGBoost, class-weighted (mitigation, **served**) | 0.396 | 0.360 | 0.486 | 0.546 |

All four unweighted models (LogReg, RF, XGBoost, MLP) collapse toward the
73%-majority "fit" class — the class-weighted retrain trades headline
accuracy for actually catching misfits, which is what prevents returns.
Full reasoning in [docs/model_selection.md](docs/model_selection.md) and
[docs/fairness_report.md](docs/fairness_report.md).

## Documentation

| Doc | Contents |
|---|---|
| [docs/data_dictionary.md](docs/data_dictionary.md) | Every column of the model-ready dataset and the men's synthetic extension; missing-value strategy |
| [docs/model_selection.md](docs/model_selection.md) | 4-model comparison, why macro-F1, why the class-weighted model is served |
| [docs/fairness_report.md](docs/fairness_report.md) | Full bias audit: groups, disparate impact, equalized odds, SHAP, mitigation before/after |
| [docs/scope_and_data_provenance.md](docs/scope_and_data_provenance.md) | Real vs synthetic data boundaries, catalog image sourcing decisions |
| [docs/privacy.md](docs/privacy.md) | Photo-upload protections (24 h TTL, opt-in crop-at-upload face privacy, UUID sessions), RA 10173 framing |
| [docs/genai_usage.md](docs/genai_usage.md) | Where generative AI is used (advice text, background removal) and why it is fenced off from the graded ML |
| [docs/deployment.md](docs/deployment.md) | Render + Netlify setup, Docker rationale, redeploy steps |
| [docs/planning/FitML_Master_Plan.md](docs/planning/FitML_Master_Plan.md) | Authoritative project plan (scope, data sources, locked decisions) |

## Repo structure

```
pillar5-capstone/
├── src/           # data prep, training, fairness audit, catalog pipeline
├── notebooks/     # 01 data inspection · 02 EDA + feature engineering
├── data/          # raw/ processed/ catalog/   (raw + image assets gitignored)
├── models/        # saved pipelines, comparison.csv, audit summary, test indices
├── backend/       # Flask app (upload, recommend-size, try-on, advice)
├── frontend/      # plain HTML/CSS/JS — landing, about, profile, catalog, item, history
├── docs/          # reports & documentation (see table above)
├── scripts/       # deploy-time catalog asset fetch
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/debiegalleros/fitml-capstone.git
cd fitml-capstone
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Data (needed for training/audit reproduction)

Kaggle API credentials go in `~/.kaggle/kaggle.json` (never committed —
see [Kaggle API docs](https://www.kaggle.com/docs/api)), then:

```bash
kaggle datasets download rmisra/clothing-fit-dataset-for-size-recommendation -p data/raw
python src/prepare_data.py        # -> data/processed/model_ready.csv
python src/mens_extension.py      # -> data/processed/mens_synthetic.csv
```

### Reproduce the graded pipeline (fixed seed 42 throughout)

```bash
python src/train_models.py        # 4 models -> models/*.joblib + comparison.csv
python src/fairness_audit.py      # audit tables/plots + xgboost_weighted.joblib
```

### Run the app locally

Catalog images are gitignored; fetch them once from the GitHub release
asset:

```bash
./scripts/fetch_catalog_assets.sh
```

Then in two terminals:

```bash
python backend/app.py             # Flask API on http://localhost:5001
python frontend/devserver.py      # frontend on http://localhost:8000
```

The advice endpoint needs `ANTHROPIC_API_KEY` in `backend/.env` (every
other feature works without it).

## Privacy

Uploaded photos are stored under anonymous UUID session folders and
auto-deleted after 24 hours. An opt-in checkbox at upload (unchecked by
default) crops the photo above the nose before it's ever written to disk,
so no face pixels reach storage; left unchecked, generated try-on images
still get the shopper's real face re-composited back on, protecting
against the generative engine drawing an incorrect one. Details and RA
10173 framing in [docs/privacy.md](docs/privacy.md).

## Data & asset attribution

- [Clothing Fit Dataset for Size Recommendation](https://www.kaggle.com/datasets/rmisra/clothing-fit-dataset-for-size-recommendation) (ModCloth + RentTheRunway) — model training and fairness audit
- [Fashion Product Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset) (MIT license) — demo catalog imagery
- Uniqlo published men's size charts — label source for the small,
  clearly-flagged synthetic men's extension (excluded from the fairness
  audit; see [docs/scope_and_data_provenance.md](docs/scope_and_data_provenance.md))

Catalog prices are programmatically generated demo pricing, not real
merchant data. FitML is a functional prototype — no checkout, payments,
or inventory.

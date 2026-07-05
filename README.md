# FitML — Capstone Project

A fairness-audited size-recommendation + 2D AR try-on system for e-commerce.

Capstone project for AIM PGDAIML, "Pillar 5: Capstone Project" (due 2026-07-17).

## Project plan

See [`docs/planning/FitML_Master_Plan.md`](docs/planning/FitML_Master_Plan.md) — authoritative source of truth for scope, data sources, models, and build phases.

## Repo structure

```
pillar5-capstone/
├── src/           # scripts (data prep, training, fairness audit, catalog processing)
├── notebooks/     # EDA notebook(s)
├── data/          # raw/ processed/ catalog/  (raw + catalog gitignored)
├── models/        # saved artifacts + comparison.csv
├── backend/       # Flask app
├── frontend/      # Fable app
├── docs/          # data_dictionary.md, model_selection.md, fairness_report.md, privacy.md, planning/
└── requirements.txt
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Kaggle API credentials go in `~/.kaggle/` (never committed) — see [Kaggle API docs](https://www.kaggle.com/docs/api).

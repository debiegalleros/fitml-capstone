# FitML — Full Build Guide (Mac Terminal)

Deadline: July 17, 2026. This guide covers Phase 0 (setup) through Phase 9 (deployment).
Work through it top to bottom — each phase produces files the next phase needs.

---

## Phase 0 — Environment Setup (Day 1, ~1 hr)

Open Terminal.

```bash
# Check Python (Mac usually ships 3.9+; confirm)
python3 --version

# Install Homebrew if you don't have it
which brew || /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install git, if needed
brew install git

# Create project folder + structure
mkdir -p ~/pillar5-capstone/{src,notebooks,data/{raw,processed,catalog},models,frontend,backend,docs}
cd ~/pillar5-capstone

# Set up Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install core Python packages
pip install --upgrade pip
pip install pandas numpy scikit-learn xgboost matplotlib seaborn \
    opencv-python-headless pillow mediapipe rembg flask flask-cors \
    kaggle shap jupyter

# Initialize git repo
git init
echo "venv/
data/raw/
data/catalog/
__pycache__/
*.pyc
.env" > .gitignore
git add .gitignore
git commit -m "Initial project structure"
```

Create a Kaggle account (free) at kaggle.com if you don't have one, then:

```bash
# Get your API token from kaggle.com/settings -> Create New Token
# This downloads kaggle.json -> move it here:
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

---

## Phase 1 — Data Collection (Days 1–2)

### 1a. Garment catalog images (iMaterialist)

```bash
cd ~/pillar5-capstone/data/raw
kaggle competitions download -c imaterialist-fashion-2020-fgvc7 -p ./imaterialist
cd imaterialist && unzip -q *.zip
```

If that specific competition dataset requires a joined competition, search Kaggle for
"iMaterialist Fashion" and use whichever public version is downloadable — there are
several years' versions with similar category/segmentation structure.

### 1b. Body measurement data (ANSUR II)

```bash
cd ~/pillar5-capstone/data/raw
mkdir ansur && cd ansur
# ANSUR II is hosted by the US Army; search "ANSUR II public data" —
# it's distributed as an Excel file (male + female measurements).
# Download manually via browser into this folder, then:
open .   # confirms the file landed here
```

### 1c. Public size charts (manual compilation)

Create `~/pillar5-capstone/data/raw/size_charts.csv` by hand, pulling published ranges from
Zara, H&M, Uniqlo, ASOS size-guide pages (open each in browser, copy values):

```
category,brand,size,chest_min_cm,chest_max_cm,waist_min_cm,waist_max_cm,hip_min_cm,hip_max_cm
tshirt,Zara,S,86,90,70,74,88,92
tshirt,Zara,M,91,95,75,79,93,97
...
```

Aim for ~5–10 rows per category × size combo across 2–3 brands. This is the ground
truth your synthetic customers get labeled against.

---

## Phase 2 — Synthetic Customer Dataset (Day 2)

Create `src/generate_customers.py`:

```python
import pandas as pd
import numpy as np

np.random.seed(42)
N = 5000

# Sample from realistic population distributions (cm)
df = pd.DataFrame({
    "height_cm": np.random.normal(165, 10, N).clip(140, 200),
    "chest_cm": np.random.normal(92, 12, N).clip(70, 140),
    "waist_cm": np.random.normal(78, 13, N).clip(60, 130),
    "hip_cm": np.random.normal(97, 12, N).clip(75, 140),
    "shoulder_width_cm": np.random.normal(40, 4, N).clip(32, 52),
})

# Inject some missingness and outliers for realistic preprocessing work
mask = np.random.rand(N) < 0.03
df.loc[mask, "shoulder_width_cm"] = np.nan
outlier_idx = np.random.choice(N, 15, replace=False)
df.loc[outlier_idx, "chest_cm"] *= 1.8  # extreme outliers to catch in EDA

df["customer_id"] = range(1, N + 1)
df.to_csv("data/processed/synthetic_customers.csv", index=False)
print(df.describe())
```

Create `src/label_sizes.py` to join against your size chart and assign ground-truth
size labels (nearest-fit logic against `chest_min/max` etc. per category), writing
`data/processed/labeled_dataset.csv`. This is your Step 2 deliverable dataset.

```bash
cd ~/pillar5-capstone
python src/generate_customers.py
python src/label_sizes.py
```

Write the data dictionary by hand into `docs/data_dictionary.md` — one row per column,
type, unit, allowed range, source (synthetic vs. size-chart-derived).

---

## Phase 3 — EDA, Preprocessing & Feature Engineering (Days 2–3)

Open a notebook:

```bash
jupyter notebook notebooks/
```

In `notebooks/01_eda_feature_engineering.ipynb`, cover:
- Nulls/outliers: `df.isna().sum()`, boxplots per feature, cap/impute
- Distributions: histograms per size class (seaborn `histplot`)
- Correlation heatmap (`sns.heatmap(df.corr())`)
- Feature engineering: `StandardScaler` on numeric cols, `OneHotEncoder` on
  category/fabric/brand
- Feature selection: `SelectKBest` or `RandomForestClassifier.feature_importances_`
- PCA: `sklearn.decomposition.PCA(n_components=2)`, scatter plot colored by size label

Save the cleaned, encoded dataset to `data/processed/model_ready.csv` at the end.

---

## Phase 4 — Model Training & Comparison (Days 3–4)

Create `src/train_models.py`:

```python
import pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

df = pd.read_csv("data/processed/model_ready.csv")
X = df.drop(columns=["size_label", "customer_id"])
y = df["size_label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

models = {
    "logreg": LogisticRegression(max_iter=1000),
    "rf": RandomForestClassifier(n_estimators=200, random_state=42),
    "xgb": XGBClassifier(eval_metric="mlogloss", random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")
    results[name] = {"accuracy": acc, "f1_macro": f1}
    print(name, results[name])
    print(classification_report(y_test, preds))
    joblib.dump(model, f"models/{name}.joblib")

pd.DataFrame(results).T.to_csv("models/comparison.csv")
```

```bash
python src/train_models.py
```

Pick the best model by F1-macro (not just accuracy — macro F1 catches minority-size
class failures). Note your reasoning in `docs/model_selection.md`.

---

## Phase 5 — Fairness Audit (Days 4–5)

Create `src/fairness_audit.py`:

```python
import pandas as pd, joblib
from sklearn.metrics import accuracy_score

model = joblib.load("models/xgb.joblib")  # or whichever you picked
df = pd.read_csv("data/processed/model_ready.csv")

# Stratify by a body-type bucket you derive (e.g. from height/chest ratios)
df["body_group"] = pd.cut(df["height_cm"], bins=[0,155,175,300], labels=["petite","average","tall"])

for group, sub in df.groupby("body_group"):
    X_sub = sub.drop(columns=["size_label", "customer_id", "body_group"])
    y_sub = sub["size_label"]
    preds = model.predict(X_sub)
    acc = accuracy_score(y_sub, preds)
    print(f"{group}: n={len(sub)}, accuracy={acc:.3f}")
```

From this output, compute:
- **Disparate impact ratio** = (min group accuracy) / (max group accuracy) — flag if < 0.8
- **Equalized odds** — compare true positive rates per group per size class
- Try a **mitigation**: retrain with `class_weight="balanced"` or per-group thresholds,
  rerun the audit, and show the gap narrow before/after.

Write findings to `docs/fairness_report.md` with the before/after numbers and SHAP
plots (`shap.TreeExplainer(model)`) for explainability.

---

## Phase 6 — Backend: Flask API (Days 5–8)

```bash
cd ~/pillar5-capstone/backend
```

Create `app.py` with these endpoints:
- `POST /upload-profile` — accepts photos, runs MediaPipe pose extraction + saves
- `GET /catalog` — returns garment metadata (filtered by query params)
- `POST /recommend-size` — loads user profile + item size chart → calls your trained
  model → returns size + confidence
- `POST /try-on` — runs the body-wrap compositing (garment PNG + pose keypoints →
  warped, alpha-blended result image)
- `POST /advice` — calls Claude Vision API for personalized advice text

Minimal skeleton:

```python
from flask import Flask, request, jsonify
from flask_cors import CORS
import mediapipe as mp
import cv2, joblib

app = Flask(__name__)
CORS(app)
model = joblib.load("../models/xgb.joblib")

@app.route("/upload-profile", methods=["POST"])
def upload_profile():
    file = request.files["photo"]
    file.save(f"uploads/{file.filename}")
    # run mediapipe pose extraction here, return keypoints + measurements
    return jsonify({"status": "ok"})

@app.route("/catalog", methods=["GET"])
def catalog():
    # read data/catalog/metadata.csv, filter by request.args, return JSON
    return jsonify([])

if __name__ == "__main__":
    app.run(debug=True, port=5000)
```

Test locally:

```bash
python app.py
# in another terminal tab:
curl http://127.0.0.1:5000/catalog
```

### Garment processing (do this once, populates your catalog)

```bash
cd ~/pillar5-capstone
python src/process_catalog.py   # rembg background removal + resize + hue-variant generation
# outputs 100+ PNGs into data/catalog/ + metadata.csv
```

---

## Phase 7 — Frontend: Fable UI (Days 7–10)

```bash
# Install .NET SDK (Fable needs it)
brew install dotnet-sdk
dotnet tool install fable --global

cd ~/pillar5-capstone/frontend
dotnet new console -lang F# -o .
dotnet add package Fable.React
dotnet add package Fable.Browser.Dom
npm init -y
npm install vite
```

Build the four screens from the mockup we discussed (profile setup → catalog/filter →
try-on result → history), each calling your Flask endpoints via `fetch`.

```bash
# Run dev server
dotnet fable watch --run npx vite
```

---

## Phase 8 — Deployment (Days 12–13)

### Backend → Render (free tier)

```bash
cd ~/pillar5-capstone/backend
pip freeze > requirements.txt
git add . && git commit -m "backend ready for deploy"
```

Push to GitHub, then on render.com: New → Web Service → connect repo → set start
command `python app.py` → deploy. Free tier is enough for a capstone demo.

### Frontend → Netlify or Vercel (free tier)

```bash
cd ~/pillar5-capstone/frontend
npx vite build
```

Drag the resulting `dist/` folder into netlify.com's deploy UI, or:

```bash
npm install -g netlify-cli
netlify deploy --prod --dir=dist
```

Update the frontend's API base URL to point to your Render backend URL, rebuild,
redeploy.

---

## Phase 9 — Docs, GitHub, Presentations (Days 13–14)

```bash
cd ~/pillar5-capstone
mkdir -p src notebooks data models docs
```

Confirm structure matches:
```
pillar5-capstone/
├── src/
├── notebooks/
├── data/
├── models/
├── backend/
├── frontend/
├── docs/
│   ├── data_dictionary.md
│   ├── model_selection.md
│   ├── fairness_report.md
├── README.md
└── requirements.txt
```

Write `README.md` (setup + run instructions), push everything:

```bash
git remote add origin https://github.com/<your-username>/fitml.git
git add .
git commit -m "FitML capstone complete"
git push -u origin main
```

Build your two decks (technical + business) from `docs/fairness_report.md` and
`models/comparison.csv` — the numbers are already sitting there, ready to drop into
slides.

---

## Quick checklist by rubric step

- [ ] Step 1 — problem framing written up in `docs/`
- [ ] Step 2 — `data_dictionary.md` + sourced datasets
- [ ] Step 3 — `01_eda_feature_engineering.ipynb`
- [ ] Step 4 — `train_models.py` output + `comparison.csv`
- [ ] Step 5 — `fairness_audit.py` output + `fairness_report.md`
- [ ] Step 6 — two decks
- [ ] Step 7 — GitHub repo pushed
- [ ] Step 8 — Flask deployed live (bonus)
- [ ] Step 9 — Claude Vision usage documented (bonus)

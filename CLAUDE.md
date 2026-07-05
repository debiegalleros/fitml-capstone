# FitML Capstone — Project Instructions for Claude Code

## What this project is
FitML: a fairness-audited size-recommendation + 2D AR try-on system for e-commerce.
Capstone for AIM PGDAIML, LMS assignment "Pillar 5: Capstone Project", **due July 17, 2026**.
Folder/repo on disk: `~/pillar5-capstone`. GitHub repo name: `fitml-capstone`.
Report title: "FitML: A Fairness-Audited Machine Learning System for Sizing and AR Try-On in E-Commerce".

## Source-of-truth rule (IMPORTANT)
Two planning docs may exist in `docs/planning/`:
- `FitML_Master_Plan.md` — **AUTHORITATIVE. Always follow this.**
- `FitML_Build_Guide.md` — older draft, kept only for code snippets. Where it conflicts
  with the Master Plan, the Master Plan wins. Known stale items in the Build Guide —
  DO NOT follow these:
  - ANSUR II dataset → superseded. Primary data is Kaggle "Clothing Fit Dataset for
    Size Recommendation" (ModCloth + RentTheRunway, real women's data).
  - Fully synthetic 5,000-customer dataset as the graded core → superseded. Synthetic
    data is ONLY the small men's extension (~10–15 items, Uniqlo size charts,
    rule-based, EXCLUDED from the fairness audit).
  - "Four screens" → superseded. Six pages: Landing, About/How-it-works, Profile
    setup, Catalog, Try-on result, History.
  - Multi-brand size chart compilation (Zara/H&M/ASOS) → superseded. Uniqlo men's
    charts only, for the extension.
  - "Claude Vision estimates measurements" → wrong. Claude generates ONLY advice text.
    Measurements are always manually entered by the user.
  - Repo name `fitml` → use `fitml-capstone`.

## Locked decisions — do not revisit or "improve"
- Models: Logistic Regression, Random Forest, XGBoost, MLP. **No CNNs, no diffusion
  models** — explicitly scoped out. If tree models beat the MLP, report that honestly.
- Try-on: 2D body-wrap compositing (MediaPipe pose keypoints + affine-warped
  transparent garment PNG, alpha-composited on HTML5 Canvas). **NOT 3D, NOT Three.js,
  NOT WebGL.** Do not suggest 3D.
- Task type: supervised classification (fit/size prediction) on the real women's
  dataset. This is the entire graded ML core.
- Fairness audit (Step 5, 20 pts): real dataset/model only — disparate impact,
  equalized odds, SHAP, at least one mitigation with before/after numbers. Men's
  synthetic extension is EXCLUDED (documented reasoning: sample too small for valid
  group comparisons).
- Measurements: manual entry — height (cm), weight (kg, optional — RentTheRunway
  only), bust via toggle (band+cup dropdowns OR chest cm converted to nearest
  band+cup, flagged lower-precision, input method tracked per user), waist (cm),
  hip (cm), body type dropdown (hourglass/pear/apple/rectangle/athletic/petite).
  Uploaded photo is for try-on pose ONLY — never feeds the size model.
- Claude API role: advice text only (personalized fit explanation, history-based
  brand/fabric notes). Never measurement estimation, never part of graded ML.
- History suggestions: rule-based DB lookup (2+ prior items same brand/fabric with
  consistent fit → add note to Claude advice prompt). NOT a second trained model.
- Catalog: iMaterialist images (join competition on Kaggle first), rembg background
  removal, ~100+ women's + ~10–15 men's items, 2–3 hue-shifted variants each.
  Metadata per item: item_id, category, gender, color, plausible fabric, size_range,
  price via random.randint within category bands, rounded to nearest 10 PHP:
  tshirt/tank/shorts 400–700 · polo/blouse 600–1000 · jeans/slacks/sweater 800–1400 ·
  jacket/dress 1200–1800. Document as programmatically generated demo pricing.
- Color-suitability matching: OUT OF SCOPE. Mention as future work only.
- Product: functional prototype website — no checkout, no payments, no inventory.
  Describe as "functional prototype", never "an online store".
- Stack: Flask backend, Fable (F#→JS) frontend, plain Canvas compositing.
  Deploy: Render/Railway (backend) + Netlify/Vercel (frontend), free tiers.
- UI: flat cards, hairline borders, minimal — follow the screen-by-screen spec in
  the Master Plan (sidebar filters, 2-col grid, confidence box with blue/amber
  states, size pills, color swatches). Do not redesign.
- Confidence box borderline rule: predicted size near boundary AND low-stretch
  fabric AND fitted/athletic cut → amber box, recommend one size up, lower displayed
  confidence, tradeoff-explaining advice text.
- Privacy: uploads under `backend/uploads/{uuid}/`, 24h auto-delete, face blur ON by
  default (MediaPipe + Gaussian), HTTPS only, cite RA 10173 in `docs/privacy.md`.

## Rubric (100 pts) — effort should track points
Problem Framing 10 · Data Collection 10 · EDA/FE 10 · **Models 20** ·
**Ethical AI & Bias Audit 20** · Presentations 10 · **GitHub 15** · Bonus 5.
Steps 8 (Deployment) and 9 (GenAI documentation) are optional, no dedicated points.
Model Implementation + Bias Audit + GitHub = 55 pts — prioritize these over UI polish.

## Repo structure (rubric requirement)
```
pillar5-capstone/
├── src/           # scripts (data prep, training, fairness audit, catalog processing)
├── notebooks/     # EDA notebook(s)
├── data/          # raw/ processed/ catalog/  (raw + catalog gitignored)
├── models/        # saved artifacts + comparison.csv
├── backend/       # Flask app
├── frontend/      # Fable app
├── docs/          # data_dictionary.md, model_selection.md, fairness_report.md,
│                  # privacy.md, scope_and_data_provenance.md, planning/
├── README.md
└── requirements.txt
```
Keep commit history clean and incremental (one commit per meaningful unit of work,
descriptive messages) — commit history is explicitly graded.

## Submission file naming (exact)
- Report → `Debie_Galleros_Pillar5_CapstoneProject_Report.pdf`
- Technical deck → `Debie_Galleros_Pillar5_CapstoneProject_TechnicalDeck.pptx`
- Business deck → `Debie_Galleros_Pillar5_CapstoneProject_BusinessDeck.pptx`
- Code zip (if needed) → `Debie_Galleros_Pillar5_CapstoneProject_Code.zip`
- Single-file fallback → `Debie_Galleros_Pillar5_CapstoneProject.pdf` (sectioned:
  Report → Technical Deck → Business Deck)
Cover page title uses the FitML title; filenames use the LMS assignment name.

## Phase plan (work one phase at a time; stop and confirm before moving on)
0. Environment: venv, folder structure, git init, Kaggle CLI + token
1. Download both Kaggle datasets, verify files landed
2. Load + inspect real fit dataset (rows, columns, nulls, dtypes) — no cleaning yet
3. Cleaning + EDA + feature engineering (nulls/outliers, scaling, encoding, PCA,
   feature selection) → notebook + `model_ready` dataset
4. Men's synthetic extension (Uniqlo charts → rule-based lookup, clearly flagged)
5. Train + compare LogReg/RF/XGBoost/MLP, save artifacts + comparison.csv
6. Fairness audit (real model only) + mitigation before/after → fairness_report.md
7. Catalog processing (rembg, resize, hue variants) + metadata.csv with pricing
8. Flask backend (upload, catalog, recommend-size, try-on, advice endpoints)
9. Fable frontend (six pages per spec)
10. Deployment (Render + Netlify/Vercel)
11. Docs, README, decks, GitHub push, rename submission files per convention

## Working style
- Do not jump ahead of the current phase. Finish, verify (run the code / check
  outputs), summarize what was produced, then wait for go-ahead.
- Everything reproducible: fixed random seeds, saved configs, requirements.txt.
- Never commit raw datasets, kaggle.json, API keys, or uploaded photos.
- Python venv at `./venv` — activate before running anything.

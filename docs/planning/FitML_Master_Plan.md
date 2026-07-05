# FitML — Master Project Plan
A Fairness-Audited Machine Learning System for Sizing and AR Try-On in
E-Commerce
Capstone Project — AI/ML Postgraduate Diploma | Due July 17, 2026

---

## 1. Problem Framing

**Business problem:** Online clothing returns run ~35% (vs. ~8% in physical stores),
costing retailers $100B+ annually. Sizing uncertainty is the leading driver, but
shoppers also hesitate due to uncertainty about fit and appearance — not knowing
how a garment will actually look on their body. FitML addresses both: a
fairness-audited sizing model for the "will it fit" question, and an AR try-on
feature so shoppers can see the garment on their own photo before buying —
addressing "how will this actually look on me." Color-suitability matching (does
this shade complement the shopper) is identified as a natural extension, out of
scope for this submission.

**Data science problem:** Multi-class classification — predict the correct garment
size for a customer given their body measurements and the garment's size chart —
paired with a fairness audit layer to ensure predictions are equally accurate across
body types.

**Task type:** Supervised classification (size label), with a rule-derived label
source (see Data, below) rather than collaborative-filtering recommendation.

**Success metrics:**
- Technical: Accuracy, F1-macro (catches minority-class/body-type failures that plain
  accuracy hides), disparate impact ratio, equalized odds
- Business KPI: Projected return-rate reduction (35% → 15–20% target)

---

## 2. Data Sources

| Need | Source | Notes |
|---|---|---|
| **Model training + full fairness audit** (graded core, women's apparel) | **Kaggle: "Clothing Fit Dataset for Size Recommendation"** — combines ModCloth + RentTheRunway data | Real customer bust/waist/hip/height/weight/body-type measurements, item ordered, size ordered, real fit feedback (small/fit/large) |
| **Men's catalog extension** (small, ~10-15 items, NOT in the fairness audit) | Synthetic customers, size-labeled via **Uniqlo's published men's size charts** | Real chart as label source, synthetic customers; sample size too small for valid group comparisons, so excluded from Step 5 |
| Garment catalog images (visual only, not training data) | **Fashion Product Images Dataset (Kaggle, `paramaggarwal/fashion-product-images-dataset`, full-res, MIT license)** | Plain-background product/model shots, 1080x1440, `styles.csv` metadata (gender/category/color). Background-removed via `rembg`, ~100+ women's + ~10-15 men's items. Supersedes iMaterialist Fashion (evaluated and rejected: in-the-wild street-style/red-carpet/runway photos, some watermarked, unsuitable for compositing) |
| Fairness test groups | Stratified subsets of the real Kaggle dataset itself | Not a separate synthetic persona set — stratify by body_type/other fields already present in the real data |

**Why this combination:** the Kaggle fit dataset gives real, citable, high-quality
data (exactly what Step 2's top rating asks for) but is women's-only, since both
ModCloth and RentTheRunway sell women's apparel. Rather than force a single
dataset to cover both genders, the plan uses real audited data where it's
available (women's) and clearly flags a small, unaudited synthetic extension
for the gap (men's) — an explicit, documented limitation rather than a silently
patched one. Uniqlo's charts are used only as the men's-extension label source,
chosen for their Asian-market sizing relevance given the CDO/Philippines context.

**Compilation approach (men's extension only):** manually transcribe a small
slice of Uniqlo's published men's size guide (t-shirt, polo, jeans, jacket,
shorts — 2-3 items each) into `data/raw/mens_size_charts.csv`:

```
category,brand,size,chest_min_cm,chest_max_cm,waist_min_cm,waist_max_cm
tshirt,Uniqlo,S,86,90,68,72
tshirt,Uniqlo,M,91,96,73,78
...
```

Then generate a small synthetic men's customer set, labeled by which size range
their measurements fall into — real sizing standard, synthetic population, rule-
based lookup, not a trained model, not included in the fairness audit.

---

## 3. Data Dictionary (summary — full version lives in `docs/data_dictionary.md`)

| Column | Type | Unit | Source |
|---|---|---|---|
| height_cm | float | cm | real (Kaggle fit dataset) |
| weight_kg | float, optional | kg | real — RentTheRunway rows only, missing for ModCloth |
| bust_band / bust_cup | int / categorical | band size + cup letter | real — user's native input format matches this |
| waist_cm | float | cm | real |
| hip_cm | float | cm | real |
| body_type | categorical | — | real (hourglass/pear/apple/rectangle/athletic/petite) |
| category | categorical | — | catalog metadata |
| fabric | categorical | — | catalog metadata (assigned, not ground-truth) |
| brand | categorical | — | real (ModCloth or RentTheRunway item brand) |
| fit_feedback | categorical (target) | — | real: small / fit / large — the actual label used for classification |

Men's extension uses a separate, smaller schema: height_cm, chest_cm, waist_cm,
hip_cm, category, size_label (derived from Uniqlo men's size chart) — kept in its
own table, not merged with the real women's data.

---

## 4. Preprocessing, EDA & Feature Engineering

- Clean nulls/outliers — real ones this time (e.g. weight missing for all ModCloth
  rows, self-reported measurement outliers), not artificially injected
- Scale numeric features, encode categoricals (fabric, brand, fit-type, body_type)
- EDA: distributions per fit_feedback class, correlation heatmap, class imbalance
  check (fit_feedback is likely skewed toward "fit" — relevant for metric choice)
- Feature selection: tree-based importances
- Dimensionality reduction: PCA (2D visualization of body-type clustering)

---

## 5. Model Implementation

Train and compare: **Logistic Regression** (baseline), **Random Forest**, **XGBoost**,
and a simple **Multi-Layer Perceptron (MLP)** neural network — all on the same real
labeled dataset (Kaggle ModCloth/RentTheRunway), same train/test split, same
metrics. Evaluate with accuracy + F1-macro, save models/configs for reproducibility,
select best by F1-macro (not raw accuracy, to protect against minority-class
blind spots given the likely "fit"-heavy class imbalance).
Expect tree-based models to likely outperform the MLP on this small tabular dataset —
that's a legitimate, reportable finding, not a failure; explain why in the writeup.

**Not included (documented as future work, not built):** a CNN-based fit-assessment
model (classifying garment fit quality from the composited try-on image) was
considered and scoped out. No labeled dataset exists for this task, and building one
would require generating a synthetic labeled set from the compositing pipeline itself
— feasible in principle but too time/risk-heavy given the deadline. Mentioned in the
business deck as a natural next step, not attempted in this submission.

---

## 6. Fairness & Ethical AI Audit

- Stratify test set by body-type bucket (petite/average/tall, and others as designed)
- Metrics: per-group accuracy, disparate impact ratio (<0.8 flags bias), equalized odds
- Explainability: SHAP on the selected model
- Mitigation trial: class-weighted retraining, before/after comparison
- Documented in `docs/fairness_report.md`

---

## 7. Product: Virtual Fitting Room + Try-On

**Rendering approach — CONFIRMED, do not revisit:** 2D body-wrap compositing
("static-image AR"), not 3D/Three.js. Full 3D (rigid or draped) was
considered and ruled out: realistic 3D requires either cloth-simulation
physics or a diffusion-model approach, both out of scope for a solo build
in the remaining timeline before July 17. 2D compositing still qualifies
as genuine AR — the original uploaded photo (background, pose, face)
remains fully visible; only the garment layer is added, scaled and rotated
using real MediaPipe pose keypoints so it tracks the user's actual body
proportions. It won't simulate fabric folds or shadows, but it correctly
and convincingly positions the garment on the person's real photo, which
is sufficient for the demo — none of this AR fidelity is part of the
graded rubric (Steps 4 and 5, worth 40/100 points, are the model and
fairness audit, not the visual).

MediaPipe extracts pose keypoints from the user's uploaded photo; a
background-removed garment PNG is scaled/rotated (affine transform) to
match shoulder width and torso angle, then alpha-composited onto the
photo. Size/color swaps re-run the same warp with a different PNG —
instant, no 3D asset pipeline needed.

**Catalog:** ~100+ garments across 13 categories (from the Fashion Product Images
Dataset, `paramaggarwal/fashion-product-images-dataset` — full-res, MIT license,
plain-background product/model shots with `styles.csv` gender/category/color
metadata), each generating 2–3 color variants via programmatic hue-shifting
rather than sourcing 300 unique photos. iMaterialist Fashion was evaluated
first and rejected: its images are in-the-wild street-style/red-carpet/runway
photos (busy backgrounds, some with visible third-party photographer/
publication watermarks) rather than clean product shots, making them a poor
fit for `rembg` + affine-warp compositing. Pricing is category-bounded, not
flat-random across the whole catalog
(tshirt/tank/shorts PHP 400-700, polo/blouse PHP 600-1000, jeans/slacks/sweater
PHP 800-1400, jacket/dress PHP 1200-1800), generated alongside color variants and
documented in the report as a demo-catalog convenience, not real merchant pricing.

**Scope note:** this is a functional prototype of an e-commerce sizing/try-on
experience, not a real online store — no checkout, no payment processing, no real
inventory/stock system. Catalog, pricing, profiles, model recommendations, and the
AR try-on are all real and functional; describe it as a "functional prototype" in
the report, not as "an online store."

---

## 8. UI — Web App (not native app)

Free, no app-store cost/review, and matches the Flask + Fable stack already planned.

**Six pages total:**
1. **Title/Landing** — FitML name, tagline ("Shop smarter. Fit better."), hero
   visual, "Get Started" button. Below the hero, a feature callout section
   styled as short labeled cards (icon + bold title + one-line description),
   e.g.:
     - **Virtual Fitting Room** — "Interactive clothing try-on for e-commerce"
     - **Fair Sizing** — "Size recommendations audited for fairness across body types"
     - (optionally a third card for the price/catalog browsing feature)
   This mirrors the concise glossary-card style used by industry tools
   (e.g. Vue.ai's "Virtual Fitting Rooms" card) — short label + one plain
   descriptive line, no long paragraphs.
2. **About/How it works** — the problem (35% return rate), how sizing + AR try-on
   work, the fairness differentiator (real trained model + published fairness
   audit vs. commercial black-box tools) — doubles as an in-product business pitch
3. **Profile setup** — photo upload (for try-on pose only), manual measurement
   entry (height, weight, bust, waist, hip, body type)
4. **Catalog/browse** — filter sidebar (category, size, gender, fabric, color, price)
   + garment grid with "try on" buttons
5. **Try-on result** — composited image, size/color swap pills, personalized advice
   text (confidence box: blue for confident match, amber for borderline/fabric-
   flagged cases)
6. **History** — past try-ons, fit feedback, and now also feeds lightweight
   personalization (see below)

**History-based suggestions (lightweight, rule-based — not a second trained
model):** each try-on outcome (size shown, confirmed measurements, brand,
fabric) is stored. When a user views a new item from a brand/fabric they've
tried before, check history: if 2+ prior items from that brand/fabric show
consistent fit feedback, surface that in the advice text (e.g. "You've tried
2 items from this brand — they've run true to size for you"). This is a
lookup + conditional prompt to Claude, not a new model — reuses data already
being stored, no added ML risk.

Design language: flat cards, hairline borders, no heavy styling — fast to build,
consistent across screens.

---

## 9. Privacy & Data Protection (body photo uploads)

**Storage:** `backend/uploads/{session_id}/` — random UUID folder names, never
name/email-linked. Excluded from git via `.gitignore`.

**Protections:**
- HTTPS-only transmission; disk-level encryption at rest (Render default / Mac
  FileVault for local dev)
- Auto-deletion after 24 hours via scheduled cleanup script
- **Face blur by default** (flipped from "show by default" to privacy-first) —
  MediaPipe detects face region, Gaussian-blurs it before saving, so an unblurred
  version never persists unless the user explicitly opts in to show it
- Session-scoped access — one session's token can't retrieve another's photos
- Consent notice on the upload screen: what's collected, why, retention period

**Regulatory framing:** Philippines' Data Privacy Act of 2012 (RA 10173) principles —
proportionality, transparency, legitimate purpose, retention limits — cited explicitly
in `docs/privacy.md` as part of the ethical-AI writeup. Fairness-audit test photos are
kept separate from any real user uploads; consent for one purpose isn't reused for the
other.

---

## 10. Tech Stack

- **Backend:** Flask (Python), SQLite, OpenCV/PIL for compositing, MediaPipe for pose
- **Frontend:** Fable (F# → JS), plain HTML5 Canvas (no Three.js/WebGL needed for 2D
  body-wrap)
- **AI:** Claude API for personalized advice text only — enhanced with vision:
  the /advice endpoint also sends the composited try-on image to Claude
  (multimodal input) so the advice text can include visual fit observations
  (e.g. shoulder seam alignment, hem length, fabric pull). This is qualitative
  visual commentary, NOT measurement extraction — it does not override or feed
  into the trained size model (measurements are always manually entered).
  scikit-learn/XGBoost (size classification)
- **Hosting (free tier):** Render/Railway for Flask backend, Netlify/Vercel for
  frontend

---

## 11. Build Roadmap (phases, Mac Terminal)

| Phase | Content | Timing |
|---|---|---|
| 0 | Environment setup (venv, folders, Kaggle CLI, git) | Day 1 |
| 1 | Data collection (Kaggle fit dataset, iMaterialist, Uniqlo men's charts) | Days 1–2 |
| 2 | Synthetic customer generation + labeling | Day 2 |
| 3 | EDA, preprocessing, feature engineering | Days 2–3 |
| 4 | Model training & comparison | Days 3–4 |
| 5 | Fairness audit | Days 4–5 |
| 6 | Flask backend (upload, catalog, recommend, try-on, advice) | Days 5–8 |
| 7 | Fable frontend (4 screens) | Days 7–10 |
| 8 | Deployment (Render + Netlify) | Days 12–13 |
| 9 | Docs, GitHub, presentations | Days 13–14 |

Full command-by-command detail lives in `FitML_Build_Guide.md` (already generated).

---

## 12. Rubric Coverage Checklist

Confirmed against the exact rubric point breakdown (totals to 100):

- [x] Step 1 — Problem framing (10 pts) — Section 1
- [x] Step 2 — Data collection & understanding (10 pts) — Sections 2–3
- [x] Step 3 — Preprocessing, EDA & feature engineering (10 pts) — Section 4
- [x] Step 4 — Model implementation & comparison (20 pts) — Section 5
- [x] Step 5 — Critical thinking, ethical AI & bias auditing (20 pts) — Section 6
- [ ] Step 6 — Final presentation & communication (10 pts) — technical + business decks, pending
- [ ] Step 7 — GitHub repo & upload (15 pts) — public repo, README, requirements.txt,
      final report, clean commit history — pending push
- [ ] Bonus — Creative & well-presented submission (5 pts) — separate criterion from
      Steps 1-7, awarded for originality/design/innovation across the whole
      submission (this is where the AR try-on demo and website polish earn credit)

Note: Steps 8 (Deployment/MLOps) and 9 (Generative AI use) are optional and do
NOT carry their own rubric points in the official breakdown — the 100 points
come entirely from Steps 1-7 (95 pts) + the Bonus criterion (5 pts). Steps 8-9
are worth doing anyway because: (a) Step 8's Flask deployment is genuinely low
extra cost given the backend already exists, and (b) documenting GenAI use
(Step 9) strengthens the overall polish that feeds into the Bonus criterion,
even though neither has a dedicated point line.

Assignment name (as listed on the LMS): "Pillar 5: Capstone Project" — this is
distinct from the project's own title/name (FitML) and matters for the
submission filename (see Section 13).

---

## 13. Submission Files & Naming

Per the rubric's instruction ("Rename the files as Your_Name_Assignment name"),
the assignment name on the LMS is literally "Pillar 5: Capstone Project" — so
the filename should reflect that, not the project's own name (FitML). Each
deliverable uses the base name `Debie_Galleros_Pillar5_CapstoneProject`
with a descriptive suffix:

| Deliverable | Filename |
|---|---|
| Main written report (problem framing, data, EDA, model comparison, fairness analysis — collated per submission instructions) | `Debie_Galleros_Pillar5_CapstoneProject_Report.pdf` |
| Technical presentation deck (Step 6, peers) | `Debie_Galleros_Pillar5_CapstoneProject_TechnicalDeck.pptx` |
| Business presentation deck (Step 6, executives) | `Debie_Galleros_Pillar5_CapstoneProject_BusinessDeck.pptx` |
| Code files (if the LMS requires a direct upload in addition to GitHub) | `Debie_Galleros_Pillar5_CapstoneProject_Code.zip` |
| GitHub repo name (Step 7) | `fitml-capstone` (lowercase-hyphenated, standard GitHub convention — the repo can use the project's own name since it's not the LMS file upload) |

If submitting the main report and both decks as one combined file (the
instructions allow a single approved format: .pdf/.doc/.pptx/.ppt), name it:
`Debie_Galleros_Pillar5_CapstoneProject.pdf` (or matching extension) and
clearly section it internally (Report → Technical Deck → Business Deck)
rather than uploading three separate files, if the submission portal only
accepts one. The report itself should still be titled "FitML: A
Fairness-Audited Machine Learning System for Sizing and AR Try-On in
E-Commerce" on its cover page — the filename follows the LMS assignment
name, the document title follows the project name.

1. Download and inspect the real Kaggle fit dataset (row counts, columns, nulls)
2. Transcribe the small men's-extension slice of Uniqlo's size guide (Section 2)
3. Decide fairness-audit stratification variables using fields already in the real
   dataset (body_type, height bands, etc.) — no separate synthetic personas needed
4. Write `docs/privacy.md` in full (RA 10173 framing)

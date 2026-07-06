# FitML Capstone — Project Instructions for Claude Code

## What this project is
FitML: a fairness-audited size-recommendation + 2D augmented-reality try-on system for e-commerce.
Capstone for AIM PGDAIML, LMS assignment "Pillar 5: Capstone Project", **due July 17, 2026**.
Folder/repo on disk: `~/pillar5-capstone`. GitHub repo name: `fitml-capstone`.
Report title: "FitML: A Fairness-Audited Machine Learning System for Sizing and Augmented Reality Try-On in E-Commerce".

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
  - Fable (F#→JS) frontend with .NET SDK/dotnet setup → superseded. Frontend is
    plain HTML/CSS/JavaScript (vanilla JS + fetch), no framework, no .NET
    toolchain — ignore the Build Guide's Phase 7 dotnet/Fable commands.

## Locked decisions — do not revisit or "improve"
- Models: Logistic Regression, Random Forest, XGBoost, MLP. **No CNNs, no diffusion
  models** — explicitly scoped out. If tree models beat the MLP, report that honestly.
- Model served in production (Phase 8): the /recommend-size endpoint serves the
  class-weighted XGBoost model (models/xgboost_weighted.joblib), not the unweighted
  baseline — for a sizing assistant, catching misfits (small/large) matters more
  than raw accuracy; the confidence % and amber-box layer already communicate
  uncertainty to the user. Document this reasoning in docs/model_selection.md.
- Try-on: 2D body-wrap compositing (MediaPipe pose keypoints + affine-warped
  transparent garment PNG, alpha-composited on HTML5 Canvas). **NOT 3D, NOT Three.js,
  NOT WebGL.** Do not suggest 3D. Size-proportional rendering: when compositing a
  size other than the model-recommended one, scale the garment PNG proportionally
  to that size's measurement ratios from the size chart (an XL on a small-framed
  user renders visibly wider/longer than an M). Lightweight visual size-difference
  cue; the advice text explains the fit tradeoff. Cite fit-aware generative try-on
  (FIT dataset, 2026) as the future-work upgrade path.
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
  Photo upload accepts both full-body and half-body (upper) photos. UI note
  next to the upload control: "For best results, use a full-body photo.
  Half-body works too — but you'll only be able to try on tops." Backend
  (Phase 8): after MediaPipe pose extraction, detect which keypoints are
  visible and store a photo_coverage flag (full_body / upper_body). Try-on
  endpoint: if the selected garment is a bottom (jeans, skirts, trousers,
  shorts) and photo_coverage is upper_body, return a friendly error: "This
  item needs a full-body photo to try on. Update your photo in your profile."
  Tops/dresses composite normally on upper-body photos.
- Claude API role: advice text only (personalized fit explanation, history-based
  brand/fabric notes). Enhanced with vision: the /advice endpoint also sends the
  composited try-on image to Claude (multimodal input) so the advice text can
  include visual fit observations (e.g. shoulder seam alignment, hem length,
  fabric pull). This is qualitative visual commentary, NOT measurement
  estimation — it does not override or feed into the trained size model.
  Document this in Step 9 (GenAI use) as a multimodal AI integration.
  Never measurement estimation, never part of graded ML.
- History suggestions: rule-based DB lookup (2+ prior items same brand/fabric with
  consistent fit → add note to Claude advice prompt). NOT a second trained model.
- Catalog: **Kaggle "Fashion Product Images Dataset"**
  (`paramaggarwal/fashion-product-images-dataset`, full-res, MIT license) —
  plain-background front-facing product/model shots, 1080x1440, with
  structured `styles.csv` metadata (gender, masterCategory, subCategory,
  articleType, baseColour). Superseded iMaterialist: evaluated during Phase
  1/before Phase 7 and rejected — its images are in-the-wild street-style/
  red-carpet/runway photos (busy backgrounds, some with visible third-party
  photographer/publication watermarks), not clean product shots, and
  therefore a poor fit for rembg + affine-warp compositing. The dataset's own
  low-res "small" variant (60x80px) was also rejected as too pixelated for
  compositing — use the full-res version, fetched via targeted single-file
  downloads (~100-115 images needed, not the full ~24.7GB dataset).
  rembg background removal, ~100+ women's + ~10–15 men's items, 2–3
  hue-shifted variants each. Metadata per item: item_id, category, gender,
  color, plausible fabric, size_range, price via random.randint within
  category bands, rounded to nearest 10 PHP: tshirt/tank/shorts 400–700 ·
  polo/blouse 600–1000 · jeans/slacks/sweater 800–1400 · jacket/dress
  1200–1800. Document as programmatically generated demo pricing.
- Color-suitability matching: OUT OF SCOPE. Mention as future work only.
- Product: functional prototype website — no checkout, no payments, no inventory.
  Describe as "functional prototype", never "an online store".
- Stack: Flask backend, plain HTML/CSS/JavaScript frontend (no framework —
  vanilla JS with fetch() calls to the Flask backend), plain Canvas compositing.
  Deploy: Render/Railway (backend) + Netlify/Vercel (frontend), free tiers.
- UI: flat cards, hairline borders, minimal — follow the screen-by-screen spec in
  the Master Plan (sidebar filters, 2-col grid, confidence box with blue/amber
  states, size pills, color swatches). Do not redesign.
- Responsive design: the site must work on both desktop and mobile (mobile-first
  CSS, no horizontal scrolling at any width). Breakpoint adaptations: Catalog
  page — on mobile the left filter sidebar collapses into a "Filters" button
  that opens a slide-in drawer; garment grid stays 2 columns on mobile (cards
  shrink), 3–4 columns on wide desktop. Try-on/item detail — image full-width
  on mobile, size pills and swatches wrap, buttons full-width. Profile form —
  single column on mobile, inputs full-width, touch targets min 44px.
  Landing/About — hero and feature cards stack vertically on mobile. Photo
  upload — on mobile allow direct camera capture (input accept="image/*"
  capture) as well as gallery upload. Test at 375px, 768px, and 1280px+ widths.
- Confidence box borderline rule: predicted size near boundary AND low-stretch
  fabric AND fitted/athletic cut → amber box, recommend one size up, lower displayed
  confidence, tradeoff-explaining advice text. The amber box uses a 💡 lightbulb
  icon (NOT ⚠ warning triangle — too alarming) and opens with "Sizing tip:"
  before the recommendation — the amber state is a helpful suggestion, not a
  warning. Blue box keeps the ✓ checkmark. The advice text has two
  paragraphs. Paragraph 1: measurement-based reasoning in plain language.
  Paragraph 2: starts with "Note:" followed by visual observations from the
  try-on image, written in simple everyday language a non-technical shopper
  understands — no tailoring jargon (avoid terms like "seam alignment", "hem",
  "drape"). Example tone: "Looking at your photo, this fits you well — it sits
  nicely on your shoulders and the length is just right for you." This
  plain-language requirement goes into the Claude API prompt template in the
  /advice endpoint (Phase 8).
- Privacy: uploads under `backend/uploads/{uuid}/`, 24h auto-delete, face blur ON by
  default (MediaPipe + Gaussian), HTTPS only, cite RA 10173 in `docs/privacy.md`.
- Presentation branding: White background on all slides. AIM logo
  (`docs/assets/AIM_logo_2017.svg.png`) in the upper left corner of every slide,
  small (~1 inch / 2.5cm height). No other logos. Title slide of both decks,
  top to bottom: AIM logo upper left, then project title "FitML: A
  Fairness-Audited Machine Learning System for Sizing and Augmented Reality
  Try-On in E-Commerce", then "Pillar 5: Capstone Project", then the program
  name "Postgraduate Diploma in Artificial Intelligence and Machine Learning",
  then "Debie Galleros · July 2026", then the tagline "A virtual fitting room
  with size recommendations and tailored advice".

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
├── frontend/      # plain HTML/CSS/JS app
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
9. Frontend (plain HTML/CSS/JS, six pages per spec)
10. Deployment (Render + Netlify/Vercel)
11. Docs, README, decks, GitHub push, rename submission files per convention

## Working style
- Do not jump ahead of the current phase. Finish, verify (run the code / check
  outputs), summarize what was produced, then wait for go-ahead.
- Everything reproducible: fixed random seeds, saved configs, requirements.txt.
- Never commit raw datasets, kaggle.json, API keys, or uploaded photos.
- Python venv at `./venv` — activate before running anything.

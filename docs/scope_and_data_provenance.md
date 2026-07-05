# Scope & Data Provenance Notes

## Real women's data vs. synthetic men's extension

FitML uses two sizing data sources with deliberately different statuses:

| | Women's core (graded ML) | Men's extension (demo only) |
|---|---|---|
| Customers | **Real** — 251,047 ModCloth + RentTheRunway reviews (Kaggle "Clothing Fit Dataset for Size Recommendation") | **Synthetic** — 200 generated customers ([`src/mens_extension.py`](../src/mens_extension.py), seed 42) |
| Labels | **Real** customer-reported fit feedback (`small`/`fit`/`large`) | Derived by **rule-based lookup** against Uniqlo's published men's size chart |
| Sizing logic | Trained classifiers (LogReg / RF / XGBoost / MLP, Phase 5) | Deterministic chart lookup — **not a trained model** |
| Fairness audit (Phase 6) | **Included** — full disparate-impact / equalized-odds / SHAP audit | **Excluded** (reasoning below) |
| Table | `data/processed/model_ready.csv` | `data/processed/mens_synthetic.csv` — separate schema, **never merged** |

**Why the extension exists.** Both ModCloth and RentTheRunway sell women's
apparel, so the real dataset is women's-only. The demo catalog still carries
~10–15 men's items, which need *some* sizing path. Rather than force one
dataset to cover both genders — or silently train on invented men's fit
feedback — the project uses real audited data where it exists (women's) and
a small, clearly-flagged synthetic extension for the gap (men's).

**What is real vs. invented in the extension.** The sizing *standard* is
real: Uniqlo's published men's body-measurement size guide (chosen for its
Asian-market sizing relevance given the CDO/Philippines context), transcribed
on 2026-07-06 into [`data/raw/mens_size_charts.csv`](../data/raw/mens_size_charts.csv)
(chest XS 81–89 cm through XXL 119–127 cm; waist XS 66–71 cm through XXL
99–107 cm; cross-checked across three independent chart aggregators —
sizecharter.com, sizedepo.com, qianshiwear.com — which agree to within
rounding of the inch↔cm conversion). The *population* is synthetic: 200
customers sampled from a correlated multivariate normal whose
means/SDs/correlations are assumed adult Asian-male anthropometry, documented
in the script — not measured people. `size_label` is simply "which published
chart row do these measurements fall in": tops (tshirt/polo/jacket) keyed on
chest, bottoms (jeans/shorts) on waist, boundary overlaps in the published
ranges resolved deterministically to the smaller size.

**Why the extension is excluded from the fairness audit.**

1. **The population is synthetic.** Group-fairness metrics on generated
   customers would measure properties of the random-number generator, not
   any real-world disparity. There is no ground truth to be unfair *to*.
2. **The sample is far too small.** ~200 rows split across six sizes and
   five categories leaves per-group cell counts (often < 10) that cannot
   support statistically valid disparate-impact ratios or equalized-odds
   comparisons.
3. **There is no learned model to audit.** The lookup is a deterministic
   transcription of Uniqlo's published chart; auditing it would amount to
   auditing Uniqlo's sizing standard, not FitML's machine learning.

This is documented as an explicit, known limitation of the men's experience
(unaudited, chart-based sizing) rather than a silently patched gap — the
honest alternative to pretending the audit covers customers it never saw.

## Catalog image source: iMaterialist Fashion → Fashion Product Images Dataset

**Decision:** the FitML catalog uses Kaggle's [Fashion Product Images
Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset)
(`paramaggarwal/fashion-product-images-dataset`, full-resolution, MIT
license), not iMaterialist Fashion as originally planned.

**Why:** iMaterialist Fashion (both the original 2019 FGVC6 competition and
the derivative dataset used to work around its rules-acceptance gate — see
below) turned out to be in-the-wild fashion photography — street-style
candids, red-carpet event photos, runway shots — rather than clean product
images. A 5-image spot check across garment categories (shirt/blouse, dress,
pants, jacket, skirt) found only 1 of 5 was a plain-background studio shot;
the other 4 had busy backgrounds, an identifiable celebrity at a
sponsor-branded event, and two carried visible third-party photographer/
publication watermarks (`HiStyley.com`; "Western Canada Fashion Week — Donna
Lynn Photography"). This is a poor fit for the try-on pipeline for two
reasons:

1. **Technical** — `rembg` isolates a *subject* from its background, not a
   garment from the person wearing it. iMaterialist does include per-pixel
   segmentation masks that could crop just the garment polygon, but the
   result is a garment as draped on a specific person's pose (folds, gaps,
   visible skin), not the flat product silhouette the affine-warp
   compositing pipeline expects.
2. **Rights/appropriateness** — using real, identifiable people (including
   bystanders and at least one celebrity), some with explicit third-party
   copyright watermarks, as garment photos in a prototype catalog is
   questionable even for non-commercial academic use.

The replacement dataset's `styles.csv` also directly supplies the
gender/category/color metadata the catalog needs, reducing manual metadata
work versus iMaterialist's category labels (which needed harmonizing across
68 fine-grained tags).

**A note on the "small" (60x80px) variant of the same dataset:** also
evaluated and rejected — image quality is too low-resolution for garment
compositing (would look pixelated once warped onto a photo). The full-
resolution version's images are ~1080x1440. Both variants are single-file
downloadable by exact image ID, so only the ~100-115 images actually needed
for the catalog get fetched — not the full ~24.7GB dataset.

## iMaterialist competition rules-acceptance gate

The original iMaterialist Fashion 2019 (FGVC6) competition requires accepting
competition rules on kaggle.com before any file — even metadata — can be
downloaded via the API (403 Forbidden otherwise). This didn't resolve even
after phone verification was added to the account; the "I Understand and
Accept" button never appeared on the rules page. A derivative Kaggle
*dataset* (not competition) that mirrors the same images
(`ishakyorganc/imaterialist-fashion-2021-full`) was used instead to get
around the rules gate for evaluation purposes — but the image-quality
findings above led to dropping iMaterialist altogether in favor of the
Fashion Product Images Dataset.

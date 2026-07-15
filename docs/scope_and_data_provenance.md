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

## Phase 7: catalog curation and garment isolation (July 2026)

**Selection** (`src/catalog_select.py`): `styles.csv` was filtered to
Apparel in Casual/Formal/Smart Casual usage, a solid-colour whitelist, and
name-based exclusion of prints/patterns/multi-packs/kids-mislabels, then
ranked per category ("Solid"-named first, newest year, seeded tiebreak).
A strict recent-year cutoff proved unusable — women's target categories
from 2015+ contain almost no dresses/jeans/skirts (the dataset's apparel
mass is 2011–2012) — so recency is a ranking preference and the
solid-colour/clean-cut filters carry the contemporary aesthetic.

**Garment isolation** (`src/catalog_isolate.py`): every apparel photo
inspected was an on-model shot, so plain `rembg`/u2net (subject-from-
background) would keep the person. The pipeline instead uses rembg's
`u2net_cloth_seg` clothing-parsing model, which returns stacked
upper-body / lower-body / full-body garment bands; the band is chosen by
category (tops→upper, bottoms→lower then full, dresses→full), with plain
u2net as a fallback for the rare flat-lay. Cutout cleanup: alpha
binarisation, largest-connected-component filtering, bbox crop, downscale
to 1200px max.

**Visual QA** (`src/catalog_exclusions.csv`, 147 exclusions over 8
rounds): every cutout was reviewed on contact sheets; rejects were
replaced by the next-ranked candidate (`--extend` mode tops up categories
whose pool runs dry). Common rejection reasons: model/face retained,
hair merged into dark garments, hand-shaped holes where hands covered
the garment, multi-pack product shots, layered inner garments, prints
that slipped the name filter, and heavy brand graphics. Per the
curation decision, logo-heavy items were dropped; note that *every*
men's polo in the dataset is U.S. Polo Assn. branded, so the three kept
polos carry small chest logos — the least-branded available.

**Colour variants** (`src/catalog_variants.py`): 2 named-palette
recolours per item (250 total) via HSV hue/saturation replacement with
median-value matching — gamma when lightening, linear scaling when
darkening (gamma posterises near-white garments). Neutral (black/white/
grey) garments accept any target colour; chromatic ones only targets
≥40° of hue away.

**Metadata** (`src/catalog_metadata.py` → `data/catalog/metadata.csv`):
125 items — 112 women's across 11 categories, 13 men's limited to the
four categories covered by `data/raw/mens_size_charts.csv` (tshirt/polo/
jeans/jacket) so every men's item is sizeable by the Uniqlo chart lookup.
Prices are programmatically generated demo pricing (seeded
`random.randint` within per-category PHP bands, rounded to nearest 10;
skirts assigned to the jeans/slacks 550–900 band). Fabric is a
plausible per-category assignment, not ground truth, mixing stretch and
low-stretch options because fabric stretch feeds the borderline-sizing
(amber box) rule.

## Phase 7 cleanup round: hand-gap artifact repair (July 2026)

After catalog approval, a dedicated cleanup pass addressed the most common
cutout artifact: hand-shaped transparent holes where the model's hands-on-
hips pose covered the garment. `src/catalog_repair.py` (per-item modes in
`src/catalog_repairs.csv`) repairs these against the original photos:

- **default**: enclosed holes and small boundary bites filled with the
  median color of the surrounding fully-opaque fabric ring (+noise);
  size/elongation caps keep genuine between-legs background transparent.
- **source**: transparent regions whose source-photo pixels are NOT white
  studio background were occluded fabric (a hand) — filled precisely in the
  occluder's shape. Genuine see-through windows stay open.
- **window**: additionally fills see-through windows that pierce the
  silhouette (arm-akimbo gaps) where the open hole reads as damage in a
  product image.

Classical inpainting (OpenCV Telea) was evaluated and rejected: it smears
non-fabric colors (waistband linings, skin) into the fill. A source-photo
audit (residual occluded-region pixels per item) drove mode escalation
rather than eyeballing thumbnails alone.

Outcome: 28 items repaired and kept; 1 reverted to its original (18897 —
hem-edge gaps where any fill protrudes outside the silhouette); 4 items
with residual visible flaws after repair were **excluded without
replacement** (25908, 39235, 44587, 41153 — the un-QA'd candidate pool's
pass rate did not justify another replacement round). Final catalog:
**121 items (109 women's, 12 men's), 242 color variants** — still above
the ~100+ women's / 10-15 men's targets. Category quotas in
`catalog_select.py` and `candidates.csv` were reduced to match, so the
shortfall logic does not backfill the four removed slots.

## Catalog display convention: source photos vs cutouts (July 2026)

The catalog browse cards and item-detail hero images show the **original
on-model source photos** (`data/catalog/photos/{id}.jpg`, standardized to
1200px max dimension — like real e-commerce listings). The transparent
garment cutouts (`data/catalog/garments/`) are reserved for two roles:

1. **Try-on compositing** — warped onto the user's photo, where the photo's
   own context masks the cutouts' residual edge artifacts.
2. **Color previews** — the original photo represents the item's native
   color; selecting a different color swatch switches the detail image to
   the hue-shifted cutout, labeled as a color preview (the recolored
   garment exists only as a cutout, so this is presented as a preview
   rather than a product photograph).

metadata.csv carries both paths per item (`photo` for display, `image` for
compositing).

**Final catalog counts (post-Phase-7 reviews, July 2026):** two later QA
rounds each excluded 2 more items with residual cutout flaws (a post-review
cleanup: 57068, 13458; then a Phase 9 live-site review: 22322, 44578 — all
logged in `src/catalog_exclusions.csv`), and the Phase 9 polish pass baked
the 1px Gaussian alpha edge feather into every garment PNG and removed
hairline strand remnants (`src/catalog_polish.py`). A later post-deployment
review excluded one more item — 52465 ("Red Rose Black Camisole") — for a
different reason than the cutout-quality flaws above: its try-on rendered
as a full-length tank when the source product is actually a cropped
bralette-style camisole, a silhouette misrepresentation rather than an
image-quality defect. Final catalog: **116 items (105 women's across 10
categories, 11 men's across the 4 chart-covered categories), 232 color
variants** — still above the ~100+ women's / 10–15 men's targets. This
resolves catalog presentability: cutout edge artifacts never appear as
standalone product imagery. A segmentation-model comparison
(SegFormer-B2-clothes vs the pipeline's u2net_cloth_seg, all 121 items,
source-photo defect audit) found no net quality advantage to switching
models — 59 vs 56 items with near-identical medians, plus 5 SegFormer
outright failures — validating the current cutouts for their compositing
role; transformer-based clothing parsing is noted as future work.

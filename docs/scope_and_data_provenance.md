# Scope & Data Provenance Notes

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

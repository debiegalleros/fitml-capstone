"""Phase 7 addendum — standardized catalog display photos.

The catalog cards and item-detail hero images use the ORIGINAL on-model
source photos (like real e-commerce listings); the transparent cutouts are
reserved for try-on compositing, where the user's photo context masks their
edge artifacts. This copies each active item's source photo into
data/catalog/photos/{id}.jpg (gitignored, like the garment PNGs),
downscaled to a consistent max dimension.
"""


from PIL import Image

from catalog_common import ROOT, active_selection

IMAGES_DIR = ROOT / "data/raw/fashion_product_images/images"
PHOTOS_DIR = ROOT / "data/catalog/photos"
MAX_DIM = 1200
JPEG_QUALITY = 90


def main() -> None:
    PHOTOS_DIR.mkdir(parents=True, exist_ok=True)
    sel = active_selection()
    n = 0
    for _, row in sel.iterrows():
        iid = int(row.id)
        dest = PHOTOS_DIR / f"{iid}.jpg"
        if dest.exists():
            continue
        im = Image.open(IMAGES_DIR / f"{iid}.jpg").convert("RGB")
        if max(im.size) > MAX_DIM:
            im.thumbnail((MAX_DIM, MAX_DIM), Image.LANCZOS)
        im.save(dest, quality=JPEG_QUALITY)
        n += 1
    print(f"{n} photos written, {len(sel)} total in {PHOTOS_DIR}")


if __name__ == "__main__":
    main()

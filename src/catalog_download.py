"""Phase 7 — targeted single-file downloads of catalog images from Kaggle.

Downloads only the images listed in data/catalog/candidates.csv (not the full
~24.7GB dataset) into data/raw/fashion_product_images/images/. Kaggle serves
each file zipped; this unzips and cleans up. Already-downloaded images are
skipped, so the script is resumable.

Usage:
    python src/catalog_download.py             # all selected candidates
    python src/catalog_download.py --sample    # 10-image approval batch only
    python src/catalog_download.py --ids 123 456
"""

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

from catalog_common import CANDIDATES_CSV, ROOT, active_selection

IMAGES_DIR = ROOT / "data/raw/fashion_product_images/images"
DATASET = "paramaggarwal/fashion-product-images-dataset"
KAGGLE = ROOT / "venv/bin/kaggle"

# Approval-batch spread: top-ranked item from 8 women's + 2 men's categories.
SAMPLE_SPREAD = [
    ("women", "Tshirts"), ("women", "Tops"), ("women", "Dresses"),
    ("women", "Shirts"), ("women", "Jeans"), ("women", "Skirts"),
    ("women", "Sweaters"), ("women", "Jackets"),
    ("men", "Polo"), ("men", "Jeans"),
]


def download_one(image_id: int) -> bool:
    dest = IMAGES_DIR / f"{image_id}.jpg"
    if dest.exists():
        print(f"  {image_id}.jpg already present, skipping")
        return True
    result = subprocess.run(
        [str(KAGGLE), "datasets", "download", DATASET,
         "-f", f"fashion-dataset/images/{image_id}.jpg",
         "-p", str(IMAGES_DIR)],
        capture_output=True, text=True,
    )
    # Kaggle serves some files zipped, others as the raw jpg.
    zip_path = IMAGES_DIR / f"{image_id}.jpg.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(IMAGES_DIR)
        zip_path.unlink()
    if result.returncode != 0 or not dest.exists():
        print(f"  FAILED {image_id}: {result.stderr.strip()[:200]}")
        return False
    print(f"  {image_id}.jpg ok ({dest.stat().st_size // 1024} KB)")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true",
                        help="download only the 10-image approval batch")
    parser.add_argument("--ids", nargs="*", type=int,
                        help="download specific image ids")
    args = parser.parse_args()

    if args.ids:
        ids = args.ids
    elif args.sample:
        df = pd.read_csv(CANDIDATES_CSV)
        ids = [
            int(df[(df.target_gender == g) & (df.articleType == a)
                   & (df.rank_in_category == 1)].iloc[0].id)
            for g, a in SAMPLE_SPREAD
        ]
    else:
        # exclusion-aware: also fetches headroom replacements
        ids = active_selection().id.astype(int).tolist()

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"downloading {len(ids)} images to {IMAGES_DIR}")
    failures = [i for i in ids if not download_one(i)]
    if failures:
        print(f"\n{len(failures)} failed: {failures}")
        sys.exit(1)
    print("\nall downloads ok")


if __name__ == "__main__":
    main()

"""Phase 7 — isolate garments from catalog photos into transparent PNGs.

The Fashion Product Images apparel shots are essentially all on-model, so
plain rembg (u2net) would keep the person. This uses rembg's u2net_cloth_seg
model, which segments *clothing* off the wearer and returns a 3x-height image
of stacked bands: upper-body / lower-body / full-body garments. The band is
chosen by the item's category (tops -> upper, bottoms -> lower then full,
dresses -> full); if every preferred band is near-empty the image is assumed
to be a flat-lay and plain u2net is used instead.

Cleanup per cutout: binarize soft alpha (< ALPHA_MIN -> 0), keep the largest
connected alpha component (drops stray fragments like hair wisps detached
from the garment), crop to the alpha bbox with a small margin.

Outputs data/catalog/garments/{id}.png and appends a row per item to
data/catalog/isolation_report.csv (band used, coverage, fallbacks) for QA.

Items listed in src/catalog_exclusions.csv (id,reason — hand-curated during
visual QA) are skipped and replaced by the next-ranked headroom candidate
from the same category, keeping per-category quotas intact.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from rembg import new_session, remove

from catalog_common import ROOT, active_selection

IMAGES_DIR = ROOT / "data/raw/fashion_product_images/images"
GARMENTS_DIR = ROOT / "data/catalog/garments"
REPORT_CSV = ROOT / "data/catalog/isolation_report.csv"

ALPHA_MIN = 60        # binarize threshold for soft mask edges
MIN_COVERAGE = 0.03   # band with < 3% opaque pixels is considered empty
BBOX_MARGIN = 0.02    # crop margin as fraction of max dimension
MAX_DIM = 1200        # downscale cutouts: plenty for canvas compositing

# cloth_seg band order: 0 = upper-body, 1 = lower-body, 2 = full-body
BAND_PREFS = {
    "Tshirts": [0], "Tops": [0], "Shirts": [0], "Sweaters": [0], "Jackets": [0],
    "Camisoles": [0], "Polo": [0],
    "Jeans": [1, 2], "Trousers": [1, 2], "Skirts": [1, 2], "Shorts": [1, 2],
    "Dresses": [2, 0],
}


def cleanup_mask(alpha: np.ndarray) -> np.ndarray:
    """Binarize soft edges and keep only the largest connected component."""
    hard = np.where(alpha >= ALPHA_MIN, alpha, 0).astype(np.uint8)
    n, labels = cv2.connectedComponents((hard > 0).astype(np.uint8))
    if n > 2:  # background + more than one blob
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        hard[labels != sizes.argmax()] = 0
    return hard


def isolate_one(row, cloth_session, u2net_session) -> dict:
    src = IMAGES_DIR / f"{int(row.id)}.jpg"
    im = Image.open(src).convert("RGB")
    stacked = remove(im, session=cloth_session)
    w, bh = stacked.width, stacked.height // 3
    bands = [stacked.crop((0, i * bh, w, (i + 1) * bh)) for i in range(3)]

    chosen, coverage, fallback = None, 0.0, False
    for idx in BAND_PREFS[row.articleType]:
        cov = (np.asarray(bands[idx].split()[3]) >= ALPHA_MIN).mean()
        if cov >= MIN_COVERAGE:
            chosen, coverage = idx, cov
            break
    if chosen is None:  # likely a flat-lay: no person to strip
        cutout = remove(im, session=u2net_session)
        coverage = (np.asarray(cutout.split()[3]) >= ALPHA_MIN).mean()
        fallback = True
    else:
        cutout = bands[chosen]

    rgba = np.asarray(cutout).copy()
    rgba[..., 3] = cleanup_mask(rgba[..., 3])
    ys, xs = np.nonzero(rgba[..., 3])
    if len(ys) == 0:
        return {"id": int(row.id), "status": "EMPTY", "band": chosen,
                "coverage": round(float(coverage), 3), "fallback": fallback}
    margin = int(max(rgba.shape[:2]) * BBOX_MARGIN)
    y0, y1 = max(ys.min() - margin, 0), min(ys.max() + margin, rgba.shape[0])
    x0, x1 = max(xs.min() - margin, 0), min(xs.max() + margin, rgba.shape[1])
    out = Image.fromarray(rgba[y0:y1, x0:x1])
    if max(out.size) > MAX_DIM:
        out.thumbnail((MAX_DIM, MAX_DIM), Image.LANCZOS)

    GARMENTS_DIR.mkdir(parents=True, exist_ok=True)
    out.save(GARMENTS_DIR / f"{int(row.id)}.png")
    return {"id": int(row.id), "status": "ok", "band": chosen,
            "coverage": round(float(coverage), 3), "fallback": fallback,
            "size": f"{out.width}x{out.height}"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", nargs="*", type=int,
                        help="only process these ids (default: full selection)")
    parser.add_argument("--force", action="store_true",
                        help="reprocess even if the garment PNG exists")
    args = parser.parse_args()

    sel = active_selection()
    if args.ids:
        sel = sel[sel.id.isin(args.ids)]

    cloth = new_session("u2net_cloth_seg")
    u2net = new_session("u2net")
    reports = []
    for _, row in sel.iterrows():
        dest = GARMENTS_DIR / f"{int(row.id)}.png"
        if dest.exists() and not args.force:
            continue
        rep = isolate_one(row, cloth, u2net)
        rep.update(gender=row.target_gender, articleType=row.articleType)
        reports.append(rep)
        print(f"{rep['id']} {row.target_gender}/{row.articleType}: "
              f"{rep['status']} band={rep['band']} cov={rep['coverage']}"
              f"{' FALLBACK-u2net' if rep['fallback'] else ''}")

    if reports:
        rep_df = pd.DataFrame(reports)
        if REPORT_CSV.exists():
            rep_df = pd.concat([pd.read_csv(REPORT_CSV), rep_df])
            rep_df = rep_df.drop_duplicates("id", keep="last")
        rep_df.to_csv(REPORT_CSV, index=False)
    print(f"\ndone: {len(reports)} processed, "
          f"{sum(r['status'] != 'ok' for r in reports)} problems")


if __name__ == "__main__":
    main()

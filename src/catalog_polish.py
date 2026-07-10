"""Phase 9 bugfix pass — cutout edge polish + exclusions from live-site QA.

Live-site QA (post-deployment review) surfaced cutout-quality issues that
only show at display size:

1. 46 of 119 base cutouts had essentially BINARY alpha (zero antialiasing
   pixels), so card/hero previews showed staircased edges. The alpha-feather
   locked decision was implemented only at try-on compositing time
   (backend/tryon.py feather_alpha); previews serve the raw PNGs. This pass
   bakes a 1px Gaussian feather (sigma 1.2, same 1-2px spirit as the locked
   decision) into every garment PNG so previews blend too. Try-on still
   applies its own feather on top — the doubled softening is imperceptible.

2. Thin hairline strand remnants (trouser contours below jacket hems on
   22312/17628, loose thread scribbles on 18957) survived isolation. They
   were invisible against the old all-white cards but showed against the
   aligned grey silhouette. A morphological de-wisp (3x3 opening on the
   alpha mask) removes structures 1-2px thin — even where attached to the
   garment — while leaving straps (4px+) and bulk edges alone.

3. Two items excluded without replacement (Phase 7 mechanism: row in
   src/catalog_exclusions.csv, metadata row dropped, files deleted):
   - 22322 (women's tshirt): solid hair mass merged into the left
     shoulder — not strand-thin, not fillable.
   - 44578 (men's jeans): hand-occlusion notch in the right leg. A
     repair_from_source attempt filled the notch but sampled shadow pixels
     into the fill (grey patches + edge halo — worse than the notch), so
     the item is excluded instead. Men's set: 12 -> 11, within the ~10-15
     extension spec; 11341 remains as the men's jeans.

One-off: running the feather twice softens edges further; don't re-run
casually. Run AFTER catalog_repair.py / catalog_variants.py if those rerun.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from catalog_common import ROOT  # noqa: E402

GARMENTS_DIR = ROOT / "data/catalog/garments"
PHOTOS_DIR = ROOT / "data/catalog/photos"
METADATA_CSV = ROOT / "data/catalog/metadata.csv"
EXCLUSIONS_CSV = ROOT / "src/catalog_exclusions.csv"

EXCLUSIONS = {
    22322: ("solid hair mass merged into left shoulder — excluded without "
            "replacement (Phase 9 live-site review)"),
    44578: ("hand notch in right leg; source repair produced grey fill "
            "artifacts — excluded without replacement (Phase 9 live-site "
            "review)"),
}

KERNEL_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def dewisp_and_feather(rgba: np.ndarray) -> tuple[np.ndarray, int]:
    """Remove 1-2px alpha strands, then bake a 1px Gaussian alpha feather."""
    out = rgba.copy()
    alpha = out[..., 3]
    mask = (alpha > 10).astype(np.uint8)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL_3)
    wisps = (mask == 1) & (opened == 0)
    alpha[wisps] = 0
    feathered = cv2.GaussianBlur(alpha.astype(np.float32), (5, 5), 1.2)
    out[..., 3] = feathered.clip(0, 255).astype(np.uint8)
    return out, int(wisps.sum())


def exclude(item_id: int, reason: str) -> None:
    meta = pd.read_csv(METADATA_CSV)
    if item_id not in set(meta.item_id):
        print(f"{item_id}: already excluded")
        return
    meta = meta[meta.item_id != item_id]
    meta.to_csv(METADATA_CSV, index=False)
    excl = pd.read_csv(EXCLUSIONS_CSV)
    if item_id not in set(excl.id):
        excl.loc[len(excl)] = [item_id, reason]
        excl.to_csv(EXCLUSIONS_CSV, index=False)
    removed = []
    for p in list(GARMENTS_DIR.glob(f"{item_id}*.png")) + [PHOTOS_DIR / f"{item_id}.jpg"]:
        if p.exists():
            p.unlink()
            removed.append(p.name)
    print(f"{item_id}: excluded — removed {removed}, catalog now {len(meta)} items")


def main() -> None:
    for item_id, reason in EXCLUSIONS.items():
        exclude(item_id, reason)
    total_wisps = 0
    heavy = []
    pngs = sorted(GARMENTS_DIR.glob("*.png"))
    for p in pngs:
        rgba = np.asarray(Image.open(p).convert("RGBA")).copy()
        polished, wisps = dewisp_and_feather(rgba)
        Image.fromarray(polished).save(p)
        total_wisps += wisps
        if wisps > 2000:
            heavy.append((p.name, wisps))
    print(f"polished {len(pngs)} PNGs, removed {total_wisps}px of wisps")
    print("heaviest de-wisps:", heavy)


if __name__ == "__main__":
    main()

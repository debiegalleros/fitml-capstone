"""Phase 7 — generate hue-shifted color variants of isolated garment PNGs.

Each catalog item gets 2 extra color variants picked (seeded) from a named
target palette, so variant color names are known for metadata/swatches
instead of arbitrary hue rotations. Recoloring works in HSV on the opaque
pixels only:

- hue: set to the target hue
- saturation: blended toward the target (chromatic garments keep some of
  their own saturation texture)
- value: gamma-mapped so the garment's median brightness lands on the
  target's — this is what makes neutral (black/white/grey) garments
  recolorable at all, since a plain hue rotation leaves them unchanged.

Targets whose hue is within MIN_HUE_DIST of the garment's dominant hue are
skipped (a "navy" variant of a navy shirt is not a variant).

Outputs data/catalog/garments/{id}__{variant}.png alongside the base PNG and
writes data/catalog/variants.csv (id, variant_name) for the metadata step.
"""

import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

SEED = 42
N_VARIANTS = 2
MIN_HUE_DIST = 40  # degrees
ROOT = Path(__file__).resolve().parent.parent
GARMENTS_DIR = ROOT / "data/catalog/garments"
VARIANTS_CSV = ROOT / "data/catalog/variants.csv"

# name -> (hue deg, saturation 0-255, target median value 0-255)
PALETTE = {
    "navy": (220, 180, 90), "burgundy": (345, 150, 100),
    "forest green": (140, 120, 90), "mustard": (48, 170, 170),
    "dusty rose": (350, 90, 180), "slate blue": (215, 110, 150),
    "olive": (75, 120, 120), "plum": (290, 110, 110),
    "teal": (180, 140, 120), "terracotta": (15, 140, 150),
}


def dominant_hsv(hsv: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    h = hsv[..., 0][mask]
    s = hsv[..., 1][mask]
    v = hsv[..., 2][mask]
    # circular mean of hue, weighted by saturation so grey pixels don't vote
    w = s.astype(float) + 1e-6
    ang = np.deg2rad(h.astype(float) * 2)  # OpenCV hue is 0-179
    mean_ang = np.arctan2((np.sin(ang) * w).sum(), (np.cos(ang) * w).sum())
    return float(np.rad2deg(mean_ang) % 360), float(np.median(s)), float(np.median(v))


def recolor(rgba: np.ndarray, target: tuple[int, int, int]) -> np.ndarray:
    t_hue, t_sat, t_val = target
    out = rgba.copy()
    mask = out[..., 3] > 0
    hsv = cv2.cvtColor(out[..., :3], cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1].astype(float), hsv[..., 2].astype(float)

    h[mask] = round(t_hue / 2) % 180
    s[mask] = np.clip(0.4 * s[mask] + 0.6 * t_sat, 0, 255)
    med_v = max(np.median(v[mask]), 1.0)
    if t_val < med_v:
        # darkening: linear scale — gamma explodes on near-white garments
        # (median ~250 -> exponent ~50 -> posterized/metallic output)
        v[mask] = np.clip(v[mask] * (t_val / med_v), 0, 255)
    else:
        # lightening: gamma keeps shadows dark so texture survives
        gamma = np.log(max(t_val, 1) / 255.0) / np.log(med_v / 255.0)
        v[mask] = np.clip((v[mask] / 255.0) ** gamma * 255.0, 0, 255)

    hsv[..., 1], hsv[..., 2] = s.astype(np.uint8), v.astype(np.uint8)
    out[..., :3] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return out


def hue_distance(a: float, b: float) -> float:
    d = abs(a - b) % 360
    return min(d, 360 - d)


def main() -> None:
    rng = random.Random(SEED)
    rows = []
    base_pngs = sorted(p for p in GARMENTS_DIR.glob("*.png") if "__" not in p.stem)
    for png in base_pngs:
        rgba = np.asarray(Image.open(png).convert("RGBA"))
        mask = rgba[..., 3] > 0
        hsv = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2HSV)
        dom_h, dom_s, _ = dominant_hsv(hsv, mask)
        neutral = dom_s < 40  # grey/black/white garment: any target hue is fine
        eligible = [
            name for name, (h, _, _) in PALETTE.items()
            if neutral or hue_distance(h, dom_h) >= MIN_HUE_DIST
        ]
        for name in rng.sample(eligible, N_VARIANTS):
            out = recolor(rgba, PALETTE[name])
            slug = name.replace(" ", "-")
            Image.fromarray(out).save(GARMENTS_DIR / f"{png.stem}__{slug}.png")
            rows.append({"id": int(png.stem), "variant": name, "slug": slug})
        print(f"{png.stem}: {[r['variant'] for r in rows[-N_VARIANTS:]]}")

    pd.DataFrame(rows).to_csv(VARIANTS_CSV, index=False)
    print(f"\n{len(rows)} variants for {len(base_pngs)} items")


if __name__ == "__main__":
    main()

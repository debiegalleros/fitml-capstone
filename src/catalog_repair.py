"""Phase 7 cleanup — repair hand-gap artifacts in garment cutouts.

The dataset's on-model poses (hands on hips) leave hand-shaped transparent
holes in otherwise-approved cutouts. Items listed in src/catalog_repairs.csv
get repaired in place:

1. Dark garments only: opaque near-white fringe blobs touching transparency
   (background slivers the mask kept, e.g. between belt loops) are made
   transparent so step 2 treats them as holes.
2. Interior holes (transparent regions fully enclosed by garment) are
   re-opaqued — these are where a hand covered fabric.
3. Small boundary bites are closed morphologically (radius ~1.2% of the
   image, so genuine concavities like the gap between trouser legs survive).
4. RGB under all transparent pixels is inpainted (Telea) from garment
   texture before the new alpha is applied, so filled areas get plausible
   fabric color instead of stale background pixels.

Originals are backed up to data/catalog/garments_prerepair/ for the
before/after review sheet. Variants must be regenerated afterwards.
"""

import shutil

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from catalog_common import ROOT

GARMENTS_DIR = ROOT / "data/catalog/garments"
BACKUP_DIR = ROOT / "data/catalog/garments_prerepair"
REPAIRS_CSV = ROOT / "src/catalog_repairs.csv"


def strip_white_fringe(rgba: np.ndarray) -> np.ndarray:
    """On dark garments, turn near-white blobs touching transparency into
    holes (they are background slivers, not fabric)."""
    alpha = rgba[..., 3]
    mask = alpha > 0
    hsv = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2HSV)
    if np.median(hsv[..., 2][mask]) >= 150:
        return rgba  # light garment: white pixels are probably fabric
    # 215 not 235: white background caught in the mask is often slightly
    # shadowed by the model's body (V ~220-235)
    whiteish = mask & (hsv[..., 2] > 215) & (hsv[..., 1] < 30)
    if not whiteish.any():
        return rgba
    # keep only white blobs that touch transparency (fringe, not highlights)
    n, labels = cv2.connectedComponents(whiteish.astype(np.uint8))
    near_transparent = cv2.dilate((~mask).astype(np.uint8),
                                  np.ones((3, 3), np.uint8)) > 0
    for lbl in range(1, n):
        blob = labels == lbl
        if (blob & near_transparent).any():
            rgba[..., 3][blob] = 0
    return rgba


MAX_HOLE_FRAC = 0.04   # holes bigger than this fraction of the garment are
                       # real background (e.g. between trouser legs) — keep
MIN_MASK_CONTACT = 0.75  # boundary holes must be mostly surrounded by fabric


def hole_components(mask: np.ndarray) -> np.ndarray:
    """Label transparent regions not reachable from the image border."""
    inv = (1 - mask).astype(np.uint8)
    ff = inv.copy()
    ff_mask = np.zeros((inv.shape[0] + 2, inv.shape[1] + 2), np.uint8)
    for pt in [(0, 0), (inv.shape[1] - 1, 0), (0, inv.shape[0] - 1),
               (inv.shape[1] - 1, inv.shape[0] - 1)]:
        if ff[pt[1], pt[0]]:
            cv2.floodFill(ff, ff_mask, pt, 0)
    return ff.astype(bool)


def repair(rgba: np.ndarray, aggressive: bool = False) -> tuple[np.ndarray, int]:
    rgba = strip_white_fringe(rgba)
    mask = (rgba[..., 3] > 0).astype(np.uint8)
    mask_b = mask.astype(bool)
    garment_px = int(mask.sum())
    max_hole = int(MAX_HOLE_FRAC * garment_px)
    ys, xs = np.nonzero(mask)
    gh, gw = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1

    # candidates: enclosed holes + channel-connected holes exposed by a wide
    # closing (hand-at-hip holes connect to background through a narrow gap
    # between arm and torso, so plain interior detection misses them).
    # aggressive mode (per-item opt-in) bridges wider channels for stubborn
    # arm-akimbo holes.
    interior = hole_components(mask)
    r = max(48 if aggressive else 20, int(0.025 * max(mask.shape)))
    min_contact = 0.5 if aggressive else MIN_MASK_CONTACT
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel).astype(bool)
    candidates = (interior | (closed & ~mask_b)).astype(np.uint8)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(candidates)
    ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    rim_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    out = rgba.copy()
    filled = 0
    rng = np.random.default_rng(42)
    for lbl in range(1, n):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area == 0 or area > max_hole:
            continue  # too big: genuine background window
        # elongated slivers are between-legs / along-silhouette background,
        # never hand holes — leave them transparent
        if (stats[lbl, cv2.CC_STAT_HEIGHT] > 0.35 * gh
                or stats[lbl, cv2.CC_STAT_WIDTH] > 0.5 * gw):
            continue
        comp = labels == lbl
        # require the hole to be mostly hugged by fabric, so shallow notches
        # in the true silhouette (waist curves etc.) are left alone
        shell = cv2.dilate(comp.astype(np.uint8), ring_kernel).astype(bool) & ~comp
        contact = mask_b[shell].mean()
        if not interior[comp].any() and contact < min_contact:
            continue
        # fill with the median color of the surrounding fabric ring, plus a
        # touch of noise — inpainting smears in non-fabric colors (waistband
        # linings, skin) when they border the hole
        # sample only fully-opaque fabric: semi-transparent rim pixels carry
        # skin/background tints that poison the median
        ring = shell & (rgba[..., 3] == 255)
        if not ring.any():
            continue
        median = np.median(rgba[..., :3][ring], axis=0)
        # paint slightly past the hole so anti-aliased dark rim pixels are
        # covered too, but never out into true background
        paint = cv2.dilate(comp.astype(np.uint8), rim_kernel).astype(bool)
        paint &= (mask_b | candidates.astype(bool))
        noise = rng.normal(0, 4, (int(paint.sum()), 3))
        out[..., :3][paint] = np.clip(median + noise, 0, 255).astype(np.uint8)
        out[..., 3][paint] = 255
        filled += area
    return out, filled


def source_occluders(item_id: int, article: str, rgba: np.ndarray) -> np.ndarray | None:
    """Mask of transparent pixels that are NOT white studio background in the
    source photo — i.e. something (a hand) covered the garment there.

    Reproduces the isolation crop/resize so source pixels align with the
    cutout. Returns None if the crop cannot be reproduced.
    """
    from catalog_isolate import (ALPHA_MIN, BAND_PREFS, BBOX_MARGIN, IMAGES_DIR,
                                 MIN_COVERAGE, cleanup_mask)
    from rembg import new_session, remove

    src_img = Image.open(IMAGES_DIR / f"{item_id}.jpg").convert("RGB")
    stacked = remove(src_img, session=new_session("u2net_cloth_seg"))
    w, bh = stacked.width, stacked.height // 3
    for idx in BAND_PREFS[article]:
        band = stacked.crop((0, idx * bh, w, (idx + 1) * bh))
        if (np.asarray(band.split()[3]) >= ALPHA_MIN).mean() >= MIN_COVERAGE:
            break
    band_rgba = np.asarray(band).copy()
    band_rgba[..., 3] = cleanup_mask(band_rgba[..., 3])
    ys, xs = np.nonzero(band_rgba[..., 3])
    margin = int(max(band_rgba.shape[:2]) * BBOX_MARGIN)
    y0, y1 = max(ys.min() - margin, 0), min(ys.max() + margin, band_rgba.shape[0])
    x0, x1 = max(xs.min() - margin, 0), min(xs.max() + margin, band_rgba.shape[1])
    crop = src_img.crop((x0, y0, x1, y1))
    crop = crop.resize((rgba.shape[1], rgba.shape[0]), Image.LANCZOS)

    hsv = cv2.cvtColor(np.asarray(crop), cv2.COLOR_RGB2HSV)
    background = (hsv[..., 2] > 235) & (hsv[..., 1] < 20)
    return (rgba[..., 3] == 0) & ~background


def repair_from_source(item_id: int, article: str, rgba: np.ndarray,
                       fill_windows: bool = False) -> tuple[np.ndarray, int]:
    """Fill hand-shaped holes identified against the source photo.

    fill_windows additionally fills genuine see-through windows (white
    background in the source, e.g. arm-akimbo gaps piercing a dress) — used
    per-item where the window reads as damage in the product image. Fills
    follow the actual hole contour, so silhouettes stay natural.
    """
    rgba = strip_white_fringe(rgba)
    mask_b = rgba[..., 3] > 0
    occ = source_occluders(item_id, article, rgba)
    if fill_windows:
        occ = rgba[..., 3] == 0
    garment_px = int(mask_b.sum())
    max_hole_frac = 0.08 if fill_windows else MAX_HOLE_FRAC
    min_contact = 0.3 if fill_windows else 0.4
    ys, xs = np.nonzero(mask_b)
    gh, gw = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1

    # gate to the garment's closed hull: a hand ON the garment lies inside
    # it, while the arm it connects to (also skin, also transparent) lies
    # outside — without this the hand+arm merge into one oversized component
    # that the size caps reject. window mode bridges wider channels.
    r = max(70 if fill_windows else 40, int(0.033 * max(mask_b.shape)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    hull = cv2.morphologyEx(mask_b.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    occ &= hull

    n, labels, stats, _ = cv2.connectedComponentsWithStats(occ.astype(np.uint8))
    ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    rim_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    out = rgba.copy()
    filled = 0
    rng = np.random.default_rng(42)
    for lbl in range(1, n):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area < 40 or area > max_hole_frac * garment_px:
            continue
        if (stats[lbl, cv2.CC_STAT_HEIGHT] > 0.35 * gh
                or stats[lbl, cv2.CC_STAT_WIDTH] > 0.5 * gw):
            continue  # e.g. the model's shirt above a waistband
        comp = labels == lbl
        shell = cv2.dilate(comp.astype(np.uint8), ring_kernel).astype(bool) & ~comp
        if mask_b[shell].mean() < min_contact:
            continue  # isolated island, not a hole hugged by fabric
        # sample only fully-opaque fabric: semi-transparent rim pixels carry
        # skin/background tints that poison the median
        ring = shell & (rgba[..., 3] == 255)
        if not ring.any():
            continue
        median = np.median(rgba[..., :3][ring], axis=0)
        paint = cv2.dilate(comp.astype(np.uint8), rim_kernel).astype(bool)
        paint &= (mask_b | occ)
        noise = rng.normal(0, 4, (int(paint.sum()), 3))
        out[..., :3][paint] = np.clip(median + noise, 0, 255).astype(np.uint8)
        out[..., 3][paint] = 255
        filled += area
    return out, filled


def main() -> None:
    repairs = pd.read_csv(REPAIRS_CSV)
    cand = pd.read_csv(ROOT / "data/catalog/candidates.csv").set_index("id")
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    for _, row in repairs.iterrows():
        src = GARMENTS_DIR / f"{int(row.id)}.png"
        backup = BACKUP_DIR / f"{int(row.id)}.png"
        if not backup.exists():
            shutil.copy2(src, backup)
        rgba = np.asarray(Image.open(backup).convert("RGBA")).copy()
        mode = str(row.get("mode", ""))
        if mode in ("source", "window"):
            article = cand.loc[int(row.id)].articleType
            fixed, filled = repair_from_source(int(row.id), article, rgba,
                                               fill_windows=(mode == "window"))
        else:
            fixed, filled = repair(rgba)
        Image.fromarray(fixed).save(src)
        pct = 100 * filled / (fixed[..., 3] > 0).sum()
        print(f"{int(row.id)}: filled {filled}px ({pct:.1f}% of garment) — {row.note}")


if __name__ == "__main__":
    main()

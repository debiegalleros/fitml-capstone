"""Phase 7 — build data/catalog/metadata.csv for the FitML demo catalog.

One row per catalog item (variants share the row via variant_colors):
item_id, category, gender, color, variant_colors, fabric, size_range,
price_php, product_name, image, photo.

Display convention: `photo` (the original on-model source photo,
data/catalog/photos/) is the catalog card / item-detail hero image;
`image` (the transparent cutout) is used for try-on compositing and as the
"color preview" shown when a non-native color swatch is selected.

Pricing is programmatically generated demo pricing (documented as such, not
real merchant data): random.randint within the category band, rounded to the
nearest 10 PHP. Bands per CLAUDE.md: tshirt/tank/shorts 250-450,
polo/blouse 400-650, jeans/slacks/skirt/sweater 550-900,
jacket/dress 750-1200 (skirts assigned to the jeans/slacks band).

Fabric is a plausible per-category assignment (not ground truth — documented
in the data dictionary); low-stretch fabrics matter later for the amber-box
borderline rule, so each category pool mixes stretch and low-stretch options.
"""

import random
from pathlib import Path

import pandas as pd

from catalog_common import ROOT, active_selection

SEED = 42
VARIANTS_CSV = ROOT / "data/catalog/variants.csv"
GARMENTS_DIR = ROOT / "data/catalog/garments"
OUT_CSV = ROOT / "data/catalog/metadata.csv"

# articleType (+gender where needed) -> catalog category
CATEGORY_MAP = {
    "Tshirts": "tshirt", "Camisoles": "tank", "Shorts": "shorts",
    "Polo": "polo", "Tops": "blouse", "Shirts": "blouse",
    "Jeans": "jeans", "Trousers": "slacks", "Skirts": "skirt",
    "Sweaters": "sweater", "Jackets": "jacket", "Dresses": "dress",
}

PRICE_BANDS = {
    "tshirt": (250, 450), "tank": (250, 450), "shorts": (250, 450),
    "polo": (400, 650), "blouse": (400, 650),
    "jeans": (550, 900), "slacks": (550, 900), "skirt": (550, 900),
    "sweater": (550, 900),
    "jacket": (750, 1200), "dress": (750, 1200),
}

FABRICS = {
    "tshirt": ["cotton jersey", "cotton-spandex blend"],
    "tank": ["cotton-modal blend", "cotton jersey"],
    "shorts": ["cotton twill", "linen-cotton blend"],
    "polo": ["cotton pique"],
    "blouse": ["polyester crepe", "rayon", "cotton poplin"],
    "jeans": ["stretch denim", "rigid denim"],
    "slacks": ["polyester-viscose twill", "cotton twill"],
    "skirt": ["polyester twill", "stretch cotton twill"],
    "sweater": ["cotton knit", "acrylic knit"],
    "jacket": ["polyester shell", "cotton twill"],
    "dress": ["viscose jersey", "polyester crepe"],
}

SIZE_RANGES = {"women": "XS,S,M,L,XL", "men": "XS,S,M,L,XL,XXL"}  # men per Uniqlo charts

# Simple generated care instructions per fabric (documented demo convention,
# not manufacturer-verified — shown in the catalog card's Composition & Care panel).
CARE = {
    "acrylic knit": "Machine wash cold, lay flat to dry.",
    "cotton jersey": "Machine wash cold, tumble dry low.",
    "cotton knit": "Machine wash cold, tumble dry low.",
    "cotton pique": "Machine wash cold, tumble dry low.",
    "cotton poplin": "Machine wash cold, hang to dry.",
    "cotton twill": "Machine wash cold, tumble dry low.",
    "cotton-modal blend": "Machine wash cold, tumble dry low.",
    "cotton-spandex blend": "Machine wash cold, tumble dry low, do not bleach.",
    "linen-cotton blend": "Machine wash cold on gentle cycle, hang to dry.",
    "polyester crepe": "Machine wash cold on gentle cycle, hang to dry.",
    "polyester shell": "Machine wash cold, tumble dry low.",
    "polyester twill": "Machine wash cold, tumble dry low.",
    "polyester-viscose twill": "Dry clean recommended, or hand wash cold.",
    "rayon": "Hand wash cold, hang to dry.",
    "rigid denim": "Machine wash cold inside out, tumble dry low.",
    "stretch cotton twill": "Machine wash cold, tumble dry low.",
    "stretch denim": "Machine wash cold inside out, hang to dry.",
    "viscose jersey": "Hand wash cold, hang to dry.",
}


def main() -> None:
    rng = random.Random(SEED)
    df = active_selection()
    variants = pd.read_csv(VARIANTS_CSV)

    rows = []
    for _, r in df.sort_values(["target_gender", "articleType", "rank_in_category"]).iterrows():
        png = GARMENTS_DIR / f"{int(r.id)}.png"
        if not png.exists():  # not part of the processed selection
            continue
        cat = CATEGORY_MAP[r.articleType]
        lo, hi = PRICE_BANDS[cat]
        v = variants[variants.id == r.id].variant.tolist()
        fabric = rng.choice(FABRICS[cat])
        rows.append({
            "item_id": int(r.id),
            "category": cat,
            "gender": r.target_gender,
            "color": str(r.baseColour).lower(),
            "variant_colors": "|".join(v),
            "fabric": fabric,
            "care": CARE[fabric],
            "size_range": SIZE_RANGES[r.target_gender],
            "price_php": round(rng.randint(lo, hi) / 10) * 10,
            "product_name": r.productDisplayName,
            "image": f"garments/{int(r.id)}.png",
            "photo": f"photos/{int(r.id)}.jpg",
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"{len(out)} items -> {OUT_CSV}")
    print(out.groupby(["gender", "category"]).agg(
        n=("item_id", "size"), price_min=("price_php", "min"),
        price_max=("price_php", "max")).to_string())


if __name__ == "__main__":
    main()

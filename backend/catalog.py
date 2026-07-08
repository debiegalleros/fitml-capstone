"""Catalog access: metadata.csv loaded once, filterable for GET /catalog.

Display convention (locked decision): catalog cards and item-detail heroes
serve the `photo` column (on-model original); the `image` cutout PNG is used
only for try-on compositing and color-swatch previews.
"""
import os

import pandas as pd

from config import METADATA_CSV, CATALOG_DIR

_df = pd.read_csv(METADATA_CSV, dtype={"item_id": str})


def _brand(product_name: str) -> str:
    for sep in (" Women ", " Men "):
        if sep in product_name:
            return product_name.split(sep)[0]
    return product_name.split()[0]


_df["brand"] = _df["product_name"].map(_brand)


def all_items() -> pd.DataFrame:
    return _df


def get_item(item_id: str):
    rows = _df[_df["item_id"] == str(item_id)]
    return rows.iloc[0].to_dict() if len(rows) else None


def filter_items(args) -> list:
    """Apply query-string filters: category, gender, size, fabric, color,
    price_min/price_max."""
    df = _df
    for col in ("category", "gender", "color", "fabric"):
        value = args.get(col)
        if value:
            df = df[df[col].str.lower() == value.strip().lower()]
    size = args.get("size")
    if size:
        df = df[df["size_range"].str.split(",").map(
            lambda sizes: size.strip().upper() in [s.strip() for s in sizes])]
    if args.get("price_min"):
        df = df[df["price_php"] >= float(args["price_min"])]
    if args.get("price_max"):
        df = df[df["price_php"] <= float(args["price_max"])]

    items = []
    for _, row in df.iterrows():
        items.append({
            "item_id": row["item_id"],
            "product_name": row["product_name"],
            "brand": row["brand"],
            "category": row["category"],
            "gender": row["gender"],
            "color": row["color"],
            "variant_colors": row["variant_colors"].split("|") if row["variant_colors"] else [],
            "fabric": row["fabric"],
            "care": row["care"],
            "size_range": [s.strip() for s in row["size_range"].split(",")],
            "price_php": int(row["price_php"]),
            # browse/detail image = on-model photo (display convention)
            "photo_url": f"/images/{row['photo']}",
            # cutout = try-on + swatch previews only
            "cutout_url": f"/images/{row['image']}",
        })
    return items


def garment_png_path(item: dict, color: str = None) -> str:
    """Resolve the cutout PNG for a base item or one of its hue variants."""
    base = os.path.join(CATALOG_DIR, item["image"])
    if not color or color.strip().lower() == item["color"].lower():
        return base
    slug = color.strip().lower().replace(" ", "-")
    stem, ext = os.path.splitext(base)
    variant = f"{stem}__{slug}{ext}"
    return variant if os.path.exists(variant) else base

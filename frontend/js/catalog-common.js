/* FitML — helpers shared by the catalog grid and the item detail page:
   color hex lookup, bottom-vs-top silhouette selection, and the small
   templated description/fabric strings. */

// Swatch-dot hex map — covers both the base `color` column AND every
// hue-shifted `variant_colors` name (a larger, separate vocabulary from the
// base filter list in catalog.js). Audited against data/catalog/metadata.csv
// directly; anything missing here silently fell back to grey, which read as
// a color mismatch on the card (e.g. an olive variant showing a grey dot).
const COLOR_HEX = {
  beige: "#e3d5b8", black: "#1a1a1a", blue: "#2f6fcb", brown: "#7b4b2a",
  burgundy: "#6d1b2f", charcoal: "#3b3b3b", cream: "#f2e9d8",
  "dusty rose": "#c98fa0", "forest green": "#2e5339", green: "#3e7d4c",
  grey: "#9b9b93", "grey melange": "#b7b2a9", lavender: "#c7b8e8",
  mustard: "#d4ac2b", navy: "#1b2a4a", "navy blue": "#1f3864",
  olive: "#6b6b3a", orange: "#e07b39", pink: "#e8a0bf", plum: "#7d3c5c",
  purple: "#7b4b9e", red: "#c0392b", "slate blue": "#5b6d8c",
  teal: "#1f7a72", terracotta: "#c1663a", white: "#ffffff",
  yellow: "#e8c547",
};

// Garment cutouts vary a lot in intrinsic aspect ratio (pants are tall and
// narrow, jackets are wider), and object-fit: contain scales the silhouette
// and the cutout independently based on their own ratios. One generic
// full-body shape left the silhouette peeking out the sides for pants and
// below the hem for jackets — so bottoms get a legs-only silhouette shaped
// to match that narrower cutout ratio, everything else gets the full torso.
const BOTTOM_CATEGORIES = new Set(["jeans", "shorts", "skirt", "slacks"]);
function _silhouetteFor(category) {
  return BOTTOM_CATEGORIES.has(category) ? "images/silhouette-bottom.svg" : "images/silhouette.svg";
}

function _titleCase(s) {
  return s.replace(/(^|\s|-)\w/g, (c) => c.toUpperCase());
}
function _slugify(color) {
  return color.trim().toLowerCase().replace(/\s+/g, "-");
}
function _variantCutoutUrl(baseCutoutUrl, color) {
  const slug = _slugify(color);
  return baseCutoutUrl.replace(/(\.[^./]+)$/, `__${slug}$1`);
}
function _fabricDescription(fabric) {
  const label = _titleCase(fabric);
  return fabric.includes("blend") ? label : `100% ${label}`;
}
function _productDescription(item) {
  const extra = (item.variant_colors || []).length
    ? ` Also available in ${item.variant_colors.map(_titleCase).join(", ")}.`
    : "";
  return `This ${item.product_name} is crafted from ${_fabricDescription(item.fabric).toLowerCase()}, ` +
    `designed as an everyday ${_titleCase(item.category)} with a comfortable fit. ` +
    `Shown here in ${_titleCase(item.color)}.${extra}`;
}

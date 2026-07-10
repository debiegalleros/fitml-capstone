/* FitML — helpers shared by the catalog grid and the item detail page:
   color hex lookup, cutout preloading, and the small templated
   description/fabric strings. */

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

// Swap an <img> to a new source only after the file has decoded, so the
// card never sits empty while a large cutout PNG streams in (on the
// deployed site this used to blank cards for seconds). Calls done(ok)
// after the swap — or with ok=false on error.
function _swapWhenLoaded(imgEl, newSrc, done) {
  const pre = new Image();
  pre.onload = () => {
    imgEl.src = newSrc;
    done(true);
  };
  pre.onerror = () => done(false);
  pre.src = newSrc;
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

// Informational size charts for the "Size measurements" panel (display
// only — never used for the actual /recommend-size calculation). Women's
// bust/hip figures match the demo chart already used server-side for
// size-proportional try-on rendering (size_logic.py); waist is a newly
// authored demo row in the same convention, not from that file. Men's
// figures are the real Uniqlo-derived ranges in data/raw/mens_size_charts.csv
// (identical across all five chart categories, so one shared table works).
const WOMENS_CHART = {
  XS: { bust: 82, waist: 66, hip: 88 },
  S: { bust: 87, waist: 71, hip: 93 },
  M: { bust: 93, waist: 77, hip: 99 },
  L: { bust: 99, waist: 83, hip: 105 },
  XL: { bust: 106, waist: 90, hip: 112 },
  XXL: { bust: 113, waist: 97, hip: 119 },
};
const MENS_CHART = {
  XS: { chest: [81, 89], waist: [66, 71] },
  S: { chest: [89, 97], waist: [69, 76] },
  M: { chest: [97, 104], waist: [76, 84] },
  L: { chest: [104, 112], waist: [84, 91] },
  XL: { chest: [112, 119], waist: [91, 99] },
  XXL: { chest: [119, 127], waist: [99, 107] },
};

function _sizeChartHeader(item) {
  return item.gender === "men"
    ? "<tr><th>Size</th><th>Chest</th><th>Waist</th></tr>"
    : "<tr><th>Size</th><th>Bust</th><th>Waist</th><th>Hip</th></tr>";
}

function _sizeChartRows(item) {
  if (item.gender === "men") {
    return item.size_range
      .filter((s) => MENS_CHART[s])
      .map((s) => {
        const c = MENS_CHART[s];
        return `<tr><td>${s}</td><td>${c.chest[0]}–${c.chest[1]} cm</td><td>${c.waist[0]}–${c.waist[1]} cm</td></tr>`;
      })
      .join("");
  }
  return item.size_range
    .filter((s) => WOMENS_CHART[s])
    .map((s) => {
      const c = WOMENS_CHART[s];
      return `<tr><td>${s}</td><td>${c.bust} cm</td><td>${c.waist} cm</td><td>${c.hip} cm</td></tr>`;
    })
    .join("");
}

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

// ------------------------------------------------- silhouette alignment
// The silhouette and the garment cutout used to be two independent
// object-fit:contain images, each scaled by its own intrinsic aspect ratio —
// so the grey body poked out unpredictably (head over collars, torso below
// short jackets, legs beside narrow jeans) and read as a rendering artifact.
// Instead, compute the garment's actual drawn rectangle inside its contain
// box and size/position the silhouette against it.
//
// Geometry constants from the SVGs themselves:
//   silhouette.svg (300x370): torso spans x 82-218 (w 136), shoulder line y~111
//   silhouette-bottom.svg (200x460): legs span x 38-162 (w 124), waist top y~8
function _alignSilhouette(imgEl, silEl, category) {
  const iw = imgEl.naturalWidth, ih = imgEl.naturalHeight;
  if (!iw || !ih) return;
  const cs = getComputedStyle(imgEl);
  const padX = parseFloat(cs.paddingLeft) || 0;
  const padY = parseFloat(cs.paddingTop) || 0;
  const boxW = imgEl.clientWidth - 2 * padX;
  const boxH = imgEl.clientHeight - 2 * padY;
  if (boxW <= 0 || boxH <= 0) return;
  const s = Math.min(boxW / iw, boxH / ih);
  const gw = iw * s, gh = ih * s;                    // garment drawn size
  const gx = padX + (boxW - gw) / 2;                 // garment drawn origin
  const gy = padY + (boxH - gh) / 2;

  let silW, silH, silX, silY;
  if (BOTTOM_CATEGORIES.has(category)) {
    silW = gw * (200 / 124);
    silH = silW * (460 / 200);
    silX = gx + gw / 2 - silW / 2;
    silY = gy + 0.01 * gh - silH * (8 / 460);        // waistband at garment top
  } else {
    silW = gw * (300 / 136);
    silH = silW * (370 / 300);
    silX = gx + gw / 2 - silW / 2;
    silY = gy + 0.04 * gh - silH * (111 / 370);      // shoulders near garment top
  }
  silEl.style.left = `${silX}px`;
  silEl.style.top = `${silY}px`;
  silEl.style.width = `${silW}px`;
  silEl.style.height = `${silH}px`;
  silEl.style.right = "auto";
  silEl.style.bottom = "auto";
  silEl.style.padding = "0";
}

function _resetSilhouette(silEl) {
  ["left", "top", "width", "height", "right", "bottom", "padding"]
    .forEach((p) => silEl.style.removeProperty(p));
}

// Swap an <img> to a new source only after the file has decoded, so the
// grey silhouette is never shown alone while a large cutout PNG streams in
// (on the deployed site this used to leave silhouette-only cards for
// seconds). Calls done(ok) after the swap — or with ok=false on error.
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

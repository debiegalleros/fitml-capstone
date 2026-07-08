/* FitML — Catalog page: category tab bar, filter drawer (draft-until-"Show
   items"), sort, search, wishlist, and a grid of items with per-item color
   swatch dots that swap the card image to that color's cutout preview
   inline (no navigation). Non-native colors composite over a neutral
   silhouette placeholder so the preview never reads as a disembodied
   garment. Each card also has a collapsible details panel (sizes, size
   measurements, description, composition & care, and — if a profile
   exists — a size recommendation). */

const CATEGORIES = [
  "blouse", "dress", "jacket", "jeans", "polo", "shorts",
  "skirt", "slacks", "sweater", "tank", "tshirt",
];
const CATEGORIES_BY_GENDER = {
  men: ["jacket", "jeans", "polo", "tshirt"],
  women: ["blouse", "dress", "jacket", "jeans", "shorts", "skirt", "slacks", "sweater", "tank", "tshirt"],
};
const FABRICS = [
  "acrylic knit", "cotton jersey", "cotton knit", "cotton pique",
  "cotton poplin", "cotton twill", "cotton-modal blend",
  "cotton-spandex blend", "linen-cotton blend", "polyester crepe",
  "polyester shell", "polyester twill", "polyester-viscose twill",
  "rayon", "rigid denim", "stretch cotton twill", "stretch denim",
  "viscose jersey",
];
const SIZES = ["XS", "S", "M", "L", "XL", "XXL"];

// Filterable colors (matches the catalog's base `color` column, which is
// all /catalog?color= actually filters on).
const COLORS = [
  "beige", "black", "blue", "brown", "charcoal", "cream", "green", "grey",
  "grey melange", "lavender", "navy blue", "olive", "orange", "pink",
  "purple", "red", "white", "yellow",
];

// COLOR_HEX, BOTTOM_CATEGORIES/_silhouetteFor, and the small templated
// description helpers live in catalog-common.js (shared with item.js).

const WISHLIST_KEY = "fitml_wishlist";

// WOMENS_CHART, MENS_CHART, and _sizeChartRows/_sizeChartHeader live in
// catalog-common.js (shared with item.js).

function getWishlist() {
  try {
    return JSON.parse(localStorage.getItem(WISHLIST_KEY) || "[]");
  } catch (e) {
    return [];
  }
}
function isWishlisted(itemId) {
  return getWishlist().includes(String(itemId));
}
function toggleWishlist(itemId) {
  const ids = getWishlist();
  const idx = ids.indexOf(String(itemId));
  if (idx >= 0) ids.splice(idx, 1);
  else ids.push(String(itemId));
  localStorage.setItem(WISHLIST_KEY, JSON.stringify(ids));
  return idx < 0;
}

function defaultFilters() {
  return { gender: "", category: "", color: "", size: "", fabric: "", price_min: "", price_max: "", sort: "default" };
}

const params = new URLSearchParams(window.location.search);
const wishlistMode = params.get("wishlist") === "1";
const searchQuery = (params.get("q") || "").trim().toLowerCase();
const appliedFilters = defaultFilters();
if (params.get("gender") && ["women", "men"].includes(params.get("gender"))) {
  appliedFilters.gender = params.get("gender");
}
if (params.get("category") && CATEGORIES.includes(params.get("category"))) {
  appliedFilters.category = params.get("category");
}
let draft = { ...appliedFilters };

// ------------------------------------------------------------ category tabs

function categoryListForGender(gender) {
  return CATEGORIES_BY_GENDER[gender] || CATEGORIES;
}

function renderCategoryTabs() {
  const tabs = document.getElementById("category-tabs");
  if (wishlistMode) {
    tabs.style.display = "none";
    return;
  }
  const list = categoryListForGender(appliedFilters.gender);
  const items = [{ value: "", label: "All" }, ...list.map((c) => ({ value: c, label: _titleCase(c) }))];
  tabs.innerHTML = items
    .map((i) => `<button type="button" class="category-tab" data-category="${i.value}">${i.label}</button>`)
    .join("");
  tabs.querySelectorAll(".category-tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      appliedFilters.category = tab.dataset.category;
      draft.category = tab.dataset.category;
      syncCategoryTabs();
      document.getElementById("category-select").value = appliedFilters.category;
      loadCatalog();
    });
  });
  syncCategoryTabs();
}

function syncCategoryTabs() {
  document.querySelectorAll(".category-tab").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.category === appliedFilters.category);
  });
}

renderCategoryTabs();

// -------------------------------------------------------------- gender tabs

function syncGenderTabs() {
  document.querySelectorAll(".gender-tab").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.gender === appliedFilters.gender);
  });
}

document.querySelectorAll(".gender-tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    appliedFilters.gender = tab.dataset.gender;
    draft.gender = tab.dataset.gender;
    syncGenderTabs();
    document.querySelectorAll('input[name="gender"]').forEach((r) => (r.checked = r.value === appliedFilters.gender));
    applyGenderChange();
    loadCatalog();
  });
});
syncGenderTabs();

// ------------------------------------------------------------ static lists

(function populateStaticFilters() {
  const categorySelect = document.getElementById("category-select");
  CATEGORIES.forEach((c) => {
    const opt = document.createElement("option");
    opt.value = c;
    opt.textContent = _titleCase(c);
    categorySelect.appendChild(opt);
  });

  const fabricSelect = document.getElementById("fabric-select");
  FABRICS.forEach((f) => {
    const opt = document.createElement("option");
    opt.value = f;
    opt.textContent = _titleCase(f);
    fabricSelect.appendChild(opt);
  });

  const sizePills = document.getElementById("size-pills");
  SIZES.forEach((s) => {
    const pill = document.createElement("button");
    pill.type = "button";
    pill.className = "pill";
    pill.textContent = s;
    pill.dataset.size = s;
    pill.addEventListener("click", () => {
      draft.size = draft.size === s ? "" : s;
      syncDraftUI();
    });
    sizePills.appendChild(pill);
  });

  const colorGrid = document.getElementById("color-filter-grid");
  COLORS.forEach((c) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "color-option";
    btn.dataset.color = c;
    btn.innerHTML = `<span class="swatch-box" style="--swatch-color:${COLOR_HEX[c] || "#ccc"}"></span>${_titleCase(c)}`;
    btn.addEventListener("click", () => {
      draft.color = draft.color === c ? "" : c;
      syncDraftUI();
    });
    colorGrid.appendChild(btn);
  });
})();

document.querySelectorAll('input[name="gender"]').forEach((radio) => {
  radio.addEventListener("change", () => {
    draft.gender = radio.value;
  });
});
document.getElementById("category-select").addEventListener("change", (e) => {
  draft.category = e.target.value;
});
document.getElementById("fabric-select").addEventListener("change", (e) => {
  draft.fabric = e.target.value;
});
document.getElementById("price-min").addEventListener("input", (e) => {
  draft.price_min = e.target.value;
});
document.getElementById("price-max").addEventListener("input", (e) => {
  draft.price_max = e.target.value;
});
document.querySelectorAll('input[name="sort"]').forEach((radio) => {
  radio.addEventListener("change", () => {
    draft.sort = radio.value;
  });
});

function syncDraftUI() {
  document.querySelectorAll('input[name="gender"]').forEach((r) => (r.checked = r.value === draft.gender));
  document.getElementById("category-select").value = draft.category;
  document.getElementById("fabric-select").value = draft.fabric;
  document.getElementById("price-min").value = draft.price_min;
  document.getElementById("price-max").value = draft.price_max;
  document.querySelectorAll('input[name="sort"]').forEach((r) => (r.checked = r.value === draft.sort));
  document.querySelectorAll("#size-pills .pill").forEach((p) => {
    p.classList.toggle("active", p.dataset.size === draft.size);
  });
  document.querySelectorAll("#color-filter-grid .color-option").forEach((c) => {
    c.classList.toggle("selected", c.dataset.color === draft.color);
  });
}

// ------------------------------------------------------------------ accordion

document.querySelectorAll(".filters-drawer .accordion-header").forEach((header) => {
  header.addEventListener("click", () => {
    header.closest(".accordion-section").classList.toggle("open");
  });
});

// -------------------------------------------------------------------- drawer

const drawer = document.getElementById("filters-drawer");
const overlay = document.getElementById("drawer-overlay");

function applyGenderChange() {
  const list = categoryListForGender(appliedFilters.gender);
  if (appliedFilters.category && !list.includes(appliedFilters.category)) {
    appliedFilters.category = "";
    draft.category = "";
    document.getElementById("category-select").value = "";
  }
  renderCategoryTabs();
}

function openDrawer() {
  draft = { ...appliedFilters };
  syncDraftUI();
  drawer.classList.add("open");
  overlay.classList.add("open");
}
function closeDrawer() {
  drawer.classList.remove("open");
  overlay.classList.remove("open");
}
document.getElementById("filters-toggle").addEventListener("click", openDrawer);
document.getElementById("filters-close").addEventListener("click", closeDrawer);
overlay.addEventListener("click", closeDrawer);

document.getElementById("show-items-btn").addEventListener("click", () => {
  Object.assign(appliedFilters, draft);
  closeDrawer();
  applyGenderChange();
  loadCatalog();
});
document.getElementById("clear-filters-btn").addEventListener("click", () => {
  draft = defaultFilters();
  Object.assign(appliedFilters, draft);
  syncDraftUI();
  applyGenderChange();
  loadCatalog();
});

if (appliedFilters.category || appliedFilters.gender) {
  syncDraftUI();
}
if (wishlistMode) {
  document.getElementById("filters-toggle").style.display = "none";
  document.getElementById("catalog-gender-nav").style.display = "none";
}

// -------------------------------------------------------------- grid density

const DENSITY_KEY = "fitml_grid_density";
let gridDensity = localStorage.getItem(DENSITY_KEY) || "default";

function applyDensity() {
  const grid = document.getElementById("catalog-grid");
  grid.classList.remove("density-1", "density-2", "density-4");
  if (gridDensity !== "default") grid.classList.add(`density-${gridDensity}`);
  document.querySelectorAll(".density-btn").forEach((b) => {
    b.classList.toggle("active", b.dataset.density === gridDensity);
  });
}

document.querySelectorAll(".density-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    gridDensity = btn.dataset.density;
    localStorage.setItem(DENSITY_KEY, gridDensity);
    applyDensity();
  });
});
applyDensity();

// ------------------------------------------------------------------- fetch

let currentItems = [];
let itemsById = {};
const recommendationCache = {};

async function loadCatalog() {
  const grid = document.getElementById("catalog-grid");
  const countEl = document.getElementById("item-count");
  grid.innerHTML = '<div class="loading">Loading catalog...</div>';
  try {
    const res = await apiGet("/catalog", wishlistMode ? {} : {
      gender: appliedFilters.gender,
      category: appliedFilters.category,
      color: appliedFilters.color,
      size: appliedFilters.size,
      fabric: appliedFilters.fabric,
      price_min: appliedFilters.price_min,
      price_max: appliedFilters.price_max,
    });
    let items = res.items || [];

    if (wishlistMode) {
      const saved = getWishlist();
      items = items.filter((i) => saved.includes(String(i.item_id)));
    }
    if (searchQuery) {
      items = items.filter((i) =>
        i.product_name.toLowerCase().includes(searchQuery) ||
        i.category.toLowerCase().includes(searchQuery) ||
        i.fabric.toLowerCase().includes(searchQuery) ||
        i.brand.toLowerCase().includes(searchQuery)
      );
    }

    currentItems = items;
    itemsById = Object.fromEntries(items.map((i) => [String(i.item_id), i]));
    if (wishlistMode) {
      countEl.textContent = `${items.length} saved item${items.length === 1 ? "" : "s"}`;
    } else if (searchQuery) {
      countEl.textContent = `${items.length} result${items.length === 1 ? "" : "s"} for "${params.get("q")}"`;
    } else {
      countEl.textContent = `${items.length} item${items.length === 1 ? "" : "s"}`;
    }
    renderGrid();
  } catch (err) {
    grid.innerHTML = `<div class="empty-state">Couldn't load the catalog: ${err.message}</div>`;
    countEl.textContent = "";
  }
}

function renderGrid() {
  const grid = document.getElementById("catalog-grid");
  let items = [...currentItems];

  if (appliedFilters.sort === "price_asc") items.sort((a, b) => a.price_php - b.price_php);
  else if (appliedFilters.sort === "price_desc") items.sort((a, b) => b.price_php - a.price_php);
  else if (appliedFilters.sort === "name_asc") items.sort((a, b) => a.product_name.localeCompare(b.product_name));

  if (items.length === 0) {
    const message = wishlistMode
      ? "Your wishlist is empty. Browse the catalog and tap the heart on any item to save it here."
      : "No items match your filters. Try clearing some.";
    grid.innerHTML = `<div class="empty-state">${message}</div>`;
    return;
  }

  const hasProfile = !!(typeof getProfile === "function" && getProfile());

  grid.innerHTML = items
    .map((item) => {
      const photoUrl = imageUrl(item.photo_url);
      const cutoutUrl = imageUrl(item.cutout_url);
      const colors = [item.color, ...(item.variant_colors || [])];
      const swatches = colors
        .map(
          (c, i) => `<button type="button" class="swatch-dot${i === 0 ? " active" : ""}" data-color="${c}" style="--swatch-color:${COLOR_HEX[c.toLowerCase()] || "#ccc"}" aria-label="${_titleCase(c)}" title="${_titleCase(c)}"></button>`
        )
        .join("");
      const wished = isWishlisted(item.item_id);
      const sizeChartHeader = _sizeChartHeader(item);

      return `
    <div class="card item-card" data-item-id="${item.item_id}">
      <div class="thumb-wrap">
        <img class="thumb-silhouette" src="${_silhouetteFor(item.category)}" alt="" aria-hidden="true">
        <img class="thumb" src="${photoUrl}" data-photo="${photoUrl}" data-cutout="${cutoutUrl}" data-native="${item.color}" alt="${item.product_name}" loading="lazy">
        <button type="button" class="wishlist-heart${wished ? " active" : ""}" data-item-id="${item.item_id}" aria-label="Save to wishlist">${wished ? "&#9829;" : "&#9825;"}</button>
      </div>
      <div class="item-body">
        <span class="brand-name">${item.brand}</span>
        <div class="title-row">
          <span class="product-name">${item.product_name}</span>
          <button type="button" class="expand-btn" aria-label="More details" aria-expanded="false">+</button>
        </div>
        <span class="price">&#8369;${item.price_php}</span>
        <div class="swatch-row">${swatches}</div>
        <a class="btn btn-primary try-on-btn" href="item.html?item_id=${encodeURIComponent(item.item_id)}&color=${encodeURIComponent(item.color)}">Try on</a>
      </div>
      <div class="details-panel" hidden>
        <div class="details-fixed">
          <h4>Sizes</h4>
          <div class="size-chips">${item.size_range.map((s) => `<span class="chip">${s}</span>`).join("")}</div>
        </div>

        <div class="accordion-section" data-subsection="measurements">
          <button type="button" class="accordion-header">Size measurements <span class="chev">&#9662;</span></button>
          <div class="accordion-panel">
            <div class="accordion-panel-inner">
              <table class="size-chart-table"><thead>${sizeChartHeader}</thead><tbody>${_sizeChartRows(item)}</tbody></table>
            </div>
          </div>
        </div>

        <div class="accordion-section" data-subsection="description">
          <button type="button" class="accordion-header">Description <span class="chev">&#9662;</span></button>
          <div class="accordion-panel">
            <div class="accordion-panel-inner"><p>${_productDescription(item)}</p></div>
          </div>
        </div>

        <div class="accordion-section" data-subsection="care">
          <button type="button" class="accordion-header">Composition &amp; Care <span class="chev">&#9662;</span></button>
          <div class="accordion-panel">
            <div class="accordion-panel-inner">
              <p>${_fabricDescription(item.fabric)}</p>
              <p>${item.care}</p>
            </div>
          </div>
        </div>

        ${hasProfile ? `
        <div class="accordion-section" data-subsection="recommendation">
          <button type="button" class="accordion-header">Size recommendation <span class="chev">&#9662;</span></button>
          <div class="accordion-panel">
            <div class="accordion-panel-inner recommendation-body">
              <p class="text-muted">Open this to check your size for this item.</p>
            </div>
          </div>
        </div>` : ""}
      </div>
    </div>`;
    })
    .join("");
}

async function loadRecommendation(card, itemId) {
  const body = card.querySelector(".recommendation-body");
  const profile = getProfile();
  if (!profile || !profile.session_id) return;
  body.innerHTML = '<p class="text-muted">Checking your size…</p>';
  try {
    const rec = await apiPostJSON("/recommend-size", { session_id: profile.session_id, item_id: itemId });
    const tip = rec.state === "amber"
      ? "<p class=\"hint\">💡 Sizing tip: this runs a little snug — consider sizing up.</p>"
      : "";
    body.innerHTML = `
      <p><strong>Recommended size: ${rec.recommended_size}</strong> (${rec.confidence}% confidence)</p>
      ${tip}
      <p class="text-muted">Try this item on for personalized fit advice.</p>
    `;
  } catch (err) {
    body.innerHTML = `<p class="error-text">Couldn't get a recommendation: ${err.message}</p>`;
  }
}

document.getElementById("catalog-grid").addEventListener("click", (e) => {
  const heart = e.target.closest(".wishlist-heart");
  if (heart) {
    const nowSaved = toggleWishlist(heart.dataset.itemId);
    heart.classList.toggle("active", nowSaved);
    heart.innerHTML = nowSaved ? "&#9829;" : "&#9825;";
    if (wishlistMode && !nowSaved) {
      loadCatalog();
    }
    return;
  }

  const expandBtn = e.target.closest(".expand-btn");
  if (expandBtn) {
    const card = expandBtn.closest(".item-card");
    const panel = card.querySelector(".details-panel");
    const isOpen = !panel.hidden;
    panel.hidden = isOpen;
    expandBtn.textContent = isOpen ? "+" : "−";
    expandBtn.setAttribute("aria-expanded", String(!isOpen));
    return;
  }

  const subHeader = e.target.closest(".details-panel .accordion-header");
  if (subHeader) {
    const section = subHeader.closest(".accordion-section");
    const wasOpen = section.classList.contains("open");
    section.classList.toggle("open");
    if (!wasOpen && section.dataset.subsection === "recommendation") {
      const card = subHeader.closest(".item-card");
      loadRecommendation(card, card.dataset.itemId);
    }
    return;
  }

  const dot = e.target.closest(".swatch-dot");
  if (!dot) return;
  e.preventDefault();
  const card = dot.closest(".item-card");
  const wrap = card.querySelector(".thumb-wrap");
  const img = card.querySelector(".thumb");
  const color = dot.dataset.color;
  const native = img.dataset.native;
  const isNative = color === native;
  img.src = isNative ? img.dataset.photo : _variantCutoutUrl(img.dataset.cutout, color);
  wrap.classList.toggle("showing-cutout", !isNative);
  card.querySelectorAll(".swatch-dot").forEach((d) => d.classList.toggle("active", d === dot));
  const tryOnLink = card.querySelector("a.btn");
  const itemId = card.dataset.itemId;
  tryOnLink.href = `item.html?item_id=${encodeURIComponent(itemId)}&color=${encodeURIComponent(color)}`;
});

loadCatalog();

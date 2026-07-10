/* FitML — Item detail / try-on result page. Loads the item by item_id
   (matching catalog.js's own "fetch the whole list, filter client-side"
   approach — there's no GET /item/<id> endpoint), lets the shopper pick a
   color/size, fetches the size recommendation, runs the pose-anchored
   try-on composite, then fetches the two-paragraph Claude advice text. */

const params = new URLSearchParams(window.location.search);
const itemId = params.get("item_id");
const urlColor = params.get("color");

const root = document.getElementById("item-detail");

let item = null;
let profile = null;
let selectedColor = null;
let selectedSize = null;
let recommendation = null;
let tryonResult = null; // { image_url, tryon_id }
let tryonError = null;
let adviceText = null;
let adviceLoading = false;
let tryonLoading = false;
let recommendationError = null;

function money(n) {
  return `₱${n}`;
}

function render() {
  if (!item) {
    root.innerHTML = '<div class="empty-state">Item not found. <a href="catalog.html">Back to catalog</a></div>';
    return;
  }

  const colors = [item.color, ...(item.variant_colors || [])];
  const isNative = selectedColor.toLowerCase() === item.color.toLowerCase();
  const heroPhoto = tryonResult ? imageUrl(tryonResult.image_url)
    : isNative ? imageUrl(item.photo_url)
    : _variantCutoutUrl(imageUrl(item.cutout_url), selectedColor);

  const swatches = colors.map((c) => `
    <button type="button" class="swatch-dot${c.toLowerCase() === selectedColor.toLowerCase() ? " active" : ""}"
      data-color="${c}" style="--swatch-color:${COLOR_HEX[c.toLowerCase()] || "#ccc"}"
      aria-label="${_titleCase(c)}" title="${_titleCase(c)}"></button>
  `).join("");

  const sizePills = item.size_range.map((s) => {
    const isRec = recommendation && recommendation.recommended_size === s;
    const isSelected = selectedSize === s;
    return `<button type="button" class="pill size-pill${isSelected ? " active" : ""}${isRec ? " recommended" : ""}" data-size="${s}">${s}</button>`;
  }).join("");

  let confidenceBoxHtml = "";
  if (recommendation) {
    if (recommendation.state === "amber") {
      confidenceBoxHtml = `<div class="confidence-box amber">
        💡 <strong>Sizing tip:</strong> this runs a little snug in ${recommendation.recommended_size} —
        we've suggested sizing up for a more comfortable fit. (${recommendation.confidence}% confidence)
      </div>`;
    } else {
      confidenceBoxHtml = `<div class="confidence-box blue">
        ✓ <strong>Recommended size: ${recommendation.recommended_size}</strong> — ${recommendation.confidence}% confidence
      </div>`;
    }
  }

  let noProfileHtml = "";
  let tryOnActionHtml = "";
  if (!profile) {
    noProfileHtml = `<div class="no-profile-note">
      Set up your profile to get a personalized size recommendation and try this item on.
      <a href="profile.html">Set up profile &rarr;</a>
    </div>`;
  } else {
    tryOnActionHtml = `<div class="item-actions">
      <button type="button" class="btn btn-primary" id="try-on-btn" ${tryonLoading ? "disabled" : ""}>
        ${tryonLoading ? "Compositing…" : "Try on"}
      </button>
    </div>`;
  }

  let tryonErrorHtml = tryonError ? `<div class="confidence-box amber">${tryonError}</div>` : "";

  let adviceHtml = "";
  if (tryonResult) {
    if (adviceLoading) {
      adviceHtml = `<div class="advice-block"><h3>Fit advice</h3><p class="text-muted">Getting personalized advice…</p></div>`;
    } else if (adviceText) {
      const paragraphs = adviceText.split(/\n+/).filter(Boolean).map((p) => `<p>${p}</p>`).join("");
      adviceHtml = `<div class="advice-block"><h3>Fit advice</h3>${paragraphs}</div>`;
    } else if (adviceText === "") {
      adviceHtml = `<div class="advice-block"><h3>Fit advice</h3><p class="text-muted">Advice isn't available right now.</p></div>`;
    }
  }

  // Preserve which accordion subsections are open across re-renders
  // (size/color clicks re-render the whole panel).
  const openSections = [...root.querySelectorAll(".accordion-section.open")]
    .map((s) => s.dataset.subsection);

  root.innerHTML = `
    <div class="item-detail-grid">
      <div>
        <div class="item-hero-wrap${!tryonResult && !isNative ? " showing-cutout" : ""}" id="item-hero-wrap">
          ${!tryonResult ? `<img class="item-hero-silhouette" src="${_silhouetteFor(item.category)}" alt="" aria-hidden="true">` : ""}
          <img class="item-hero-img" src="${heroPhoto}" alt="${item.product_name}">
          ${tryonLoading ? '<div class="item-hero-loading">Compositing your try-on…</div>' : ""}
        </div>
        <div class="item-swatch-row">${swatches}</div>
      </div>
      <div class="item-info">
        <span class="brand-name">${item.brand}</span>
        <h1>${item.product_name}</h1>
        <div class="price">${money(item.price_php)}</div>

        <span class="size-select-label">Size</span>
        <div class="size-pill-row">${sizePills}</div>

        ${confidenceBoxHtml}
        ${tryonErrorHtml}
        ${noProfileHtml}
        ${tryOnActionHtml}
        ${adviceHtml}

        <div class="details-panel" id="item-details-panel">
          <div class="details-fixed">
            <h4>Sizes</h4>
            <div class="size-chips">${item.size_range.map((s) =>
              `<button type="button" class="chip size-chip${selectedSize === s ? " active" : ""}" data-size="${s}">${s}</button>`).join("")}</div>
          </div>

          <div class="accordion-section" data-subsection="measurements">
            <button type="button" class="accordion-header">Size measurements <span class="chev">&#9662;</span></button>
            <div class="accordion-panel">
              <div class="accordion-panel-inner">
                <table class="size-chart-table"><thead>${_sizeChartHeader(item)}</thead><tbody>${_sizeChartRows(item)}</tbody></table>
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

          ${profile ? `
          <div class="accordion-section" data-subsection="recommendation">
            <button type="button" class="accordion-header">Size recommendation <span class="chev">&#9662;</span></button>
            <div class="accordion-panel">
              <div class="accordion-panel-inner recommendation-body">
                ${recommendation
                  ? `<p><strong>Recommended size: ${recommendation.recommended_size}</strong> (${recommendation.confidence}% confidence)</p>
                     ${recommendation.state === "amber" ? '<p class="hint">💡 Sizing tip: this runs a little snug — consider sizing up.</p>' : ""}`
                  : recommendationError
                    ? `<p class="error-text">${recommendationError}</p>`
                    : `<p class="text-muted">Checking your size…</p>`}
              </div>
            </div>
          </div>` : ""}
        </div>
      </div>
    </div>
  `;

  openSections.forEach((name) => {
    const section = root.querySelector(`.accordion-section[data-subsection="${name}"]`);
    if (section) section.classList.add("open");
  });

  // Align the hero silhouette to the garment's drawn rect once the cutout
  // has dimensions (see _alignSilhouette in catalog-common.js).
  if (!tryonResult && !isNative) {
    const heroImg = root.querySelector(".item-hero-img");
    const heroSil = root.querySelector(".item-hero-silhouette");
    const align = () => _alignSilhouette(heroImg, heroSil, item.category);
    if (heroImg.complete && heroImg.naturalWidth) align();
    else heroImg.addEventListener("load", align, { once: true });
  }

  wireEvents();
}

function wireEvents() {
  const heroWrap = document.getElementById("item-hero-wrap");
  heroWrap && heroWrap.querySelectorAll(".swatch-dot");

  document.querySelectorAll(".swatch-dot").forEach((dot) => {
    dot.addEventListener("click", () => {
      const color = dot.dataset.color;
      const isNative = color.toLowerCase() === item.color.toLowerCase();
      const apply = () => {
        selectedColor = color;
        tryonResult = null;
        tryonError = null;
        adviceText = null;
        render();
      };
      if (isNative) {
        apply();
      } else {
        // Preload the variant cutout so the hero never shows the grey
        // silhouette alone while the PNG streams in.
        _swapWhenLoaded(new Image(), _variantCutoutUrl(imageUrl(item.cutout_url), color), apply);
      }
    });
  });

  // Main size-pill row AND the "Sizes" chips in the details panel both
  // select the size (the chips read as pills to shoppers, so they must act
  // like them — inert spans tested as a broken control).
  document.querySelectorAll(".size-pill, .size-chip").forEach((pill) => {
    pill.addEventListener("click", () => {
      selectedSize = pill.dataset.size;
      tryonResult = null;
      tryonError = null;
      adviceText = null;
      render();
    });
  });

  const tryOnBtn = document.getElementById("try-on-btn");
  if (tryOnBtn) tryOnBtn.addEventListener("click", runTryOn);

  document.querySelectorAll("#item-details-panel .accordion-header").forEach((header) => {
    header.addEventListener("click", () => {
      header.closest(".accordion-section").classList.toggle("open");
    });
  });
}

async function loadRecommendation() {
  if (!profile) return;
  try {
    recommendation = await apiPostJSON("/recommend-size", { session_id: profile.session_id, item_id: itemId });
    if (!selectedSize) selectedSize = recommendation.recommended_size;
  } catch (err) {
    // Non-fatal: size pills still work without a recommendation.
    recommendationError = err.message || "Couldn't get a recommendation.";
  }
  render();
}

async function runTryOn() {
  if (!profile || !selectedSize) return;
  tryonLoading = true;
  tryonError = null;
  render();
  try {
    const res = await apiPostJSON("/try-on", {
      session_id: profile.session_id,
      item_id: itemId,
      size: selectedSize,
      color: selectedColor,
    });
    tryonResult = { image_url: res.image_url, tryon_id: res.tryon_id };
    recommendation = { ...recommendation, recommended_size: res.recommended_size, confidence: res.confidence, state: res.state };
    tryonLoading = false;
    render();

    addHistoryEntry({
      tryon_id: res.tryon_id,
      item_id: itemId,
      product_name: item.product_name,
      brand: item.brand,
      image_url: res.image_url,
      // Permanent fallback thumbnail: composites are deleted server-side
      // after 24h (privacy) and on redeploys, but history entries persist
      // in localStorage — the card falls back to the catalog photo.
      photo_url: item.photo_url,
      size: res.size,
      recommended_size: res.recommended_size,
      confidence: res.confidence,
      state: res.state,
      color: selectedColor,
    });

    loadAdvice(res.tryon_id);
  } catch (err) {
    tryonLoading = false;
    tryonError = err.message || "Couldn't generate the try-on preview. Please try again.";
    render();
  }
}

async function loadAdvice(tryonId) {
  adviceLoading = true;
  render();
  try {
    const res = await apiPostJSON("/advice", { session_id: profile.session_id, tryon_id: tryonId });
    adviceText = res.advice;
    updateHistoryEntry(tryonId, { advice: res.advice });
  } catch (err) {
    adviceText = "";
  } finally {
    adviceLoading = false;
    render();
  }
}

window.addEventListener("resize", () => {
  const heroImg = document.querySelector(".item-hero-img");
  const heroSil = document.querySelector(".item-hero-silhouette");
  if (item && heroImg && heroSil && document.querySelector(".item-hero-wrap.showing-cutout")) {
    _alignSilhouette(heroImg, heroSil, item.category);
  }
});

async function init() {
  if (!itemId) {
    root.innerHTML = '<div class="empty-state">No item selected. <a href="catalog.html">Back to catalog</a></div>';
    return;
  }
  profile = requireProfile();
  try {
    const res = await apiGet("/catalog", {});
    item = (res.items || []).find((i) => String(i.item_id) === String(itemId)) || null;
  } catch (err) {
    root.innerHTML = `<div class="empty-state">Couldn't load this item: ${err.message}</div>`;
    return;
  }
  if (!item) {
    render();
    return;
  }
  selectedColor = urlColor && [item.color, ...(item.variant_colors || [])].some((c) => c.toLowerCase() === urlColor.toLowerCase())
    ? urlColor : item.color;
  selectedSize = null;
  render();
  if (profile) loadRecommendation();
}

init();

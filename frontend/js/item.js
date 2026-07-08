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
        <p class="meta-line">${_fabricDescription(item.fabric)}</p>
        <p class="meta-line">${item.care}</p>
        <p class="description">${_productDescription(item)}</p>

        <span class="size-select-label">Size</span>
        <div class="size-pill-row">${sizePills}</div>

        ${confidenceBoxHtml}
        ${tryonErrorHtml}
        ${noProfileHtml}
        ${tryOnActionHtml}
        ${adviceHtml}
      </div>
    </div>
  `;

  wireEvents();
}

function wireEvents() {
  const heroWrap = document.getElementById("item-hero-wrap");
  heroWrap && heroWrap.querySelectorAll(".swatch-dot");

  document.querySelectorAll(".swatch-dot").forEach((dot) => {
    dot.addEventListener("click", () => {
      selectedColor = dot.dataset.color;
      tryonResult = null;
      tryonError = null;
      adviceText = null;
      render();
    });
  });

  document.querySelectorAll(".size-pill").forEach((pill) => {
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
}

async function loadRecommendation() {
  if (!profile) return;
  try {
    recommendation = await apiPostJSON("/recommend-size", { session_id: profile.session_id, item_id: itemId });
    if (!selectedSize) selectedSize = recommendation.recommended_size;
    render();
  } catch (err) {
    // Non-fatal: size pills still work without a recommendation.
  }
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

/* FitML — History page. No backend history-list endpoint (see state.js);
   reconstructed from try-ons recorded to localStorage as they complete on
   the item detail page. */

function formatDate(iso) {
  const d = new Date(iso);
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}

function render() {
  const grid = document.getElementById("history-grid");
  const history = getHistory();

  if (history.length === 0) {
    grid.innerHTML = `<div class="empty-state">You haven't tried anything on yet. <a href="catalog.html">Browse the catalog</a> to get started.</div>`;
    return;
  }

  grid.innerHTML = history.map((entry) => {
    const badgeLabel = entry.state === "amber" ? "Sizing tip" : "Good fit";
    return `
    <div class="card history-card">
      <img src="${imageUrl(entry.image_url)}" alt="${entry.product_name}">
      <div class="history-body">
        <div class="history-brand">${entry.brand || ""}</div>
        <div class="history-name">${entry.product_name}</div>
        <div class="history-meta">Tried: ${entry.size} · Recommended: ${entry.recommended_size} (${entry.confidence}%)</div>
        <span class="history-badge ${entry.state === "amber" ? "amber" : "blue"}">${badgeLabel}</span>
        <div class="history-date">${formatDate(entry.created_at)}</div>
        <div class="history-actions">
          <a href="item.html?item_id=${encodeURIComponent(entry.item_id)}&color=${encodeURIComponent(entry.color || "")}">Try again &rarr;</a>
        </div>
      </div>
    </div>`;
  }).join("");
}

render();

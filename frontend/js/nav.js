/* FitML — shared header/footer, injected into placeholders so markup isn't
   duplicated across the static pages. Script tag must sit after the
   #site-header / #site-footer placeholder divs, and before page scripts
   that assume the DOM (nav links, profile badge) already exists.

   Header layout: brand left, page nav + search + hamburger right. Women/Men
   are catalog-page-specific sub-navigation (rendered by catalog.js, above
   the category tab bar), not part of the global header. */

const NAV_LINKS = [
  { href: "index.html", label: "Home", page: "landing" },
  { href: "catalog.html", label: "Catalog", page: "catalog" },
  { href: "history.html", label: "History", page: "history" },
  { href: "catalog.html?wishlist=1", label: "Wishlist", page: "wishlist" },
  { type: "search" },
  { href: "profile.html", label: "Profile", page: "profile" },
];

function _navItemHtml(item, activePage, profile) {
  if (item.type === "search") {
    return `<span class="nav-search-wrap" id="nav-search-wrap">
      <button type="button" class="nav-search-toggle" id="search-toggle">Search</button>
      <form class="nav-search-inline" id="search-form">
        <input type="search" id="search-input" placeholder="Search" autocomplete="off">
        <button type="button" class="search-close" id="search-close" aria-label="Close search">&times;</button>
      </form>
    </span>`;
  }
  const activeAttr = item.page === activePage ? ' class="active"' : "";
  const label = item.page === "profile" && profile && profile.name ? `Hi, ${profile.name}` : item.label;
  return `<a href="${item.href}" data-page="${item.page}"${activeAttr}>${label}</a>`;
}

function renderHeader(activePage) {
  const el = document.getElementById("site-header");
  if (!el) return;

  const profile = typeof getProfile === "function" ? getProfile() : null;
  const navHtml = NAV_LINKS.map((item) => _navItemHtml(item, activePage, profile)).join("");
  const mobileNavHtml = NAV_LINKS.filter((item) => item.type !== "search")
    .map((item) => _navItemHtml(item, activePage, profile)).join("");

  el.innerHTML = `
    <div class="container header-inner">
      <a href="index.html" class="brand">FitML</a>
      <div class="header-right">
        <nav class="main-nav">${navHtml}</nav>
        <button class="nav-toggle" aria-label="Menu">&#9776;</button>
      </div>
    </div>
    <nav class="mobile-nav">${mobileNavHtml}</nav>
  `;

  const toggle = el.querySelector(".nav-toggle");
  const mobileNav = el.querySelector(".mobile-nav");
  if (toggle && mobileNav) {
    toggle.addEventListener("click", () => mobileNav.classList.toggle("open"));
  }

  const searchWrap = el.querySelector("#nav-search-wrap");
  const searchToggle = el.querySelector("#search-toggle");
  const searchClose = el.querySelector("#search-close");
  const searchForm = el.querySelector("#search-form");
  const searchInput = el.querySelector("#search-input");
  if (searchWrap && searchToggle) {
    searchToggle.addEventListener("click", () => {
      searchWrap.classList.add("open");
      searchInput.focus();
    });
    searchClose.addEventListener("click", () => searchWrap.classList.remove("open"));
    searchForm.addEventListener("submit", (e) => {
      e.preventDefault();
      const q = searchInput.value.trim();
      window.location.href = q ? `catalog.html?q=${encodeURIComponent(q)}` : "catalog.html";
    });
  }
}

function renderFooter() {
  const el = document.getElementById("site-footer");
  if (!el) return;
  el.innerHTML = `
    <div class="container footer-grid">
      <div class="footer-col">
        <h4>About</h4>
        <a href="about.html">How it works</a>
        <a href="about.html#fairness">Fairness &amp; bias audit</a>
      </div>
      <div class="footer-col">
        <h4>Support</h4>
        <a href="about.html#privacy">Privacy policy</a>
      </div>
      <div class="footer-col">
        <h4>FitML</h4>
        <p>A virtual fitting room with size recommendations and tailored advice.</p>
      </div>
      <div class="footer-col">
        <h4>Contact</h4>
        <p>Debie Galleros</p>
        <p>0956 926 8328</p>
        <a href="mailto:debiegalleros@gmail.com">debiegalleros@gmail.com</a>
      </div>
    </div>
    <div class="container footer-bottom">
      &copy; 2026 FitML — Capstone Project, Asian Institute of Management
    </div>
  `;
}

(function init() {
  const activePage = document.body.getAttribute("data-page");
  renderHeader(activePage);
  renderFooter();
})();

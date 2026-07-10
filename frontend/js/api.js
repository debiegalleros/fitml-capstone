/* FitML — API client. Local dev backend runs on port 5001 (see
   backend/app.py); the deployed site talks to the Render-hosted backend. */

const API_BASE = ["localhost", "127.0.0.1"].includes(window.location.hostname)
  ? "http://localhost:5001"
  : "https://fitml-capstone.onrender.com";

class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

async function _handle(res) {
  let data = {};
  try {
    data = await res.json();
  } catch (e) {
    data = {};
  }
  if (!res.ok) {
    throw new ApiError(data.error || res.statusText || "Request failed", res.status);
  }
  return data;
}

async function apiGet(path, params) {
  const url = new URL(API_BASE + path);
  if (params) {
    Object.entries(params).forEach(([k, v]) => {
      if (v !== undefined && v !== null && v !== "") url.searchParams.set(k, v);
    });
  }
  const res = await fetch(url);
  return _handle(res);
}

async function apiPostJSON(path, body) {
  const res = await fetch(API_BASE + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  return _handle(res);
}

async function apiPostForm(path, formData) {
  const res = await fetch(API_BASE + path, { method: "POST", body: formData });
  return _handle(res);
}

function imageUrl(pathOrUrl) {
  if (!pathOrUrl) return "";
  if (pathOrUrl.startsWith("http")) return pathOrUrl;
  return API_BASE + pathOrUrl;
}

/* FitML — client-side session/profile/history storage.
   No auth: the upload-profile session_id (a random UUID) is the access
   token, mirroring the backend's own session-scoped image serving. There
   is no backend history-list endpoint, so History is reconstructed from
   try-ons recorded here as they complete. */

const PROFILE_KEY = "fitml_profile";
const HISTORY_KEY = "fitml_history";

function getProfile() {
  try {
    return JSON.parse(localStorage.getItem(PROFILE_KEY) || "null");
  } catch (e) {
    return null;
  }
}

function setProfile(profile) {
  localStorage.setItem(PROFILE_KEY, JSON.stringify(profile));
}

function clearProfile() {
  localStorage.removeItem(PROFILE_KEY);
}

function requireProfile() {
  const p = getProfile();
  if (!p || !p.session_id) return null;
  return p;
}

function getHistory() {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
  } catch (e) {
    return [];
  }
}

function addHistoryEntry(entry) {
  const history = getHistory();
  history.unshift({ ...entry, created_at: new Date().toISOString() });
  localStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(0, 100)));
}

function updateHistoryEntry(tryonId, patch) {
  const history = getHistory();
  const idx = history.findIndex((h) => h.tryon_id === tryonId);
  if (idx === -1) return;
  history[idx] = { ...history[idx], ...patch };
  localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
}

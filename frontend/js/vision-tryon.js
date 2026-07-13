/* FitML — generative try-on client. Calls the combined /api/tryon pipeline
   with an automatic fallback chain:

     primary engine → other engine → window.legacyTryOn (2D compositor)

   The 10–41s generation gets a status line via the onStatus callback (the
   item page paints it over the hero image). Engine names are shown in the
   status line only on localhost — shoppers never see them. A 422 (photo
   coverage: bottoms need a full-body photo) is NOT retried on another
   engine — the friendly message passes straight through. */

const VISION_TRYON = {
  primary: "idm-vton", // mirrors the backend's TRYON_ENGINE default
  fallback: "sdxl",
  dev: ["localhost", "127.0.0.1"].includes(window.location.hostname),
};

function _tryonStatusText(phase, engine) {
  const base = {
    analyzing: "Analyzing the garment on your photo…",
    generating: "Generating your try-on… this can take up to a minute.",
    fallback: "Taking a little longer than usual — trying another way…",
    legacy: "Using quick preview mode…",
  }[phase];
  return VISION_TRYON.dev ? `${base} [${engine || "legacy"}]` : base;
}

/* Resolves to the same shape as the legacy /try-on response
   ({ tryon_id, image_url, size, recommended_size, confidence, state })
   plus an `engine` field. Throws ApiError when every path fails. */
async function visionTryOn({ sessionId, itemId, size, colorVariant, seed }, onStatus) {
  const notify = typeof onStatus === "function" ? onStatus : () => {};
  const engines = [VISION_TRYON.primary, VISION_TRYON.fallback];
  let lastError = null;

  for (let i = 0; i < engines.length; i++) {
    const engine = engines[i];
    notify(_tryonStatusText(i === 0 ? "analyzing" : "fallback", engine));
    const longWait = setTimeout(
      () => notify(_tryonStatusText("generating", engine)), 4000);
    try {
      const res = await apiPostJSON("/api/tryon", {
        session_id: sessionId,
        item_id: itemId,
        size: size,
        color_variant: colorVariant,
        engine: engine,
        ...(seed !== undefined ? { seed } : {}),
      });
      clearTimeout(longWait);
      return res;
    } catch (err) {
      clearTimeout(longWait);
      // Photo-coverage guardrail or bad request: retrying elsewhere can't
      // help, and the 422 wording must reach the shopper unchanged.
      if (err.status === 422 || err.status === 400 || err.status === 404) throw err;
      lastError = err;
    }
  }

  if (typeof window.legacyTryOn === "function") {
    notify(_tryonStatusText("legacy"));
    return await window.legacyTryOn();
  }
  throw lastError || new Error("Try-on is unavailable right now.");
}

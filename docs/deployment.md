# Phase 10 — Deployment Guide

Backend on Render (free tier), frontend on Netlify (free tier). Both free
tiers spin down after ~15 minutes of inactivity — the first request after
idle can take 30-60s to cold-start (mention this if grading live).

## Live URLs

- Backend: `https://fitml-capstone.onrender.com`
- Frontend: `https://<netlify-site-name>.netlify.app`

## Why the catalog images need a build step

`data/catalog/photos` (~12MB) and `data/catalog/garments` (~183MB) are
gitignored by design (see [CLAUDE.md](../CLAUDE.md) — raw/processed image
assets are never committed). Render's free web-service tier has no
persistent disk, so anything not in the git-tracked build has to be fetched
during the build itself. The fix: the two folders are zipped and attached as
a GitHub Release asset (`catalog-assets-v1` on the `fitml-capstone` repo),
and `scripts/fetch_catalog_assets.sh` downloads + unpacks that zip as part
of Render's build command (see `render.yaml`). This keeps the git history
free of binary blobs while still making the files available at runtime.

MediaPipe's pose/face model assets (`backend/mp_models/`, ~6MB) are *not*
part of this zip — `backend/tryon.py` already auto-downloads them from
Google's public model bucket on first use, so they fetch themselves on the
container's first `/upload-profile` call.

## One-time setup already done

- `requirements.txt` frozen to exact versions from the working dev venv,
  plus `gunicorn` added as the production WSGI server (Flask's built-in dev
  server in `backend/app.py`'s `app.run(...)` is for local dev only).
- `frontend/js/api.js` picks the backend URL by hostname: `localhost`/
  `127.0.0.1` keeps hitting `http://localhost:5001`; any other hostname
  (i.e. the deployed Netlify site) targets the Render URL above.
- `render.yaml` — Render Blueprint spec (build/start commands, env var
  list). `netlify.toml` — tells Netlify to publish the `frontend/` folder
  as-is (no build step, it's plain HTML/CSS/JS).

## Manual steps (dashboard logins — can't be automated from the CLI)

### 1. Push to GitHub
Already done as part of this deploy — repo is public at
`https://github.com/debiegalleros/fitml-capstone`.

### 2. Render — backend
1. Sign up / log in at https://dashboard.render.com (GitHub OAuth is easiest).
2. **New +** → **Blueprint** → connect the `fitml-capstone` GitHub repo →
   Render reads `render.yaml` and proposes the `fitml-capstone` web service.
   (If you don't see the Blueprint option, use **New +** → **Web Service**
   instead and set these manually:
   - Root Directory: *(leave blank — repo root)*
   - Build Command: `pip install -r requirements.txt && bash scripts/fetch_catalog_assets.sh`
   - Start Command: `gunicorn --chdir backend --bind 0.0.0.0:$PORT --timeout 120 app:app`
   - Plan: Free)
3. Under **Environment**, add:
   - `ANTHROPIC_API_KEY` = *(your key — copy from `backend/.env`, never commit it)*
   - `PYTHON_VERSION` = `3.12.7`
4. Deploy. First build takes a while (installs mediapipe/xgboost/opencv +
   downloads the 193MB catalog zip). Watch the build log for
   "Catalog assets unpacked: 478 files".
5. Confirm `https://fitml-capstone.onrender.com/health` returns
   `{"status": "ok", "catalog_items": ...}`.

### 3. Netlify — frontend
1. Sign up / log in at https://app.netlify.com.
2. **Add new site** → **Import an existing project** → connect the same
   GitHub repo.
3. Build settings: Base directory blank, Build command blank (static site),
   Publish directory `frontend` (netlify.toml already encodes this, so the
   defaults Netlify detects should be correct — confirm before deploying).
4. Deploy. Netlify assigns a `https://<random-name>.netlify.app` URL (can be
   renamed under Site settings → Site details → Change site name).
5. Update the "Live URLs" section above and the frontend URL wherever the
   report/README references it.

## Verifying the live flow

Test end-to-end on the deployed site before calling this phase done:
Profile setup (photo + measurements) → Catalog (filters load, images
render) → open an item → size recommendation shows → Try on → advice text
loads. If `/advice` 503s, the `ANTHROPIC_API_KEY` env var is missing on
Render — check step 2.3 above.

## Security / privacy notes carried over from local dev

- `ANTHROPIC_API_KEY` lives only in Render's environment variables — never
  in the repo, never in `backend/.env` once that file leaves your machine.
- Uploaded photos still auto-delete after 24h (`backend/cleanup_uploads.py`,
  triggered on each request per `config.UPLOAD_TTL_HOURS`) — Render's
  ephemeral disk means a redeploy also wipes them, which is a stricter
  version of the same privacy guarantee.
- CORS is left permissive (`flask_cors.CORS(app)`, all origins) since this is
  a demo prototype with no auth/cookies to protect; tightening to the exact
  Netlify origin would be the production hardening step.

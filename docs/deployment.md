# Phase 10 — Deployment

Backend on Render (free tier), frontend on Netlify (free tier). Both free
tiers spin down after ~15 minutes of inactivity — the first request after
idle can take 30-60s to cold-start (mention this if grading live).

## Live URLs

- Backend: `https://fit-ml.onrender.com`
- Frontend: `https://fit-ml.netlify.app`
- GitHub repo: `https://github.com/debiegalleros/fitml-capstone`

## Why the catalog images need a build step

`data/catalog/photos` (~12MB) and `data/catalog/garments` (~183MB) are
gitignored by design (see [CLAUDE.md](../CLAUDE.md) — raw/processed image
assets are never committed). Render's free web-service tier has no
persistent disk, so anything not in the git-tracked build has to be fetched
during the build itself. The fix: the two folders are zipped and attached as
a GitHub Release asset (`catalog-assets-v1` on the `fitml-capstone` repo),
and `scripts/fetch_catalog_assets.sh` downloads + unpacks that zip as part
of Render's build command. This keeps the git history free of binary blobs
while still making the files available at runtime.

MediaPipe's pose/face model assets (`backend/mp_models/`, ~6MB) are *not*
part of this zip — `backend/tryon.py` already auto-downloads them from
Google's public model bucket on first use, so they fetch themselves on the
container's first `/upload-profile` call.

## What's in place for deployment

- `requirements.txt` frozen to exact versions from the working dev venv,
  plus `gunicorn` added as the production WSGI server (Flask's built-in dev
  server in `backend/app.py`'s `app.run(...)` is for local dev only).
- `frontend/js/api.js` picks the backend URL by hostname: `localhost`/
  `127.0.0.1` keeps hitting `http://localhost:5001`; any other hostname
  (i.e. the deployed Netlify site) targets `https://fit-ml.onrender.com`.
- `Dockerfile` (repo root) — the backend runs as a Docker web service on
  Render, not Render's native Python runtime. Reason: mediapipe's Tasks
  API (`PoseLandmarker`/`FaceDetector`, used for pose extraction and the
  privacy face-blur) dlopens `libGLESv2.so.2` even on the CPU delegate,
  and Render's native runtime image doesn't have it with no way to add
  system packages. The Dockerfile installs `libgl1 libegl1 libgles2
  libgbm1` (plus opencv's usual `libglib2.0-0 libsm6 libxext6
  libxrender1 libgomp1`) before `pip install`, then runs
  `scripts/fetch_catalog_assets.sh` and starts gunicorn. `render.yaml`
  mirrors this (`runtime: docker`).
- `netlify.toml` — tells Netlify to publish the `frontend/` folder as-is
  (no build step, it's plain HTML/CSS/JS).

## How this was deployed

GitHub, Render, and Netlify were all set up via their CLIs (`gh`, the
`render` CLI, and `netlify-cli`), each authenticated through a device-code
browser login — no credentials were typed into this environment directly.

- **Render** web service `fit-ml` (id `srv-d98hhamrnols73fb71e0`) was
  created via `render services create --runtime docker` (Render
  auto-detects `./Dockerfile`) with `ANTHROPIC_API_KEY` set via the
  Render API (`PUT /v1/services/{id}/env-vars`) — the key was read from
  local `backend/.env` and never printed or committed. Auto-deploy is
  on, so every push to `main` triggers a new Render build.
- **Netlify** site `fit-ml` was created via `netlify sites:create` and
  deployed via `netlify deploy --prod --dir=frontend`. This is a one-shot
  CLI deploy, **not** connected to GitHub for auto-deploy — after frontend
  changes, redeploy manually (see below).
- Render's public API doesn't allow changing a service's URL slug after
  creation (only its display `name`), so the initial `fitml-capstone` slug
  had to be deleted and recreated as `fit-ml` to get the short URL.

## Redeploying after changes

- **Backend**: just `git push` to `main` — Render auto-deploys.
- **Frontend**: run `netlify deploy --prod --dir=frontend` from the repo
  root after any change under `frontend/`.

## Verifying the live flow

Full flow verified directly against the live backend
(`https://fit-ml.onrender.com`) via curl: `/upload-profile` (multipart
photo + measurements) → `/recommend-size` → `/try-on` (composited image
confirmed real: face-blurred, garment overlaid) → `/advice` (real
Claude API call, two-paragraph output with the "Note:" plain-language
visual observation). This is what caught the libGL issue above — the
first Docker deploy still 500'd on `/upload-profile` until `libgles2`/
`libegl1`/`libgbm1` were added.

If `/advice` 503s, the `ANTHROPIC_API_KEY` env var is missing on Render.
If `/upload-profile` 500s with an `OSError` about a missing `.so` file,
a mediapipe system dependency is missing from the Dockerfile.

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

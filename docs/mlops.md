# MLOps — Monitoring, Versioning, Rollback

Step 8 of the phase plan (optional, no dedicated rubric points, but the
capstone rubric mentions MLOps practices generally). This covers what
happens *after* deployment — [deployment.md](deployment.md) covers the
initial setup. Scope note: this is a single-developer capstone prototype,
not a production ML system serving real traffic — the practices below are
sized to that (lightweight, mostly free-tier tooling), not a scaled-down
version of an enterprise MLOps stack.

## Monitoring

**Render's built-in metrics** (Metrics tab on the `fit-ml` service) are the
only monitoring in place: CPU, memory, and request logs, retained on the
free tier for the service's lifetime. No custom APM or alerting service is
wired up — for a solo capstone project with no real users, Render's
dashboard plus manual log inspection is proportionate; a paid
observability stack (Datadog, Sentry, etc.) would be overkill for the
traffic this app will ever see.

**Worked example — the OOM incident.** During the vision-tryon rebuild,
Render emailed "the service exceeded its memory limit" — the 512 MB
free-tier instance was OOM-killed. This is the concrete monitoring signal
that exists today: Render's own resource-limit alerting, not a system this
project built. The response (see git history, `fix(backend): memory
hardening for 512MB instance`):

1. Read the email → correlated it with recent changes (uploaded photos
   were being processed at full phone-camera resolution, 4000px+).
2. Fix: downscale to 1280px on upload before any processing (MediaPipe
   pose/face detection, compositing, and the inpainting mask all work
   fine at that resolution on ~10x smaller arrays), explicit `del` +
   `gc.collect()` at the end of the heaviest request handlers, and
   `gunicorn --workers 1 --threads 4` (a second worker would duplicate
   MediaPipe's models and the XGBoost artifact in RAM; threads share
   them).
3. Verified by redeploying and re-running the upload → try-on flow;
   Render's memory graph confirmed the drop.

This is the monitoring→diagnosis→fix→verify loop this project relies on:
platform-level alerts, not custom instrumentation, because the operating
scale doesn't justify more.

**What would be added at real scale:** request-level latency/error-rate
dashboards for the try-on endpoints specifically (SDXL/IDM-VTON calls are
the slowest, most failure-prone path — the 402 "insufficient credit"
class of error surfaced during Phase 2 benchmarking is exactly the kind of
thing a dashboard-with-alerting would catch before a user does), and a
cost-tracking dashboard for the pay-per-call Replicate/Anthropic usage.

## Versioning & Rollback

- **Git tags per deploy.** Each Render deploy corresponds to a commit on
  `main` (or `vision-tryon` pre-merge); the commit hash is the version
  identifier. `render deploys list` / the Render dashboard shows exactly
  which commit is live. Rollback = redeploy an earlier commit (`render
  deploys create --commit <sha>` or re-trigger from the dashboard) —
  no separate release-tagging scheme was needed at this scale, but tagging
  `submission-v1` at the final submitted commit is planned for Phase 6 so
  the graded state is unambiguous even if `main` moves afterward.
- **Pinned third-party model versions.** Both generative try-on engines
  pin an exact Replicate version hash in code (`SDXL_INPAINT_VERSION`,
  `IDM_VTON_VERSION` in `backend/vision_tryon.py`), not a bare model slug.
  Community-maintained models on Replicate can change weights under a
  slug without warning; a pinned hash means a redeploy months from now
  reproduces the same renders. Same reasoning as the training `seed=42`
  throughout the graded pipeline — reproducibility is a first-class
  requirement, not an afterthought, and it applies to the ungraded
  generative renderer too.
- **Committed model artifacts.** `models/*.joblib` (LogisticRegression,
  RandomForest, XGBoost — weighted and unweighted — MLP, plus the label
  encoder) are committed to git, not regenerated on deploy. This means:
  (a) the exact model that was fairness-audited is the exact model the
  `/recommend-size` endpoint serves — no train/serve skew — and (b)
  rollback of a *model* is just `git checkout` on `models/`, same as any
  other versioned artifact. `models/training_config.json` records the
  seed, split, and feature set used to produce them, and
  `models/comparison.csv` / `models/runs.csv` record what was tried.

## Why not MLflow (or similar)

MLflow (or W&B, Neptune, etc.) was deliberately not introduced. The
training surface here is four model families, one fixed 80/20 split, one
seed, evaluated once each (Phase 5) plus one deliberate re-run for the
fairness mitigation (Phase 6) — five total runs, ever. That's the entire
experiment history for the life of this project. A dedicated experiment
tracker earns its cost when there are many runs, several people, or
hyperparameter sweeps to compare across — none of which apply here. A
flat CSV (`models/runs.csv`) plus the committed artifacts and JSON config
gives full reproducibility and comparability with none of the
infrastructure (tracking server, UI, run-ID plumbing) an experiment
tracker would add for no payoff at this scale. If this project grew into
one with recurring retraining or a team, MLflow would be the first
addition.

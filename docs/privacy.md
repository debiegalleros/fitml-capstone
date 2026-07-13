# Privacy & Data Protection — Body Photo Uploads

FitML asks shoppers to upload a photo of themselves so the try-on feature
can composite garments onto it. A body photo is sensitive personal data,
so the prototype is built privacy-first: every protection below is
implemented in the deployed code (file references throughout), not just
policy text.

## What is collected, and why

| Data | Purpose | Used by the ML model? |
|---|---|---|
| Front photo (+ optional side/back) | Try-on pose extraction and compositing only | **No** — the photo never feeds the size model |
| Height, weight, bust, waist, hip, body type (manually entered) | Size recommendation | Yes — this is the model's only input |
| First name (optional) | Personalizing the advice text | No |

The single most important boundary: **measurements are always typed in by
the user; nothing is ever estimated from the photo.** The uploaded photo
is used exclusively for MediaPipe pose keypoints and the composited
preview. This is a locked design decision (see
[genai_usage.md](genai_usage.md)) — it keeps unquantifiable,
body-type-correlated vision errors out of both the predictions and the
fairness audit.

## Protections implemented

1. **Anonymous, session-scoped storage.** Uploads live under
   `backend/uploads/{session_id}/` where `session_id` is a random UUID4
   ([backend/app.py](../backend/app.py)). Folders are never keyed to
   name or email; the optional first name is stored only in the profile
   row for advice personalization. One session's token cannot retrieve
   another session's photos.
2. **24-hour auto-deletion.** `UPLOAD_TTL_HOURS = 24`
   ([backend/config.py](../backend/config.py));
   [backend/cleanup_uploads.py](../backend/cleanup_uploads.py) purges
   expired session folders on app startup, before every new upload, and
   as a standalone scheduled script. On the deployed Render instance the
   disk is ephemeral, so every redeploy wipes uploads as well — a
   stricter version of the same guarantee.
3. **Face blur ON by default.** Before the photo is ever written to
   disk, MediaPipe's face detector locates the face and a Gaussian blur
   is applied over a region expanded 25% beyond the detected box
   (`blur_face` in [backend/tryon.py](../backend/tryon.py)). Because the
   blur happens pre-save, **an unblurred version never persists** unless
   the user explicitly opts out on the upload form. This was deliberately
   flipped from "show by default" to privacy-first.
4. **Face region re-applied to every generated try-on render — a
   code-enforced step, not an assumption about the generative models.**
   The face box is detected at upload time regardless of the blur
   setting (`detect_face_bbox` in
   [backend/tryon.py](../backend/tryon.py)) and saved to the session's
   `pose.json`. After Claude Vision + SDXL or IDM-VTON generate a
   try-on, `_paste_source_face` in
   [backend/vision_tryon.py](../backend/vision_tryon.py) re-composites
   whatever pixels actually occupy that box on the *stored* session
   photo — blurred, or unblurred if the shopper opted out — back onto
   the generated image, with a softly feathered edge so the seam is
   invisible. This runs identically for both engines. It exists because
   SDXL's pure mask-constrained inpainting preserves the face by
   construction, but garment-conditioned engines like IDM-VTON can
   internally regenerate the entire frame, including the face — verified
   during engine benchmarking, where IDM-VTON produced a fully
   synthetic, unrelated face on some full-body renders. Rather than
   trust each engine's behavior (which can also change with a model
   version bump), the box is re-applied in code after every render, for
   both engines, so the guarantee holds even if an engine's internals
   change.
5. **HTTPS-only transmission.** Both deployed surfaces
   (`https://fit-ml.netlify.app`, `https://fit-ml.onrender.com`) serve
   over TLS; local dev photos stay on the developer machine
   (FileVault-encrypted disk).
6. **Never committed.** `backend/uploads/` is gitignored; no user photo
   can enter the repository or its history.
7. **Consent notice at the point of collection.** The profile upload
   screen ([frontend/profile.html](../frontend/profile.html)) states
   what is collected, what it is used for, and the 24-hour retention
   limit before the user submits anything.
8. **Minimal collection.** Weight, side/back photos, and name are all
   optional; the upload cap is 10 MB with an allow-listed set of image
   types ([backend/config.py](../backend/config.py)).

## Regulatory framing — Data Privacy Act of 2012 (RA 10173)

The prototype is framed against the Philippines' Data Privacy Act of
2012 (Republic Act No. 10173) and its implementing principles:

- **Transparency** — the upload screen discloses what is collected, the
  purpose, and the retention period before collection happens (§1, §7
  above); this document is public in the repository.
- **Legitimate purpose** — photos are collected for exactly one declared
  purpose (try-on rendering). They are not used for model training, not
  used for measurement inference, and not shared with third parties.
  The one external call that sees a derived image — the Claude advice
  endpoint receives the *composited, face-blurred* try-on image for
  qualitative fit commentary — serves the same declared purpose and is
  documented in [genai_usage.md](genai_usage.md).
- **Proportionality** — only data necessary for the feature is
  collected (§"What is collected"); face blur removes identity
  information the try-on does not need.
- **Retention limitation** — hard 24-hour TTL, enforced in code rather
  than by policy alone (§2 above).
- **Security** — TLS in transit, encrypted disk at rest (Render
  infrastructure / FileVault locally), anonymous UUID keying,
  session-scoped access.

## Separation from the fairness audit

The Phase 6 fairness audit runs entirely on the public Kaggle research
dataset (ModCloth + RentTheRunway reviews). No user-uploaded photos or
live profile data are used for auditing or any other secondary purpose —
consent given for the try-on feature is not reused for research, per
RA 10173's purpose-limitation principle.

## Known gaps (prototype honesty)

- **CORS is permissive** (`flask_cors.CORS(app)`, all origins) — fine for
  an unauthenticated demo; production hardening would pin the exact
  frontend origin ([deployment.md](deployment.md)).
- **No user accounts** means deletion-on-request is moot (everything
  self-deletes in 24h), but also means there is no authenticated way to
  review one's own stored data during that window beyond holding the
  session token.

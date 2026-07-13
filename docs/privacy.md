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
3. **Crop-at-upload (opt-in) — when selected, the face region is removed
   before it is ever stored or processed, not obscured after the fact.**
   A `crop_face` checkbox on the upload form, **unchecked by default**.
   When checked, MediaPipe's face detector runs once on the raw upload to
   locate the nose tip (`detect_nose_y` in
   [backend/tryon.py](../backend/tryon.py)); that detection result is
   used immediately to compute a single crop boundary and then discarded
   — no bounding box, landmarks, or face data of any kind is ever saved.
   `crop_above_nose` removes everything above that boundary before the
   photo is downscaled, written to disk, pose-extracted, or sent to any
   try-on engine. When selected, this is a **stronger, code-enforced
   data-minimization guarantee than blurring**: a blurred face is still
   face data present in the file (obscured, but there); a cropped photo
   has no face pixels at all, in the stored photo, in `pose.json`, or in
   any generated try-on image. If checked but the nose can't be
   confidently detected (face turned away, occluded, poor lighting), the
   upload is rejected with a friendly re-upload prompt rather than
   guessing a crop boundary — a wrong guess could leave part of the face
   exposed, which would be worse than asking for a clearer photo.

   **When left unchecked (the default), the photo is stored and processed
   with the face visible, and the residual risk below applies.** This
   feature replaced an earlier design: face blur applied to every upload
   by default, plus a post-generation face-region paste-back
   (`_paste_source_face`, now removed) that guarded against generative
   engines regenerating the blurred area — specifically because
   benchmarking found IDM-VTON can regenerate a fully synthetic,
   unrelated face on some full-body renders (see
   [genai_usage.md](genai_usage.md)). Cropping structurally closes that
   finding **only when the checkbox is used** — an uncropped photo goes
   into the same try-on pipeline with no equivalent protection today,
   since the paste-back mechanism was retired rather than kept
   conditionally. This is a known, currently-unmitigated gap for the
   unchecked path, not an oversight being glossed over — flagged here
   for the record and left as a decision point on whether to (a)
   reintroduce paste-back protection conditionally for uncropped photos,
   (b) make the checkbox default to checked, or (c) accept the risk as
   documented.
4. **HTTPS-only transmission.** Both deployed surfaces
   (`https://fit-ml.netlify.app`, `https://fit-ml.onrender.com`) serve
   over TLS; local dev photos stay on the developer machine
   (FileVault-encrypted disk).
5. **Never committed.** `backend/uploads/` is gitignored; no user photo
   can enter the repository or its history.
6. **Consent notice at the point of collection.** The profile upload
   screen ([frontend/profile.html](../frontend/profile.html)) states
   what is collected, what it is used for, and the 24-hour retention
   limit before the user submits anything.
7. **Minimal collection.** Weight, side/back photos, and name are all
   optional; the upload cap is 10 MB with an allow-listed set of image
   types ([backend/config.py](../backend/config.py)).

## Regulatory framing — Data Privacy Act of 2012 (RA 10173)

The prototype is framed against the Philippines' Data Privacy Act of
2012 (Republic Act No. 10173) and its implementing principles:

- **Transparency** — the upload screen discloses what is collected, the
  purpose, and the retention period before collection happens (§1, §6
  above); this document is public in the repository.
- **Legitimate purpose** — photos are collected for exactly one declared
  purpose (try-on rendering). They are not used for model training, not
  used for measurement inference, and not shared with third parties.
  The one external call that sees a derived image — the Claude advice
  endpoint receives the *composited* try-on image (no face pixels ever
  present, per §3) for qualitative fit commentary — serves the same
  declared purpose and is documented in [genai_usage.md](genai_usage.md).
- **Proportionality** — only data necessary for the feature is
  collected (§"What is collected"); the face crop removes identity
  information the try-on does not need, at the source, rather than
  merely obscuring it downstream.
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

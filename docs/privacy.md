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
   has the upper face — forehead and eyes — entirely absent from the
   frame (not merely obscured) in the stored photo, in `pose.json`, and
   in any generated try-on image, because that region of the image no
   longer exists past this point. The crop boundary is the nose tip, as
   specified: the lower face (nose tip, mouth, chin, jaw) remains visible
   in the cropped photo and in any resulting try-on render. If checked
   but the nose can't be
   confidently detected (face turned away, occluded, poor lighting), the
   upload is rejected with a friendly re-upload prompt rather than
   guessing a crop boundary — a wrong guess could leave part of the face
   exposed, which would be worse than asking for a clearer photo.

   **When left unchecked (the default), the photo is stored and processed
   with the face visible — and gets a second, defense-in-depth
   protection instead of the structural guarantee above.** This feature
   replaced an earlier design: face blur applied to every upload by
   default, plus a post-generation face-region paste-back
   (`_paste_source_face`) that guarded against generative engines
   regenerating the blurred area — specifically because benchmarking
   found IDM-VTON can regenerate a fully synthetic, unrelated face on
   some full-body renders (see [genai_usage.md](genai_usage.md)).
   Cropping structurally closes that finding for the upper face when the
   checkbox is used — there is no forehead/eye pixel in the input for any
   engine to regenerate incorrectly, because that region is outside the
   frame entirely. The lower face (mouth, chin, jaw) remains in frame on
   the checked path; that region is not separately protected by
   paste-back on that path, since `_paste_source_face` is scoped to the
   unchecked path only (below). When the checkbox is left
   unchecked, `detect_face_bbox` (in
   [backend/tryon.py](../backend/tryon.py)) locates the face once on the
   raw upload — never on the checked path — and that box travels with
   the session's `pose.json` so `_paste_source_face` (in
   [backend/vision_tryon.py](../backend/vision_tryon.py)) re-composites
   the real face region from the stored photo onto every generated
   try-on render, with a feathered edge, regardless of what the engine
   drew there. This is the same mechanism the earlier mandatory-blur
   design used, reinstated specifically for the unchecked path — so both
   checkbox states address the IDM-VTON synthetic-face-regeneration
   finding for the upper face (forehead, eyes), by two different means:
   removing that region from the input entirely, so it's structurally
   absent (checked) versus restoring the real, unaltered face onto the
   output regardless of what the engine drew there (unchecked). Only the
   unchecked path's paste-back covers the lower face; on the checked
   path the lower face is present in frame without that additional
   protection.
4. **Third-party processing disclosure — generative try-on sends the
   photo to Replicate.** The IDM-VTON and SDXL rendering engines run on
   Replicate's hosted infrastructure (see
   [genai_usage.md](genai_usage.md)), so the shopper's photo — already
   cropped above the nose first, if that checkbox was used — is
   transmitted to Replicate as a data processor, for the sole purpose of
   generating the try-on render. Replicate never receives body
   measurements, name, or any other profile field, only the image and
   the derived mask/prompt; FitML's 24-hour deletion window governs the
   copy on FitML's own infrastructure regardless of this external call.
   This is disclosed in the upload consent text
   ([frontend/profile.html](../frontend/profile.html)).
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
  purpose, and the retention period before collection happens (§1, §6
  above); this document is public in the repository.
- **Legitimate purpose** — photos are collected for exactly one declared
  purpose (try-on rendering). They are not used for model training and
  not used for measurement inference. The two external calls that see
  an image — Replicate (IDM-VTON/SDXL) receiving the shopper's photo to
  render the try-on, and Claude receiving the *composited* try-on image
  for qualitative fit commentary — both serve that same declared
  purpose, are disclosed in the consent text, and are documented in
  [genai_usage.md](genai_usage.md).
- **Proportionality** — only data necessary for the feature is
  collected (§"What is collected"); the face crop removes the upper-face
  identity information (forehead, eyes) the try-on does not need, at the
  source, rather than merely obscuring it downstream.
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

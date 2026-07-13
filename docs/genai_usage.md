# Generative AI Usage in FitML (Step 9)

FitML uses generative/deep-learning AI in three places, all deliberately
**outside** the graded ML pipeline (the trained size classifiers and their
fairness audit). This document describes each use, shows the integration code,
and explains the boundaries that keep GenAI separate from the graded work. It
becomes a section in the final report and a slide in the technical deck.

## 1. Claude API as a product feature — multimodal fit advice

**What it does.** After the try-on composite is rendered, the Flask backend's
`/advice` endpoint sends Claude two things: (a) the user's profile measurements
plus the recommended size and its confidence, and (b) the **composited try-on
image itself** (multimodal input). Claude returns a short piece of personalized
advice text that can reference what is actually visible in the composite —
e.g. shoulder seam alignment, hem length relative to the torso, or apparent
fabric pull across the chest — alongside measurement-based reasoning ("the
M is recommended because your waist sits mid-range for this size"). When the
rule-based history lookup finds 2+ prior items from the same brand/fabric with
consistent fit feedback, that note is added to the prompt so the advice can say
so ("items from this brand have run true to size for you").

**What it explicitly does NOT do:**

- Claude does **NOT estimate body measurements** — not from the uploaded
  photo, not from the composite. Measurements are always manually entered by
  the user on the profile screen.
- Claude does **NOT override or feed into the trained size model.** The size
  recommendation is produced entirely by the Phase 5 classifier before Claude
  is ever called; Claude receives that recommendation as read-only context.
- Claude's visual observations are **qualitative commentary only.** If the
  advice text mentions that a hem looks short, that is a remark about the
  rendered composite — it changes no prediction, no confidence score, and no
  stored data.

**Integration code** — implemented in
[`backend/advice_engine.py`](../backend/advice_engine.py), called by the
`/advice` route in [`backend/app.py`](../backend/app.py). The core call
(abridged from the real file):

```python
# backend/advice_engine.py (abridged)
ADVICE_MODEL = "claude-opus-4-8"

SYSTEM_PROMPT = """\
You are FitML's fit advisor for an e-commerce virtual fitting room.
Write personalized clothing-fit advice for a shopper. Rules:
- Write EXACTLY two paragraphs, separated by a blank line.
- Paragraph 1: explain the size recommendation using the shopper's
  measurements in plain language (why this size should fit their body).
- Paragraph 2: must start with the word "Note:" followed by visual
  observations from the attached try-on photo. Use simple everyday language
  a non-technical shopper understands. NO tailoring jargon ...
- The try-on image is a 2D preview composite, so comment on overall size and
  placement, not fabric texture or lighting.
...
"""

def generate_advice(profile, item, recommendation, size, image_path):
    client = anthropic.Anthropic()  # key from ANTHROPIC_API_KEY, never committed
    ...  # context lines: measurements, garment metadata, recommended size +
    # confidence, amber/borderline flag, non-recommended-size note, and any
    # rule-based history note — assembled server-side, never free-form user text
    response = client.messages.create(
        model=ADVICE_MODEL,
        max_tokens=400,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image",   # the composited try-on image
                 "source": {"type": "base64", "media_type": media_type,
                            "data": image_b64}},
                {"type": "text", "text": "\n".join(lines)},
            ],
        }],
    )
    return next(b.text for b in response.content if b.type == "text").strip()
```

The deployed prompt enforces the product's plain-language rule: the second
paragraph must open with "Note:" and describe the try-on photo in everyday
words (jargon like "seam alignment" or "hem" is explicitly banned — a
deliberate change from the early draft of this document, which had suggested
Claude could use those terms). The history lookup (`history_note()` in the
same file) is a plain SQL query — 2+ prior items from the same brand or
fabric with identical fit feedback adds one context line to the prompt.

Documented per the project's locked decisions as a **multimodal AI
integration**: image + structured text in, advice text out, nothing written
back into the ML pipeline.

## 2. rembg / U²-Net — catalog image background removal

**What it does.** Phase 7's catalog pipeline (`src/`, catalog processing)
runs every product photo from the Fashion Product Images Dataset through
[`rembg`](https://github.com/danielgatis/rembg), which wraps **U²-Net** — a
deep salient-object-detection network — to produce a per-pixel alpha matte.
The output is a transparent-background garment PNG, which is what the try-on
compositor affine-warps onto the user's photo. Automating this replaced what
would otherwise be manual background masking of ~100–115 images (plus 2–3
hue-shifted color variants each).

**Why it counts as GenAI/deep-learning usage.** U²-Net is a pretrained
convolutional neural network performing dense prediction (segmentation). It is
used purely as an **image-processing utility at catalog-build time**: it runs
offline, once per image, and its output is a static asset. It sees no user
data, produces no labels, and touches no training data.

## 3. Generative virtual try-on — Claude Vision + IDM-VTON / SDXL

**What it does.** The try-on renderer shows the shopper wearing the actual
catalog garment on their own uploaded photo. MediaPipe pose keypoints (already
extracted at upload time for the legacy 2D compositor) locate the shoulders,
hips, elbows/wrists, and ankles; the pipeline uses these to build a
clothing-region mask and to select which garment-conditioned generative
engine renders the result. Two engines are implemented behind a
`TRYON_ENGINE` flag, with the legacy 2D affine-warp compositor retained as a
final fallback:

- **IDM-VTON (primary, garment-conditioned diffusion, via Replicate).** Takes
  the shopper's photo and the actual catalog garment cutout directly; masking
  and blending happen inside the model. Selected as the default engine after
  a head-to-head benchmark (`docs/assets/tryon_samples/` — **stale as of the
  crop-at-upload change below: these were generated against the earlier
  blurred-face photos, not the current cropped-above-the-nose ones. Flagged
  for regeneration before final submission; not blocking the engine
  decision, which the crop doesn't change**) on 3 clean cases (a top on a
  full-body photo, a top on an upper-body photo, and a dress): IDM-VTON
  rendered the correct garment — color, silhouette, and detail — in all 3;
  the SDXL fallback (below) got the garment wrong in all 3.
- **SDXL inpainting (fallback, `lucataco/sdxl-inpainting` on Replicate).**
  Claude Vision (`claude-sonnet-4-6`) first analyzes the shopper's photo and
  the garment image, producing a structured JSON prompt (pose, framing,
  lighting, garment description, and one SDXL prompt + negative prompt — the
  analyzer is hard-banned from mentioning body measurements, enforced by an
  automated smoke-test assertion). SDXL then inpaints only the pose-derived
  clothing-region mask.

**Engine-selection rationale — SDXL fork limitation, investigated and
documented rather than silently worked around.** During benchmarking, SDXL
consistently rendered a garment resembling the shopper's *original* clothing
(wrong color, wrong style) rather than the requested catalog item, despite a
prompt explicitly describing the target garment and a negative prompt banning
"original clothing visible." This was investigated rather than assumed to be
a simple parameter issue:

- The model's own schema states `strength: 1.0` means "full destruction of
  information in image." Raising `strength` from 0.99 to the maximum 1.0 was
  tested — this made things *categorically worse*, not better: the mask
  boundary itself broke, and the face/background outside the mask (which
  should never change under masked inpainting) were also visibly regenerated
  and distorted, while the garment *still* did not match the requested
  color or style.
- Raising `guidance_scale` to its maximum (10) and switching schedulers
  (`DPMSolverMultistep`) were also tested against the same case; neither
  changed the outcome.
- Conclusion: this is a fork/implementation limitation, not a tunable
  parameter — this specific community-maintained cog wrapper does not
  reliably implement mask-constrained inpainting the way the standard HF
  diffusers pipeline is documented to. `strength=0.99` (not 1.0) is kept in
  the shipped fallback specifically because it is the less-broken of the two
  tested extremes (mask boundary respected; garment fidelity is the
  remaining known weakness). SDXL therefore stays a fallback path only —
  used automatically if IDM-VTON errors — never the primary engine.

**Privacy guard — two mechanisms, one per checkbox state.** SDXL's pure
mask-constrained inpainting preserves the face and background outside the
mask by construction. IDM-VTON does not offer the same guarantee —
benchmarking found it can regenerate the entire frame, including a fully
synthetic, unrelated face, on some full-body renders. Two protections now
cover this, one per state of the opt-in `crop_face` checkbox:

- **Checked — structural fix.** The photo is cropped above the nose
  before any storage or processing (see `docs/privacy.md`), so there is
  no face pixel in the input for any engine to regenerate incorrectly in
  the first place.
- **Unchecked (the default) — reinstated paste-back.** `_paste_source_face`
  re-composites the detected face region from the stored photo back onto
  every generated render, with a feathered edge. This is the same
  mechanism the original always-on-blur design used; it was retired when
  crop-at-upload first shipped (mandatory, no toggle) on the reasoning
  that a face-free input made it unnecessary, and reinstated once the
  checkbox became opt-in, specifically scoped to the unchecked path only
  — `detect_face_bbox` (in `backend/tryon.py`) is never called on the
  checked path, since a cropped photo has nothing left to protect.

Both states now close the finding this section describes, by two
different means: removing the face from the input entirely (checked)
versus restoring the real face onto the output regardless of what the
engine drew there (unchecked). This is a case where a documented bug (a
bounded, per-render defensive patch) led first to a stronger structural
fix, and then — once the toggle reopened a gap for the unchecked path —
to that original patch being deliberately brought back for the one case
that still needed it, rather than leaving the gap disclosed-but-live in
a shipping product.

**What it explicitly does NOT do:** the same measurement/audit boundaries as
the advice text (§1) apply here too — the analyzer prompt never estimates or
mentions body measurements (enforced by an automated test), and nothing from
the try-on pipeline feeds the graded size model. Seed defaults to 42 for
both engines for reproducibility, and both Replicate model versions are
pinned in code (not floating on a bare model slug) for the same reason
`seed=42` is fixed throughout the graded pipeline.

**Bottoms try-on — root-caused to source-photo confound, resolved with a
clean stand-in.** Two rounds of testing isolated the cause of the earlier
"garment barely changes" result. Round 1 used a dress-source photo (the
one-piece garment gives IDM-VTON's internal segmentation no separate
"existing bottoms" region to detect). Round 2 used a jacket+jeans photo
where the long jacket's hem sits close to the waistband — same failure,
milder. Round 3 used a genuinely clean stand-in (a fitted tank ending
well above the waist, belted, with unambiguous jeans below, confirmed via
a programmatic color-heuristic scan of the catalog rather than eyeballing
candidates) and the segmentation-confusion failure did not reproduce:
both a blue-jeans and a black-jeans request visibly regenerated the
region (the original belt disappeared, denim texture was redrawn), and
the black-jeans request measurably shifted color — mean region RGB moved
from (155, 155, 158) on the source to (145, 138, 138), a shift toward
darker and less blue-dominant, vs. (156, 155, 158) for the blue-jeans
request landing almost exactly on the source. The remaining gap — the
black-jeans result reads as a darker wash rather than a confidently black
garment — is a color-fidelity nuance, not the segmentation failure the
first two rounds showed, so bottoms try-on ships rather than being pulled
from the live catalog. SDXL failed to change the garment in all three
rounds (consistent with the fork limitation above) — bottoms try-on is
IDM-VTON-reliant in practice, with SDXL as a same-tier fallback that
inherits its general limitation here too. The lower-body standardization
feature (below) is unaffected either way, since it deliberately targets
the full hip-to-ankle region rather than relying on the engine to
auto-detect existing bottoms.

**Upper-body standardization (the symmetric case) — attempted, not
shipped.** The lower-body standardization feature (any top/outerwear
try-on standardizes what's below to plain black leggings, via a chained
IDM-VTON pass — ships, verified reliable) has a symmetric counterpart:
any bottoms try-on standardizing what's above to a plain black
long-sleeve top, so a dress-source photo doesn't leave a mismatched
original top visible. This was built and tested but is **disabled**
(`_standardizes_upper_body` always returns `False`) after two rounds of
investigation both surfaced real problems rather than fixing one:

1. First reference garment (a black long-sleeve tee, tunic-length in its
   own product photo): IDM-VTON faithfully reproduced that length and the
   standardized top visually covered the actual rendered bottoms entirely
   — a proportions problem.
2. Second reference garment (a charcoal cable-knit sweater with a correct
   waist-length hem in its own product photo): proportions were right,
   but the rendered bottoms from the first pass no longer read clearly as
   jeans in the boundary region — a different failure, boundary bleed
   between the two chained calls rather than an asset-length problem.
3. Tested whether call order was the cause (top-standardization first,
   then bottoms, mirroring the working leggings direction) — same
   boundary bleed persisted. This rules out a call-order fix and points
   to a genuine IDM-VTON limitation specific to this composition
   (redressing upper_body on an image whose lower body was itself just
   generated), not a tunable reference-image or ordering choice.

Bottoms try-on itself (a shopper already in a separate top and bottom)
is unaffected and ships normally — only the dress-source symmetric case
is affected. Left as documented future work rather than shipped with a
visually incorrect result. Comparison images from both attempts are in
`docs/assets/tryon_samples/upper_standardization_not_shipped_finding.png`.

## Responsible boundaries — why GenAI is kept out of the graded pipeline

The graded core of this capstone (Steps 4–5: model implementation and the
fairness audit, 40 of 100 points) is a supervised classifier trained on real
customer fit feedback, audited for group fairness. GenAI components were
deliberately fenced off from it, for three reasons:

1. **Auditability.** The fairness audit makes quantitative claims (disparate
   impact, equalized odds, per-group accuracy) about a model whose inputs,
   training data, and parameters are fixed and inspectable. A large generative
   model in the prediction path would make those claims unverifiable — its
   behavior can't be stratified, retrained with class weights, or explained
   with SHAP the way the graded classifiers can.
2. **No silent data invention.** The single most consequential design rule in
   FitML is that measurements are user-entered, never inferred. Letting a
   vision model estimate bust/waist/hip from a photo would inject
   unquantifiable, likely body-type-correlated error into exactly the
   variables the fairness audit stratifies on — undermining both accuracy
   claims and the audit itself.
3. **Reproducibility.** Every graded artifact (cleaned dataset, trained
   models, audit numbers) is reproducible from fixed seeds and committed
   code. Generative API output is non-deterministic and
   service-dependent, so it is confined to surfaces where variation is
   harmless: advice prose, one-time asset preparation, and try-on
   rendering (seeded, and never fed back into the sizing model).

The result is a clean separation: **deterministic, audited ML decides the
size; generative AI only renders and explains it.**

| Component | AI type | When it runs | Touches graded ML? |
|---|---|---|---|
| Size recommendation | LogReg/RF/XGBoost/MLP (trained, Phase 5) | Per request | **Is** the graded ML |
| Fairness audit | Metrics + SHAP on the above | Offline, Phase 6 | Audits the graded ML |
| Fit advice text | Claude API, multimodal (image + text in, text out) | Per request, after prediction | No — read-only consumer |
| History suggestions | Rule-based SQL lookup (not ML at all) | Per request, feeds the advice prompt | No |
| Generative virtual try-on | Claude Vision + IDM-VTON (primary) → SDXL inpainting (fallback) → 2D affine-warp compositor (final fallback) | Per request, after prediction | No — renders the recommended size, never estimates it |
| Catalog background removal | rembg / U²-Net (pretrained CNN) | Once, catalog build (Phase 7) | No — asset prep only |

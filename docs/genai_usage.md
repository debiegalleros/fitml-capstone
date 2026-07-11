# Generative AI Usage in FitML (Step 9)

FitML uses generative/deep-learning AI in exactly two places, both deliberately
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
   harmless: advice prose and one-time asset preparation.

The result is a clean separation: **deterministic, audited ML decides the
size; generative AI only explains it and helps build the demo catalog.**

| Component | AI type | When it runs | Touches graded ML? |
|---|---|---|---|
| Size recommendation | LogReg/RF/XGBoost/MLP (trained, Phase 5) | Per request | **Is** the graded ML |
| Fairness audit | Metrics + SHAP on the above | Offline, Phase 6 | Audits the graded ML |
| Fit advice text | Claude API, multimodal (image + text in, text out) | Per request, after prediction | No — read-only consumer |
| History suggestions | Rule-based SQL lookup (not ML at all) | Per request, feeds the advice prompt | No |
| Catalog background removal | rembg / U²-Net (pretrained CNN) | Once, catalog build (Phase 7) | No — asset prep only |

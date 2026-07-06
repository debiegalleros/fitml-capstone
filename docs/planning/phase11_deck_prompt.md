# Phase 11 — Deck-Building Prompt

Saved for use when Phase 11 (docs, README, decks, GitHub push) starts. Paste as
the instruction for building both presentation decks.

---

Build both presentation decks per CLAUDE.md's presentation branding rules
(white background, AIM logo upper left every slide, program name on title
slide).

Technical deck (`Debie_Galleros_Pillar5_CapstoneProject_TechnicalDeck.pptx`,
8–12 slides, audience: peers/faculty): title slide → problem framing + task
type → dataset overview (source, size, class distribution) → EDA highlights +
feature engineering → model comparison table with macro-F1 reasoning →
fairness audit methodology + before/after mitigation numbers → SHAP
explainability visuals → architecture (pipeline diagram: data → model → Flask
→ website) → GenAI usage (multimodal advice endpoint, responsible boundaries)
→ limitations + future work (cite the FIT dataset — 2026, 1.13M fit-aware
try-on triplets — as the research direction for photorealistic fit-accurate
rendering, upgrading the current size-proportional 2D scaling).

Business deck (`Debie_Galleros_Pillar5_CapstoneProject_BusinessDeck.pptx`,
8–12 slides, audience: executives, no jargon): title slide → the returns
problem ($100B+, 35% online return rate) → what FitML does (sizing + virtual
try-on, in shopper language) → how it's different (fairness-audited, works
for all body types) → demo screenshots of the website → business impact
(fewer returns, higher confidence, ROI logic) → risks + how they're managed
(privacy, data protection, RA 10173) → rollout strategy → future work (3D
try-on, color matching, photorealistic fit-accurate try-on powered by new
research data — the FIT dataset, 2026) → closing.

Use real numbers from `models/comparison.csv` and `docs/fairness_report.md`.
Keep slides visual — charts and screenshots over text walls.

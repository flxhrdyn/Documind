# Product

## Register

product

## Users

Primary viewer is a portfolio audience: recruiters, hiring managers, and potential clients evaluating
the developer's skill through a live demo. Secondary user is the developer themselves during
walkthroughs. Context is a focused, unhurried demo session (desktop-first, well-lit, likely screen-shared
or presented live) rather than daily high-volume production use. The job to be done for the viewer is to
quickly form a positive judgment of engineering and design quality while interacting with a real RAG
Q&A workflow: upload a PDF, ask questions, read grounded answers with sources, watch retrieval/generation
metrics update.

## Product Purpose

InvenioAI is a RAG (Retrieval-Augmented Generation) system for document Q&A over PDFs: hybrid
dense+sparse retrieval, RAG Fusion multi-query, cross-encoder reranking, chain-of-thought reasoning via
Groq, dual-layer semantic caching. The React frontend replaces the current Streamlit UI as the primary
demo surface. Success looks like a viewer feeling the product is "real" and carefully made, not a wrapped
notebook: fast perceived response, legible source attribution, an interface that reads as a considered
product rather than a bundled ML demo.

## Brand Personality

Cool, clean, precise. Not cold/clinical dev-tool in the sterile-admin-panel sense, not cutesy/playful.
Calm confidence: the interface gets out of the way of the conversation and the documents, using restraint
and typography rather than heavy chrome. The current cool neutral-gray + teal system (light and dark) is
a deliberate choice, confirmed by the owner over a warm-toned alternative. Clarity and calm carried through
color/type choices and copy tone, not through decoration.

## Anti-references

- Gradient-text logos, side-stripe accent borders, tiny uppercase eyebrows over every section.
- Dense, cluttered "admin panel" feel (heavy borders, noisy chrome) — cool and neutral is fine; busy and
  dense is not.
- Anything that reads as an unstyled Streamlit/Gradio demo wrapper.

## Design Principles

- **Conversation first**: chat/answer content and source attribution are the visual focus; chrome
  (nav, controls, metrics) recedes.
- **Show, don't tell**: demonstrate retrieval quality and reasoning through legible, inline surfaces —
  a clickable `[n]` citation in the answer that scrolls to and highlights its source card, not a bare
  claim the reader has to take on faith.
- **Cool restraint**: clarity expressed through a considered color/type system and copy voice, not
  through gradients, illustration, or heavy card decoration.
- **One good idea per screen**: resist the reflex to fill space with symmetric card grids; let
  hierarchy and whitespace do the work outside the analytics dashboard (where stat tiles are the
  right tool — see Design System Snapshot).
- **Dual-theme as a first-class citizen**: light and dark are both fully designed, not one derived
  by inversion.

## Design System Snapshot

Current build, kept in sync here as the source of truth — update this section whenever tokens or
layout patterns change.

- **Color**: cool neutral grays + a single teal accent, both themes hand-tuned (not a CSS invert).
  Light: bg `#f6f7f8`, surface `#ffffff`, surface-2 `#eef0f2`, ink `#14161a`, ink-muted `#676c76`,
  line `#dfe2e6`, accent `#0d9488` (soft `#14b8a6`, ink `#0f766e`). Dark: bg `#0a0b0d`,
  surface `#121317`, surface-2 `#17181d`, ink `#e7e9ec`, ink-muted `#82868f`, line `#23252b`,
  accent `#2dd4bf` (soft `#5eead4`, ink `#5eead4`).
- **Type**: Outfit for display/headings, Inter for body and UI text, JetBrains Mono for data,
  timestamps, labels, and metric values.
- **Shell layout**: fixed three-column desktop shell (`h-screen`, no page-level scroll) — left
  sidebar (upload + knowledge base list), center chat column, right references panel. Each column
  scrolls independently within its own bounds.
- **Chat**: citation markers `[n]` render as small clickable accent badges inline in the answer;
  clicking scrolls to and rings the matching card in the references panel.
- **Analytics**: stat-tile grid + area chart + metric cards are the intentional pattern for the
  session-metrics dashboard (a data-density context, distinct from marketing/landing pages where
  hero-metric tiles are the templated default to avoid).
- **Cards**: `rounded-xl`/`rounded-2xl` surfaces with a single hairline border (`border-line`), no
  double border+shadow stacking.

## Accessibility & Inclusion

WCAG 2.1 AA as the baseline (contrast ratios, keyboard navigation, focus states, semantic structure).
Light/dark theme switch is a required feature, not optional polish. Respect
`prefers-reduced-motion` for all transitions/animations.

## Known Gaps (tracked, not yet fixed)

- No responsive breakpoints below desktop — the three-column shell overflows horizontally on mobile
  viewports instead of collapsing/stacking.
- Reference cards occasionally show `unknown` file / `Page 0` / `Match 0%` when chunk metadata or
  rerank score isn't attached correctly — undermines the "show, don't tell" retrieval-quality signal.

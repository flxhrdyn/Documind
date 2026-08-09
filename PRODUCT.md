# Product

## Register

product

## Users

Researchers and knowledge workers who upload PDFs and ask questions against them. They work in
focused sessions: upload a document, interrogate it via chat, check citations against the source
text, and occasionally review retrieval-quality analytics. Context is a desktop browser, task-focused,
low tolerance for friction between asking a question and seeing a sourced answer.

## Product Purpose

InvenioAI is a RAG (retrieval-augmented generation) document Q&A tool: hybrid dense+sparse retrieval,
reranking, and cited LLM answers over uploaded PDFs. Success is a user trusting an answer because
they can see exactly which passage backs it up, and being able to gauge retrieval health via the
analytics dashboard.

## Brand Personality

Bold, energetic, playful — a technical tool that still feels alive rather than clinical. Confident
teal accent used with more presence than a typical muted enterprise app, motion that gives feedback
personality, without tipping into noise that undermines trust in the underlying answers.

## Anti-references

- Generic AI-SaaS cream/beige body background with a terracotta accent.
- Near-black background with a single neon accent used purely for shock value.
- Flat, bordered-only cards with no depth cue (the "wireframe with color" look).
- Hero-metric-card cliche (big number + small label + gradient) used everywhere.
- Numbered eyebrows / 01-02-03 markers where content isn't actually a sequence.

## Design Principles

- Citations are the trust mechanism — every answer must make its supporting passages one click away
  and legible at a glance (match strength, source file, page).
- Depth signals real hierarchy — floating panels (nav, composer, reference cards) get visible
  elevation, not just a 1px border, so the eye knows what's foreground vs background.
- One confident accent, used generously on the actions and signals that matter (send, active nav,
  match strength), not diluted across every icon.
- Motion confirms action, never decorates — sends, streaming responses, uploads, and tab switches
  should feel responsive; static screens stay calm.

## Accessibility & Inclusion

WCAG AA: body text ≥4.5:1 contrast in both light and dark themes, visible keyboard focus rings,
`prefers-reduced-motion` respected (already implemented in index.css).

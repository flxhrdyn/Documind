---
target: frontend-react/src (React chat + analytics UI)
total_score: 21
p0_count: 2
p1_count: 2
timestamp: 2026-07-11T04-30-03Z
slug: frontend-react-src
---
## Design Health Score

| # | Heuristic | Score | Key Issue |
|---|-----------|-------|-----------|
| 1 | Visibility of System Status | 3/4 | "Thinking..." indicator works well, but citations broadcast `Match 0%` / `unknown` / `Page 0` — a false "pipeline is broken" signal |
| 2 | Match System / Real World | 2/4 | Citation button `aria-label` is Indonesian while all other UI copy is English (leftover dev string) |
| 3 | User Control and Freedom | 2/4 | No stop-generation control, no edit/retract of a sent question, "Clear All" has no confirm |
| 4 | Consistency and Standards | 3/4 | Tokens (radius/spacing/color) are consistent, but card language diverges (assistant bubble has no border/bg, user bubble and source cards do) |
| 5 | Error Prevention | 1/4 | "Max 15MB" never enforced client-side; destructive "Clear All" has no confirmation |
| 6 | Recognition Rather Than Recall | 3/4 | Enter/Shift+Enter hint always visible; but no legend for what "Match %" means |
| 7 | Flexibility and Efficiency | 2/4 | No shortcut to focus input, no copy/re-ask affordance, no visible chat-history persistence signal |
| 8 | Aesthetic and Minimalist Design | 3/4 | Clean chat pane, but sidebar info card duplicates the chat empty-state copy verbatim |
| 9 | Error Recovery | 1/4 | No visible error path if a query itself fails — only the "no documents" precondition is handled |
| 10 | Help and Documentation | 1/4 | No onboarding/tooltip/legend anywhere; relies entirely on a live presenter |
| **Total** | | **21/40** | **Acceptable — significant improvements needed** |

## Anti-Patterns Verdict

**LLM assessment**: Borderline, leaning no. No gradient text, side-stripes, glassmorphism, or eyebrow labels — this reads closer to a real product than AI-templated filler. Two tells remain: the Analytics stat-tile/metric-card/table stack is the textbook "hero-metric template" repeated three times down one page with no visual differentiation between tiers, and `rounded-2xl` is applied almost everywhere uniformly (chat bubbles, stat cards, sources, dropzone) rather than a considered scale.

**Deterministic scan** (`detect.mjs` against `frontend-react/src`): exit code 2, 2 static findings —
- `bounce-easing` — `ChatPanel.tsx:68` (Tailwind `animate-bounce` on the thinking-dots indicator)
- `overused-font` — `index.css:1` (Inter, flagged as a stylistic/subjective rule, not a real defect — Inter is a legitimate, widely used, highly legible face)

**Browser overlay** (live injection, both `/chat` and `/analytics`): 8 anti-patterns on `/chat`, 9 on `/analytics` —
- `low-contrast`: **3.6:1** on `#f2fbf9` text over `#0d9488` accent background (need 4.5:1) — real AA failure
- `tiny-text` (x3–4): 10px/11px body text (sidebar helper copy, textarea hint)
- `flat-type-hierarchy`: sizes cluster 10–18px, ratio only 1.8:1
- `skipped-heading` (x2–3): `<h1>` "InvenioAI" → `<h3>` "Upload Document" with no `<h2>`; on Analytics, `<h2>` dashboard title → `<h4>` stat value with no `<h3>`
- `line-length`: ~149 chars/line on the Analytics subtitle (target <80ch)
- `nested-cards` (x3–5): upload card, knowledge-base card, and info banner all nested inside the sidebar section
- `bounce-easing`: confirms the static finding

No false positives identified with confidence; all correspond to real, visible elements in the screenshots.

## Overall Impression

The chat experience itself — empty state, message flow, and especially the clickable `[n]` citation that scrolls to and highlights its source — is genuinely well-made and matches the brief's "show, don't tell" principle. But two things undercut it hard: real accessibility defects (contrast failure, skipped heading levels, tiny text) that a deterministic scanner catches instantly, and a live data bug (`unknown`/`Page 0`/`Match 0%`) that fires on the very first query in a demo, directly contradicting the "feel real, not broken" goal the whole brief is built around. The single biggest opportunity is fixing what's already tracked as a known gap before any further visual polish — the prettiest citation UI doesn't help if the first thing a recruiter sees is a source panel that reads as broken.

## What's Working

- The `[n]` citation badge in `ChatMessage.tsx` (click → scroll + `ring-2 ring-accent` highlight on the matching source card) is exactly the "show, don't tell" mechanic the brief calls for, and it's clean, single-purpose code.
- The color system is genuinely dual-theme-designed (distinct light/dark surface and ink tokens), not a naive CSS invert.
- The chat composer (auto-growing textarea, always-visible keyboard hint, focus ring) is understated and functional without extra chrome.

## Priority Issues

- **[P0] Reference cards show `unknown` / `Page 0` / `Match 0%` on real queries.**
  Why it matters: this is the most visible thing on screen right after the first question in a recruiter demo — it reads as "the RAG pipeline is broken," the opposite of the brief's core goal.
  Fix: hide/relabel citations with missing metadata (e.g. "Unranked" instead of "Match 0%", skip filename row if `unknown`) at the render layer, and fix the root metadata/score wiring in the retrieval pipeline.
  Suggested command: `$impeccable harden`

- **[P0] `/chat` is unusable at mobile width (390×844).**
  Why it matters: three fixed-width columns render simultaneously and overlap; any accidental phone screen-share or narrow window instantly breaks the "carefully made" impression the whole brief depends on.
  Fix: collapse the sidebar and references panel behind a drawer/toggle below a `md` breakpoint for the chat route at minimum.
  Suggested command: `$impeccable adapt`

- **[P1] Accessibility defects confirmed by the automated scan: 3.6:1 contrast on accent-foreground text, skipped heading levels (h1→h3, h2→h4), and 10–11px body text.**
  Why it matters: fails the brief's own WCAG 2.1 AA baseline; screen-reader users get a broken heading outline, and low-vision users get sub-AA text.
  Fix: bump `--color-accent-fg` contrast against `--color-accent` (or darken the accent for text-on-accent use), insert the missing `<h2>`/`<h3>` levels, raise the smallest body sizes to ≥12px.
  Suggested command: `$impeccable audit`

- **[P1] Citation button `aria-label` is Indonesian ("Lihat sumber...") while all other UI copy is English; sidebar info card duplicates the chat empty-state copy verbatim.**
  Why it matters: screen readers hit a jarring language switch on the app's signature interactive element; the sidebar duplication violates the brief's "one good idea per screen" principle and wastes space that could instead explain what "Match %" means.
  Fix: translate the aria-label to English; replace the duplicate sidebar copy with a one-line legend on citations/match scores.
  Suggested command: `$impeccable clarify`

- **[P2] No visible error state if a query itself fails (only the "no documents" precondition is handled); "Max 15MB" is never enforced client-side; "Clear All" has no confirmation.**
  Why it matters: a Groq/backend hiccup during a live demo would presumably hang on "Thinking..." forever with no recovery — the worst failure mode for a demo context.
  Fix: add a failure branch in `useChat` with an inline error bubble + retry; add client-side file-size validation; add a confirm step before "Clear All".
  Suggested command: `$impeccable harden`

## Persona Red Flags

**Jordan (confused first-timer)**: Sees the same onboarding sentence twice at once (chat empty state + sidebar box) — mild "did I miss something?" hesitation. Asks a question, sees `Match 0%` / `unknown` / `Page 0`, reasonably concludes the tool doesn't know what it retrieved — trust damage before even reading the (correct) answer underneath.

**Sam (accessibility-dependent user)**: Screen reader hits the Indonesian `aria-label` on every citation badge — the component central to "show, don't tell" becomes confusing instead of clarifying. Automated scan also confirms a real 3.6:1 contrast failure and a skipped heading level Sam's screen reader would announce as a broken outline.

**Alex (power user)**: No keyboard shortcut to jump to the textarea, no copy-answer affordance, no visible signal of whether refreshing the page preserves the conversation — repeated demo runs mean re-typing everything.

## Minor Observations

- `MetricsDashboard.tsx` hardcodes `#818cf8` (indigo) for the Retrieval legend/series color — no indigo token exists in the design system; it's a one-off outside the CSS-variable palette.
- `text-emerald-500` used for high-match scores alongside the teal `--color-accent` introduces a second green hue under close inspection.
- Analytics StatCard `hint` text is often redundant with its own label (e.g. "Average Latency" + "Reflects overall search and answer time").
- Line length on the Analytics dashboard subtitle runs ~149 characters — well past the ~65–75ch reading-comfort ceiling.

## Questions to Consider

- The `unknown`/`Page 0`/`Match 0%` bug was already tracked as a known gap before this polish pass — should it have blocked further visual work rather than being deferred?
- Given the brief's own audience is a "live demo, possibly screen-shared," is "no mobile breakpoints" really an acceptable deferred gap, or should it be treated as P0?
- The chat experience is the strongest part of the app — is the Analytics dashboard's stat-tile-heavy layout actually earning its place, or would a leaner treatment (fewer, more differentiated numbers) serve the brief's "show, don't tell" principle better?

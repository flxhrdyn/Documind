# React Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Streamlit frontend with a route-based React + Vite + TypeScript app in a warm-editorial visual style, wired to the real FastAPI backend, deployed to Vercel with the backend on Azure Container Apps.

**Architecture:** New `frontend-react/` app in the same repo, seeded from `invenioai-ui-ref/` for structure only. React Router routes (`/chat`, `/analytics`). TanStack Query owns server state (documents, metrics); a custom hook consumes the SSE `/query/stream`. Chat history persists in `localStorage`. Backend needs no code change (CORS middleware already exists); only a backend-only container image and Azure/Vercel deploy config are added.

**Tech Stack:** React 19, Vite 6, TypeScript ~5.8, Tailwind CSS v4 (`@tailwindcss/vite`), react-router-dom v7, @tanstack/react-query v5, recharts v3, lucide-react, vitest + @testing-library/react.

## Global Constraints

- Frontend lives in `frontend-react/` (new folder, one repo). Do NOT modify `frontend/` (Streamlit) or `invenioai-ui-ref/` (reference only).
- No secrets in the frontend build. Only `VITE_API_BASE_URL` (public backend URL) is read from env; default `http://localhost:8000`.
- No API key: never send `X-API-Key`. Backend for this deployment runs without `INVENIOAI_API_KEY`.
- Visual style: warm editorial - warm palette (cream/charcoal), display font for headings, generous whitespace. Never the teal/indigo generic look. All colors/fonts come from Tailwind theme tokens, no hardcoded hex in components.
- Backend base URL read once via `import.meta.env.VITE_API_BASE_URL`, trailing slash stripped.
- Backend endpoints (existing, do not change): `POST /query/stream` (SSE), `POST /upload/jobs` -> `{job_id}`, `GET /upload/jobs/{job_id}`, `GET /documents` -> `{documents: string[], count: number}`, `DELETE /documents/delete?filename=`, `DELETE /documents`, `GET /metrics`, `POST /metrics/sync`.
- CORS env var is `INVENIOAI_ALLOWED_ORIGINS` (comma-separated). Middleware already wired in `backend/app/main.py`.
- Node 20+. Package manager: npm.

---

## File Structure

```
frontend-react/
  index.html
  package.json
  tsconfig.json
  vite.config.ts
  tailwind.config.ts            # warm-editorial tokens
  vitest.config.ts
  .env.example                  # VITE_API_BASE_URL
  src/
    main.tsx                    # React root + Router + QueryClientProvider
    App.tsx                     # <Routes> + <Layout>
    index.css                   # Tailwind entry + font-face + CSS vars
    types.ts                    # API + domain types
    lib/
      apiClient.ts              # base URL, fetch wrappers, error formatting
      sse.ts                    # streamQuery() SSE consumer
    hooks/
      useDocuments.ts           # TanStack Query: list/delete/delete-all
      useMetrics.ts             # TanStack Query: metrics
      useUploadJob.ts           # create + poll job with backoff
      useChat.ts                # chat state + localStorage + streaming
    components/
      Layout.tsx                # sidebar + <Outlet/>
      Sidebar.tsx               # brand, ThemeToggle, UploadPanel, KnowledgeBaseList
      ThemeToggle.tsx
      UploadPanel.tsx
      KnowledgeBaseList.tsx
      ConfirmDialog.tsx         # two-step confirm modal
      ChatPanel.tsx
      ChatMessage.tsx           # bubble + CoT collapsible + citation refs
      SourcesPanel.tsx
      MetricsDashboard.tsx
    pages/
      ChatPage.tsx
      AnalyticsPage.tsx
  DEPLOY.md                     # Azure + Vercel steps
backend/
  Dockerfile.api                # backend-only image (no Streamlit)
```

---

### Task 1: Scaffold the `frontend-react` app

**Files:**
- Create: `frontend-react/package.json`, `frontend-react/tsconfig.json`, `frontend-react/vite.config.ts`, `frontend-react/index.html`, `frontend-react/.env.example`, `frontend-react/src/main.tsx`, `frontend-react/src/App.tsx`, `frontend-react/src/index.css`, `frontend-react/vitest.config.ts`

**Interfaces:**
- Produces: a runnable Vite app that mounts `<App/>`; `App` renders the text "InvenioAI".

- [ ] **Step 1: Create `package.json`**

```json
{
  "name": "invenioai-frontend",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite --port 3000",
    "build": "tsc -b && vite build",
    "preview": "vite preview",
    "lint": "tsc --noEmit",
    "test": "vitest run"
  },
  "dependencies": {
    "@tanstack/react-query": "^5.59.0",
    "lucide-react": "^0.546.0",
    "react": "^19.0.0",
    "react-dom": "^19.0.0",
    "react-router-dom": "^7.1.0",
    "recharts": "^3.9.2"
  },
  "devDependencies": {
    "@tailwindcss/vite": "^4.1.14",
    "@testing-library/jest-dom": "^6.6.3",
    "@testing-library/react": "^16.1.0",
    "@types/node": "^22.14.0",
    "@types/react": "^19.0.0",
    "@types/react-dom": "^19.0.0",
    "@vitejs/plugin-react": "^5.0.4",
    "jsdom": "^25.0.1",
    "tailwindcss": "^4.1.14",
    "typescript": "~5.8.2",
    "vite": "^6.2.3",
    "vitest": "^2.1.8"
  }
}
```

- [ ] **Step 2: Create `tsconfig.json`**

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "useDefineForClassFields": true,
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "types": ["vitest/globals", "@testing-library/jest-dom"]
  },
  "include": ["src"]
}
```

- [ ] **Step 3: Create `vite.config.ts`**

```ts
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  plugins: [react(), tailwindcss()],
});
```

- [ ] **Step 4: Create `vitest.config.ts`**

```ts
import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/setupTests.ts'],
  },
});
```

- [ ] **Step 5: Create `src/setupTests.ts`**

```ts
import '@testing-library/jest-dom/vitest';
```

- [ ] **Step 6: Create `index.html`**

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>InvenioAI</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

- [ ] **Step 7: Create `.env.example`**

```
VITE_API_BASE_URL=http://localhost:8000
```

- [ ] **Step 8: Create `src/index.css`**

```css
@import "tailwindcss";
```

- [ ] **Step 9: Create `src/App.tsx`**

```tsx
export default function App() {
  return <div>InvenioAI</div>;
}
```

- [ ] **Step 10: Create `src/main.tsx`**

```tsx
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './index.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
```

- [ ] **Step 11: Install and verify build**

Run: `cd frontend-react && npm install && npm run build`
Expected: build succeeds, `dist/` produced.

- [ ] **Step 12: Commit**

```bash
git add frontend-react
git commit -m "chore(frontend): scaffold react + vite + tailwind app"
```

---

### Task 2: Warm-editorial theme tokens

**Files:**
- Create: `frontend-react/tailwind.config.ts`
- Modify: `frontend-react/src/index.css`

**Interfaces:**
- Produces: Tailwind tokens - colors `cream`, `charcoal`, `accent` (+ `-fg`/`-muted` scales), fonts `font-display` and `font-sans`. Components reference only these tokens.

- [ ] **Step 1: Create `tailwind.config.ts`**

```ts
import type { Config } from 'tailwindcss';

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        cream: { DEFAULT: '#f7f3ec', card: '#fffdf8', muted: '#efe8dc' },
        charcoal: { DEFAULT: '#2b2824', muted: '#6b645b', soft: '#3a3630' },
        accent: { DEFAULT: '#b4552d', soft: '#c9714b', fg: '#ffffff' },
        line: '#e4dccc',
      },
      fontFamily: {
        display: ['"Fraunces"', 'Georgia', 'serif'],
        sans: ['"Inter"', 'system-ui', 'sans-serif'],
      },
    },
  },
} satisfies Config;
```

- [ ] **Step 2: Add font imports and base styles to `src/index.css`**

```css
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Inter:wght@400;500;600&display=swap');
@import "tailwindcss";
@config "../tailwind.config.ts";

:root { color-scheme: light dark; }

body {
  @apply bg-cream text-charcoal font-sans antialiased;
}

.dark body {
  @apply bg-charcoal text-cream;
}
```

- [ ] **Step 3: Render a themed heading in `src/App.tsx` to verify tokens compile**

```tsx
export default function App() {
  return (
    <h1 className="font-display text-3xl text-accent p-8">InvenioAI</h1>
  );
}
```

- [ ] **Step 4: Verify build**

Run: `cd frontend-react && npm run build`
Expected: build succeeds with no unknown-utility errors.

- [ ] **Step 5: Commit**

```bash
git add frontend-react/tailwind.config.ts frontend-react/src/index.css frontend-react/src/App.tsx
git commit -m "feat(frontend): add warm-editorial theme tokens"
```

---

### Task 3: Types and API client

**Files:**
- Create: `frontend-react/src/types.ts`, `frontend-react/src/lib/apiClient.ts`, `frontend-react/src/lib/apiClient.test.ts`

**Interfaces:**
- Produces:
  - `types.ts`: `DocumentsResponse { documents: string[]; count: number }`, `MetricsResponse` (fields below), `Citation { file: string; page: number | null; header: string | null; score: number | null; text: string }`, `ChatMessage { id: string; role: 'user' | 'assistant'; content: string; sources?: Citation[]; thoughts?: string }`, `UploadJob { status: string; result?: { filename?: string }; error?: string }`.
  - `apiClient.ts`: `API_BASE_URL: string`, `apiFetch<T>(path: string, init?: RequestInit): Promise<T>`, `formatError(e: unknown): string`.

- [ ] **Step 1: Write the failing test `src/lib/apiClient.test.ts`**

```ts
import { describe, it, expect, vi, afterEach } from 'vitest';
import { apiFetch, formatError } from './apiClient';

afterEach(() => vi.restoreAllMocks());

describe('apiFetch', () => {
  it('returns parsed JSON on 200', async () => {
    vi.stubGlobal('fetch', vi.fn(async () =>
      new Response(JSON.stringify({ ok: true }), { status: 200 }),
    ));
    await expect(apiFetch('/documents')).resolves.toEqual({ ok: true });
  });

  it('throws with backend detail on non-2xx', async () => {
    vi.stubGlobal('fetch', vi.fn(async () =>
      new Response(JSON.stringify({ detail: 'boom' }), { status: 500 }),
    ));
    await expect(apiFetch('/documents')).rejects.toThrow('boom');
  });
});

describe('formatError', () => {
  it('maps rate limit text to a friendly message', () => {
    expect(formatError(new Error('rate_limit exceeded'))).toMatch(/rate limit/i);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend-react && npx vitest run src/lib/apiClient.test.ts`
Expected: FAIL - module not found.

- [ ] **Step 3: Create `src/types.ts`**

```ts
export interface DocumentsResponse {
  documents: string[];
  count: number;
}

export interface Citation {
  file: string;
  page: number | null;
  header: string | null;
  score: number | null;
  text: string;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: Citation[];
  thoughts?: string;
}

export interface UploadJob {
  status: string;
  result?: { filename?: string };
  error?: string;
}

export interface IRQuality {
  hit_rate: number;
  ndcg: number;
  mrr: number;
  precision: number;
  recall: number;
}

export interface MetricsResponse {
  total_queries: number;
  total_documents_indexed: number;
  avg_response_time: number;
  avg_retrieval_time: number;
  avg_generation_time: number;
  avg_docs_retrieved: number;
  ir_quality: IRQuality;
  query_history: Array<{
    question: string;
    response_time?: number;
    retrieval_time?: number;
    generation_time?: number;
    timestamp?: string;
  }>;
}
```

- [ ] **Step 4: Create `src/lib/apiClient.ts`**

```ts
export const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
).replace(/\/+$/, '');

export async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(init?.headers || {}) },
    ...init,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export function formatError(e: unknown): string {
  const msg = e instanceof Error ? e.message : String(e);
  if (/rate_?limit|quota|429/i.test(msg)) {
    return 'Groq API rate limit exceeded. Wait a moment and try again.';
  }
  if (/failed to fetch|networkerror/i.test(msg)) {
    return `Could not reach the backend at ${API_BASE_URL}.`;
  }
  return msg;
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd frontend-react && npx vitest run src/lib/apiClient.test.ts`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add frontend-react/src/types.ts frontend-react/src/lib/apiClient.ts frontend-react/src/lib/apiClient.test.ts
git commit -m "feat(frontend): add domain types and API client"
```

---

### Task 4: SSE streaming consumer

**Files:**
- Create: `frontend-react/src/lib/sse.ts`, `frontend-react/src/lib/sse.test.ts`

**Interfaces:**
- Consumes: `API_BASE_URL` from `apiClient.ts`; `Citation` from `types.ts`.
- Produces: `streamQuery(question, history, handlers): Promise<void>` where
  `handlers = { onStep?(step: string): void; onThinking?(content: string): void; onToken?(content: string): void; onDone?(payload: { answer: string; sources: Citation[]; thoughts?: string }): void; onError?(message: string): void }`. Parses `data: {json}` lines from the `POST /query/stream` body; each JSON has a `step` field (`cached|rewriting|retrieving|reranking|generating|thinking|token|done|error`).

- [ ] **Step 1: Write the failing test `src/lib/sse.test.ts`**

```ts
import { describe, it, expect, vi, afterEach } from 'vitest';
import { streamQuery } from './sse';

function sseResponse(lines: string[]): Response {
  const body = new ReadableStream({
    start(controller) {
      const enc = new TextEncoder();
      for (const l of lines) controller.enqueue(enc.encode(`data: ${l}\n\n`));
      controller.close();
    },
  });
  return new Response(body, { status: 200 });
}

afterEach(() => vi.restoreAllMocks());

describe('streamQuery', () => {
  it('dispatches tokens and a done payload', async () => {
    vi.stubGlobal('fetch', vi.fn(async () =>
      sseResponse([
        JSON.stringify({ step: 'token', content: 'Hello' }),
        JSON.stringify({ step: 'token', content: ' world' }),
        JSON.stringify({ step: 'done', answer: 'Hello world', sources: [] }),
      ]),
    ));

    const tokens: string[] = [];
    let done: { answer: string } | null = null;
    await streamQuery('q', [], {
      onToken: (c) => tokens.push(c),
      onDone: (p) => { done = p; },
    });

    expect(tokens).toEqual(['Hello', ' world']);
    expect(done!.answer).toBe('Hello world');
  });

  it('invokes onError on error step', async () => {
    vi.stubGlobal('fetch', vi.fn(async () =>
      sseResponse([JSON.stringify({ step: 'error', message: 'pipeline boom' })]),
    ));
    let err = '';
    await streamQuery('q', [], { onError: (m) => { err = m; } });
    expect(err).toBe('pipeline boom');
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend-react && npx vitest run src/lib/sse.test.ts`
Expected: FAIL - module not found.

- [ ] **Step 3: Create `src/lib/sse.ts`**

```ts
import { API_BASE_URL } from './apiClient';
import type { Citation } from '../types';

export interface StreamHandlers {
  onStep?: (step: string) => void;
  onThinking?: (content: string) => void;
  onToken?: (content: string) => void;
  onDone?: (payload: { answer: string; sources: Citation[]; thoughts?: string }) => void;
  onError?: (message: string) => void;
}

export async function streamQuery(
  question: string,
  history: string[],
  handlers: StreamHandlers,
): Promise<void> {
  const res = await fetch(`${API_BASE_URL}/query/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, history }),
  });
  if (!res.ok || !res.body) {
    handlers.onError?.(`Request failed (${res.status})`);
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const events = buffer.split('\n\n');
    buffer = events.pop() ?? '';
    for (const evt of events) {
      const line = evt.trim();
      if (!line.startsWith('data:')) continue;
      let data: Record<string, unknown>;
      try {
        data = JSON.parse(line.slice(5).trim());
      } catch {
        continue;
      }
      const step = String(data.step ?? '');
      handlers.onStep?.(step);
      if (step === 'thinking') handlers.onThinking?.(String(data.content ?? ''));
      else if (step === 'token') handlers.onToken?.(String(data.content ?? ''));
      else if (step === 'done')
        handlers.onDone?.({
          answer: String(data.answer ?? ''),
          sources: (data.sources as Citation[]) ?? [],
          thoughts: data.thoughts as string | undefined,
        });
      else if (step === 'error') handlers.onError?.(String(data.message ?? 'Unknown error'));
    }
  }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd frontend-react && npx vitest run src/lib/sse.test.ts`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add frontend-react/src/lib/sse.ts frontend-react/src/lib/sse.test.ts
git commit -m "feat(frontend): add SSE streaming consumer"
```

---

### Task 5: Router, QueryClient, and Layout shell

**Files:**
- Modify: `frontend-react/src/main.tsx`, `frontend-react/src/App.tsx`
- Create: `frontend-react/src/components/Layout.tsx`, `frontend-react/src/pages/ChatPage.tsx`, `frontend-react/src/pages/AnalyticsPage.tsx`

**Interfaces:**
- Consumes: nothing from prior tasks beyond the app shell.
- Produces: `<App/>` rendering `<Routes>`; `Layout` renders a `<aside>` sidebar placeholder + `<Outlet/>`. Routes: `/` redirects to `/chat`; `/chat` -> `ChatPage`; `/analytics` -> `AnalyticsPage`.

- [ ] **Step 1: Replace `src/main.tsx`**

```tsx
import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import App from './App';
import './index.css';

const queryClient = new QueryClient({
  defaultOptions: { queries: { retry: 1, refetchOnWindowFocus: false } },
});

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </QueryClientProvider>
  </React.StrictMode>,
);
```

- [ ] **Step 2: Replace `src/App.tsx`**

```tsx
import { Navigate, Route, Routes } from 'react-router-dom';
import Layout from './components/Layout';
import ChatPage from './pages/ChatPage';
import AnalyticsPage from './pages/AnalyticsPage';

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Navigate to="/chat" replace />} />
        <Route path="/chat" element={<ChatPage />} />
        <Route path="/analytics" element={<AnalyticsPage />} />
      </Route>
    </Routes>
  );
}
```

- [ ] **Step 3: Create `src/components/Layout.tsx`**

```tsx
import { NavLink, Outlet } from 'react-router-dom';
import { MessageSquare, BarChart3 } from 'lucide-react';

const navClass = ({ isActive }: { isActive: boolean }) =>
  `flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
    isActive ? 'bg-accent text-accent-fg' : 'text-charcoal-muted hover:bg-cream-muted'
  }`;

export default function Layout() {
  return (
    <div className="min-h-screen flex bg-cream text-charcoal">
      <aside className="w-80 shrink-0 border-r border-line bg-cream-card flex flex-col p-5 gap-6">
        <div>
          <h1 className="font-display text-2xl font-bold text-accent">InvenioAI</h1>
          <p className="text-xs text-charcoal-muted">Document Intelligence</p>
        </div>
        <nav className="flex flex-col gap-1">
          <NavLink to="/chat" className={navClass}>
            <MessageSquare className="w-4 h-4" /> Chat
          </NavLink>
          <NavLink to="/analytics" className={navClass}>
            <BarChart3 className="w-4 h-4" /> Analytics
          </NavLink>
        </nav>
        <div id="sidebar-slot" className="flex-1 min-h-0 overflow-y-auto" />
      </aside>
      <main className="flex-1 min-w-0 flex flex-col">
        <Outlet />
      </main>
    </div>
  );
}
```

- [ ] **Step 4: Create `src/pages/ChatPage.tsx`**

```tsx
export default function ChatPage() {
  return <div className="p-8 font-display text-xl">Chat</div>;
}
```

- [ ] **Step 5: Create `src/pages/AnalyticsPage.tsx`**

```tsx
export default function AnalyticsPage() {
  return <div className="p-8 font-display text-xl">Analytics</div>;
}
```

- [ ] **Step 6: Verify build and manual route check**

Run: `cd frontend-react && npm run build && npm run dev`
Expected: build passes; visiting `/` redirects to `/chat`; `/analytics` renders "Analytics".

- [ ] **Step 7: Commit**

```bash
git add frontend-react/src
git commit -m "feat(frontend): add router, query client, and layout shell"
```

---

### Task 6: Documents and metrics query hooks

**Files:**
- Create: `frontend-react/src/hooks/useDocuments.ts`, `frontend-react/src/hooks/useMetrics.ts`, `frontend-react/src/hooks/useDocuments.test.tsx`

**Interfaces:**
- Consumes: `apiFetch` from `apiClient.ts`; `DocumentsResponse`, `MetricsResponse` from `types.ts`.
- Produces:
  - `useDocuments()` -> `UseQueryResult<DocumentsResponse>` (queryKey `['documents']`).
  - `useDeleteDocument()` -> mutation `(filename: string) => Promise<void>`, invalidates `['documents']`.
  - `useDeleteAllDocuments()` -> mutation `() => Promise<void>`, invalidates `['documents']` and `['metrics']`.
  - `useMetrics()` -> `UseQueryResult<MetricsResponse>` (queryKey `['metrics']`).

- [ ] **Step 1: Write the failing test `src/hooks/useDocuments.test.tsx`**

```tsx
import { describe, it, expect, vi, afterEach } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import React from 'react';
import { useDocuments } from './useDocuments';

afterEach(() => vi.restoreAllMocks());

function wrapper({ children }: { children: React.ReactNode }) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

describe('useDocuments', () => {
  it('fetches the documents list', async () => {
    vi.stubGlobal('fetch', vi.fn(async () =>
      new Response(JSON.stringify({ documents: ['a.pdf'], count: 1 }), { status: 200 }),
    ));
    const { result } = renderHook(() => useDocuments(), { wrapper });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data?.documents).toEqual(['a.pdf']);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend-react && npx vitest run src/hooks/useDocuments.test.tsx`
Expected: FAIL - module not found.

- [ ] **Step 3: Create `src/hooks/useDocuments.ts`**

```ts
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { apiFetch } from '../lib/apiClient';
import type { DocumentsResponse } from '../types';

export function useDocuments() {
  return useQuery({
    queryKey: ['documents'],
    queryFn: () => apiFetch<DocumentsResponse>('/documents'),
  });
}

export function useDeleteDocument() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (filename: string) =>
      apiFetch<unknown>(`/documents/delete?filename=${encodeURIComponent(filename)}`, {
        method: 'DELETE',
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['documents'] }),
  });
}

export function useDeleteAllDocuments() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => apiFetch<unknown>('/documents', { method: 'DELETE' }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['documents'] });
      qc.invalidateQueries({ queryKey: ['metrics'] });
    },
  });
}
```

- [ ] **Step 4: Create `src/hooks/useMetrics.ts`**

```ts
import { useQuery } from '@tanstack/react-query';
import { apiFetch } from '../lib/apiClient';
import type { MetricsResponse } from '../types';

export function useMetrics() {
  return useQuery({
    queryKey: ['metrics'],
    queryFn: () => apiFetch<MetricsResponse>('/metrics'),
    refetchInterval: 15000,
  });
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd frontend-react && npx vitest run src/hooks/useDocuments.test.tsx`
Expected: PASS (1 test).

- [ ] **Step 6: Commit**

```bash
git add frontend-react/src/hooks/useDocuments.ts frontend-react/src/hooks/useMetrics.ts frontend-react/src/hooks/useDocuments.test.tsx
git commit -m "feat(frontend): add documents and metrics query hooks"
```

---

### Task 7: Upload job hook with polling backoff

**Files:**
- Create: `frontend-react/src/hooks/useUploadJob.ts`, `frontend-react/src/hooks/useUploadJob.test.ts`

**Interfaces:**
- Consumes: `API_BASE_URL` from `apiClient.ts`; `UploadJob` from `types.ts`.
- Produces: `nextPollInterval(current: number): number` (1000 -> 2000 -> ... capped 5000) and `pollUploadJob(jobId, { onStatus, signal }): Promise<{ ok: boolean; message: string }>`, and a `useUploadJob()` hook exposing `upload(file: File): Promise<void>`, `status: string | null`, `isUploading: boolean`. Terminal states: `succeeded`, `failed`.

- [ ] **Step 1: Write the failing test `src/hooks/useUploadJob.test.ts`**

```ts
import { describe, it, expect } from 'vitest';
import { nextPollInterval } from './useUploadJob';

describe('nextPollInterval', () => {
  it('doubles then caps at 5000ms', () => {
    expect(nextPollInterval(1000)).toBe(2000);
    expect(nextPollInterval(2000)).toBe(4000);
    expect(nextPollInterval(4000)).toBe(5000);
    expect(nextPollInterval(5000)).toBe(5000);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend-react && npx vitest run src/hooks/useUploadJob.test.ts`
Expected: FAIL - module not found.

- [ ] **Step 3: Create `src/hooks/useUploadJob.ts`**

```ts
import { useCallback, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { API_BASE_URL, formatError } from '../lib/apiClient';
import type { UploadJob } from '../types';

export function nextPollInterval(current: number): number {
  return Math.min(current * 2, 5000);
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function createUploadJob(file: File): Promise<string> {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(`${API_BASE_URL}/upload/jobs`, { method: 'POST', body: form });
  if (!res.ok) throw new Error(`Upload failed (${res.status})`);
  const body = await res.json();
  if (!body.job_id) throw new Error('Missing job id from backend');
  return String(body.job_id);
}

async function fetchJob(jobId: string): Promise<UploadJob> {
  const res = await fetch(`${API_BASE_URL}/upload/jobs/${jobId}`);
  if (!res.ok) throw new Error(`Job status failed (${res.status})`);
  return res.json();
}

export function useUploadJob() {
  const qc = useQueryClient();
  const [status, setStatus] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);

  const upload = useCallback(
    async (file: File) => {
      setIsUploading(true);
      setStatus('uploading');
      try {
        const jobId = await createUploadJob(file);
        let interval = 1000;
        while (true) {
          const job = await fetchJob(jobId);
          setStatus(job.status);
          if (job.status === 'succeeded') {
            qc.invalidateQueries({ queryKey: ['documents'] });
            qc.invalidateQueries({ queryKey: ['metrics'] });
            break;
          }
          if (job.status === 'failed') {
            throw new Error(job.error || 'Indexing failed');
          }
          await sleep(interval);
          interval = nextPollInterval(interval);
        }
      } catch (e) {
        setStatus('failed');
        throw new Error(formatError(e));
      } finally {
        setIsUploading(false);
      }
    },
    [qc],
  );

  return { upload, status, isUploading };
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd frontend-react && npx vitest run src/hooks/useUploadJob.test.ts`
Expected: PASS (1 test).

- [ ] **Step 5: Commit**

```bash
git add frontend-react/src/hooks/useUploadJob.ts frontend-react/src/hooks/useUploadJob.test.ts
git commit -m "feat(frontend): add upload job hook with polling backoff"
```

---

### Task 8: Chat hook (localStorage + streaming)

**Files:**
- Create: `frontend-react/src/hooks/useChat.ts`, `frontend-react/src/hooks/useChat.test.tsx`

**Interfaces:**
- Consumes: `streamQuery` from `lib/sse.ts`; `ChatMessage` from `types.ts`.
- Produces: `useChat()` -> `{ messages: ChatMessage[]; isGenerating: boolean; send(text: string): Promise<void>; clear(): void }`. Persists `messages` to `localStorage` key `invenioai_chat`. `send` appends a user message, streams the assistant reply (tokens accumulate into the last assistant message), and finalizes with `sources`/`thoughts` from `onDone`.

- [ ] **Step 1: Write the failing test `src/hooks/useChat.test.tsx`**

```tsx
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';
import * as sse from '../lib/sse';
import { useChat } from './useChat';

beforeEach(() => localStorage.clear());
afterEach(() => vi.restoreAllMocks());

describe('useChat', () => {
  it('appends user + assistant messages and persists to localStorage', async () => {
    vi.spyOn(sse, 'streamQuery').mockImplementation(async (_q, _h, handlers) => {
      handlers.onToken?.('Hi');
      handlers.onDone?.({ answer: 'Hi there', sources: [], thoughts: undefined });
    });

    const { result } = renderHook(() => useChat());
    await act(async () => {
      await result.current.send('hello');
    });

    await waitFor(() => expect(result.current.messages.length).toBe(2));
    expect(result.current.messages[0].role).toBe('user');
    expect(result.current.messages[1].content).toBe('Hi there');
    expect(localStorage.getItem('invenioai_chat')).toContain('Hi there');
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend-react && npx vitest run src/hooks/useChat.test.tsx`
Expected: FAIL - module not found.

- [ ] **Step 3: Create `src/hooks/useChat.ts`**

```ts
import { useCallback, useEffect, useRef, useState } from 'react';
import { streamQuery } from '../lib/sse';
import type { ChatMessage } from '../types';

const STORAGE_KEY = 'invenioai_chat';

function load(): ChatMessage[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as ChatMessage[]) : [];
  } catch {
    return [];
  }
}

export function useChat() {
  const [messages, setMessages] = useState<ChatMessage[]>(load);
  const [isGenerating, setIsGenerating] = useState(false);
  const messagesRef = useRef(messages);
  messagesRef.current = messages;

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
  }, [messages]);

  const send = useCallback(async (text: string) => {
    const trimmed = text.trim();
    if (!trimmed || isGenerating) return;

    const userMsg: ChatMessage = { id: `u-${Date.now()}`, role: 'user', content: trimmed };
    const assistantId = `a-${Date.now()}`;
    const assistantMsg: ChatMessage = { id: assistantId, role: 'assistant', content: '' };

    const history = messagesRef.current.map((m) => `${m.role}: ${m.content}`);
    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setIsGenerating(true);

    const patch = (fn: (m: ChatMessage) => ChatMessage) =>
      setMessages((prev) => prev.map((m) => (m.id === assistantId ? fn(m) : m)));

    try {
      await streamQuery(trimmed, history, {
        onToken: (c) => patch((m) => ({ ...m, content: m.content + c })),
        onDone: (p) =>
          patch((m) => ({ ...m, content: p.answer, sources: p.sources, thoughts: p.thoughts })),
        onError: (msg) => patch((m) => ({ ...m, content: `Error: ${msg}` })),
      });
    } finally {
      setIsGenerating(false);
    }
  }, [isGenerating]);

  const clear = useCallback(() => setMessages([]), []);

  return { messages, isGenerating, send, clear };
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd frontend-react && npx vitest run src/hooks/useChat.test.tsx`
Expected: PASS (1 test).

- [ ] **Step 5: Commit**

```bash
git add frontend-react/src/hooks/useChat.ts frontend-react/src/hooks/useChat.test.tsx
git commit -m "feat(frontend): add chat hook with localStorage and streaming"
```

---

### Task 9: ThemeToggle + ConfirmDialog primitives

**Files:**
- Create: `frontend-react/src/components/ThemeToggle.tsx`, `frontend-react/src/components/ConfirmDialog.tsx`

**Interfaces:**
- Produces:
  - `ThemeToggle()` - toggles `document.documentElement.classList` `dark`, persists to `localStorage` `invenio_theme`.
  - `ConfirmDialog({ open, title, message, confirmLabel, onConfirm, onCancel })` - modal overlay; renders nothing when `open` is false.

- [ ] **Step 1: Create `src/components/ThemeToggle.tsx`**

```tsx
import { useEffect, useState } from 'react';
import { Moon, Sun } from 'lucide-react';

export default function ThemeToggle() {
  const [dark, setDark] = useState(
    () => localStorage.getItem('invenio_theme') === 'dark',
  );

  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark);
    localStorage.setItem('invenio_theme', dark ? 'dark' : 'light');
  }, [dark]);

  return (
    <button
      onClick={() => setDark((v) => !v)}
      aria-label="Toggle theme"
      className="p-2 rounded-lg text-charcoal-muted hover:bg-cream-muted"
    >
      {dark ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
    </button>
  );
}
```

- [ ] **Step 2: Create `src/components/ConfirmDialog.tsx`**

```tsx
interface Props {
  open: boolean;
  title: string;
  message: string;
  confirmLabel?: string;
  onConfirm: () => void;
  onCancel: () => void;
}

export default function ConfirmDialog({
  open, title, message, confirmLabel = 'Confirm', onConfirm, onCancel,
}: Props) {
  if (!open) return null;
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-charcoal/40 backdrop-blur-sm"
      onClick={onCancel}
    >
      <div
        className="w-full max-w-sm rounded-2xl bg-cream-card border border-line p-6 shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <h3 className="font-display text-lg font-bold">{title}</h3>
        <p className="text-sm text-charcoal-muted mt-2">{message}</p>
        <div className="flex justify-end gap-2 mt-5">
          <button onClick={onCancel} className="px-4 py-2 rounded-lg text-sm hover:bg-cream-muted">
            Cancel
          </button>
          <button
            onClick={onConfirm}
            className="px-4 py-2 rounded-lg text-sm bg-accent text-accent-fg hover:bg-accent-soft"
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Verify build**

Run: `cd frontend-react && npm run build`
Expected: build succeeds.

- [ ] **Step 4: Commit**

```bash
git add frontend-react/src/components/ThemeToggle.tsx frontend-react/src/components/ConfirmDialog.tsx
git commit -m "feat(frontend): add theme toggle and confirm dialog"
```

---

### Task 10: UploadPanel + KnowledgeBaseList + Sidebar wiring

**Files:**
- Create: `frontend-react/src/components/UploadPanel.tsx`, `frontend-react/src/components/KnowledgeBaseList.tsx`, `frontend-react/src/components/Sidebar.tsx`
- Modify: `frontend-react/src/components/Layout.tsx`

**Interfaces:**
- Consumes: `useUploadJob`, `useDocuments`, `useDeleteDocument`, `useDeleteAllDocuments`, `ConfirmDialog`, `ThemeToggle`.
- Produces: `Sidebar()` composing brand + `ThemeToggle` + `UploadPanel` + `KnowledgeBaseList`. `Layout` renders `<Sidebar/>` in place of the placeholder nav block (keeps the `<NavLink>` nav).

- [ ] **Step 1: Create `src/components/UploadPanel.tsx`**

```tsx
import { useRef, useState } from 'react';
import { UploadCloud } from 'lucide-react';
import { useUploadJob } from '../hooks/useUploadJob';

export default function UploadPanel() {
  const { upload, status, isUploading } = useUploadJob();
  const inputRef = useRef<HTMLInputElement>(null);
  const [error, setError] = useState<string | null>(null);

  const onFile = async (file: File | undefined) => {
    if (!file) return;
    setError(null);
    try {
      await upload(file);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Upload failed');
    }
  };

  return (
    <div>
      <h2 className="font-display text-sm font-semibold mb-2">Upload PDF</h2>
      <button
        onClick={() => inputRef.current?.click()}
        disabled={isUploading}
        className="w-full flex items-center justify-center gap-2 border border-dashed border-line rounded-xl py-4 text-sm text-charcoal-muted hover:bg-cream-muted disabled:opacity-60"
      >
        <UploadCloud className="w-4 h-4" />
        {isUploading ? `Indexing... (${status})` : 'Add a document'}
      </button>
      <input
        ref={inputRef}
        type="file"
        accept="application/pdf"
        className="hidden"
        onChange={(e) => onFile(e.target.files?.[0])}
      />
      {error && <p className="text-xs text-accent mt-2">{error}</p>}
    </div>
  );
}
```

- [ ] **Step 2: Create `src/components/KnowledgeBaseList.tsx`**

```tsx
import { useState } from 'react';
import { FileText, Trash2 } from 'lucide-react';
import { useDocuments, useDeleteDocument, useDeleteAllDocuments } from '../hooks/useDocuments';
import ConfirmDialog from './ConfirmDialog';

export default function KnowledgeBaseList() {
  const { data, isLoading, isError } = useDocuments();
  const deleteOne = useDeleteDocument();
  const deleteAll = useDeleteAllDocuments();
  const [confirmAll, setConfirmAll] = useState(false);

  const docs = data?.documents ?? [];

  return (
    <div>
      <h2 className="font-display text-sm font-semibold mb-2">Knowledge Base</h2>
      {isLoading && <p className="text-xs text-charcoal-muted">Loading...</p>}
      {isError && <p className="text-xs text-accent">Cannot reach backend.</p>}
      {!isLoading && !isError && docs.length === 0 && (
        <p className="text-xs text-charcoal-muted">No documents yet.</p>
      )}
      <ul className="flex flex-col gap-1">
        {docs.map((name) => (
          <li key={name} className="flex items-center gap-2 text-sm group">
            <FileText className="w-4 h-4 text-charcoal-muted shrink-0" />
            <span className="truncate flex-1" title={name}>{name}</span>
            <button
              onClick={() => deleteOne.mutate(name)}
              aria-label={`Delete ${name}`}
              className="opacity-0 group-hover:opacity-100 text-charcoal-muted hover:text-accent"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          </li>
        ))}
      </ul>
      {docs.length > 0 && (
        <button
          onClick={() => setConfirmAll(true)}
          className="mt-3 text-xs text-accent hover:underline"
        >
          Delete all documents
        </button>
      )}
      <ConfirmDialog
        open={confirmAll}
        title="Delete all documents?"
        message="This permanently deletes all indexed documents and cannot be undone."
        confirmLabel="Delete all"
        onConfirm={() => {
          deleteAll.mutate();
          setConfirmAll(false);
        }}
        onCancel={() => setConfirmAll(false)}
      />
    </div>
  );
}
```

- [ ] **Step 3: Create `src/components/Sidebar.tsx`**

```tsx
import UploadPanel from './UploadPanel';
import KnowledgeBaseList from './KnowledgeBaseList';

export default function Sidebar() {
  return (
    <div className="flex flex-col gap-6">
      <UploadPanel />
      <div className="border-t border-line pt-5">
        <KnowledgeBaseList />
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Wire `Sidebar` + `ThemeToggle` into `Layout.tsx`**

Replace the `<div id="sidebar-slot" .../>` line with `<div className="flex-1 min-h-0 overflow-y-auto"><Sidebar /></div>`, add `import Sidebar from './Sidebar';` and `import ThemeToggle from './ThemeToggle';`, and place `<ThemeToggle />` in the brand header row. Final `Layout.tsx`:

```tsx
import { NavLink, Outlet } from 'react-router-dom';
import { MessageSquare, BarChart3 } from 'lucide-react';
import Sidebar from './Sidebar';
import ThemeToggle from './ThemeToggle';

const navClass = ({ isActive }: { isActive: boolean }) =>
  `flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
    isActive ? 'bg-accent text-accent-fg' : 'text-charcoal-muted hover:bg-cream-muted'
  }`;

export default function Layout() {
  return (
    <div className="min-h-screen flex bg-cream text-charcoal dark:bg-charcoal dark:text-cream">
      <aside className="w-80 shrink-0 border-r border-line bg-cream-card flex flex-col p-5 gap-6">
        <div className="flex items-start justify-between">
          <div>
            <h1 className="font-display text-2xl font-bold text-accent">InvenioAI</h1>
            <p className="text-xs text-charcoal-muted">Document Intelligence</p>
          </div>
          <ThemeToggle />
        </div>
        <nav className="flex flex-col gap-1">
          <NavLink to="/chat" className={navClass}>
            <MessageSquare className="w-4 h-4" /> Chat
          </NavLink>
          <NavLink to="/analytics" className={navClass}>
            <BarChart3 className="w-4 h-4" /> Analytics
          </NavLink>
        </nav>
        <div className="flex-1 min-h-0 overflow-y-auto">
          <Sidebar />
        </div>
      </aside>
      <main className="flex-1 min-w-0 flex flex-col">
        <Outlet />
      </main>
    </div>
  );
}
```

- [ ] **Step 5: Verify build + manual check**

Run: `cd frontend-react && npm run build && npm run dev`
Expected: sidebar shows upload area + knowledge base list (with backend running, real documents appear).

- [ ] **Step 6: Commit**

```bash
git add frontend-react/src/components
git commit -m "feat(frontend): wire upload panel and knowledge base into sidebar"
```

---

### Task 11: ChatMessage + SourcesPanel

**Files:**
- Create: `frontend-react/src/components/ChatMessage.tsx`, `frontend-react/src/components/SourcesPanel.tsx`

**Interfaces:**
- Consumes: `ChatMessage`, `Citation` types.
- Produces:
  - `ChatMessage({ message })` - renders a bubble; assistant messages with `thoughts` show a collapsible "Thought Process"; renders `content` as text.
  - `SourcesPanel({ citations })` - renders grouped citation cards (`id="source-card-<file>-<page>"`), each showing file, page, header, relevance score, snippet. Empty state when no citations.

- [ ] **Step 1: Create `src/components/ChatMessage.tsx`**

```tsx
import { useState } from 'react';
import { Brain, ChevronDown } from 'lucide-react';
import type { ChatMessage as Msg } from '../types';

export default function ChatMessage({ message }: { message: Msg }) {
  const [showThoughts, setShowThoughts] = useState(false);
  const isUser = message.role === 'user';

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`max-w-2xl rounded-2xl px-4 py-3 ${
          isUser ? 'bg-accent text-accent-fg' : 'bg-cream-card border border-line'
        }`}
      >
        {!isUser && message.thoughts && (
          <div className="mb-2">
            <button
              onClick={() => setShowThoughts((v) => !v)}
              className="flex items-center gap-1.5 text-xs text-charcoal-muted hover:text-charcoal"
            >
              <Brain className="w-3.5 h-3.5" /> Thought Process
              <ChevronDown className={`w-3.5 h-3.5 transition-transform ${showThoughts ? 'rotate-180' : ''}`} />
            </button>
            {showThoughts && (
              <pre className="mt-2 text-xs whitespace-pre-wrap text-charcoal-muted bg-cream-muted rounded-lg p-3">
                {message.thoughts}
              </pre>
            )}
          </div>
        )}
        <p className="whitespace-pre-wrap text-sm leading-relaxed">{message.content}</p>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Create `src/components/SourcesPanel.tsx`**

```tsx
import { FileText } from 'lucide-react';
import type { Citation } from '../types';

export default function SourcesPanel({ citations }: { citations: Citation[] }) {
  return (
    <div className="h-full overflow-y-auto border-l border-line bg-cream-card/50 p-5">
      <h2 className="font-display text-sm font-semibold mb-3">Sources</h2>
      {citations.length === 0 && (
        <p className="text-xs text-charcoal-muted">Sources appear here after you ask a question.</p>
      )}
      <div className="flex flex-col gap-3">
        {citations.map((c, i) => (
          <div
            key={`${c.file}-${c.page}-${i}`}
            id={`source-card-${c.file}-${c.page}`}
            className="rounded-xl border border-line bg-cream-card p-3"
          >
            <div className="flex items-center gap-2 text-xs font-medium">
              <FileText className="w-3.5 h-3.5 text-accent shrink-0" />
              <span className="truncate" title={c.file}>{c.file}</span>
            </div>
            <div className="text-[11px] text-charcoal-muted mt-1 flex flex-wrap gap-x-2">
              {c.page != null && <span>Page {c.page}</span>}
              {c.header && <span className="italic">{c.header}</span>}
              {typeof c.score === 'number' && <span>Relevance {c.score.toFixed(2)}</span>}
            </div>
            <p className="text-xs text-charcoal-muted mt-2 line-clamp-4">{c.text}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Verify build**

Run: `cd frontend-react && npm run build`
Expected: build succeeds.

- [ ] **Step 4: Commit**

```bash
git add frontend-react/src/components/ChatMessage.tsx frontend-react/src/components/SourcesPanel.tsx
git commit -m "feat(frontend): add chat message and sources panel components"
```

---

### Task 12: ChatPanel + ChatPage assembly

**Files:**
- Create: `frontend-react/src/components/ChatPanel.tsx`
- Modify: `frontend-react/src/pages/ChatPage.tsx`

**Interfaces:**
- Consumes: `useChat`, `useDocuments`, `ChatMessage` component, `SourcesPanel`.
- Produces: `ChatPanel()` - message feed + input box, auto-scroll on new content, disabled input while generating, empty welcome state. `ChatPage` renders `ChatPanel` (flex-1) + `SourcesPanel` (fixed right column, fed by the latest assistant message's `sources`).

- [ ] **Step 1: Create `src/components/ChatPanel.tsx`**

```tsx
import { useEffect, useRef, useState } from 'react';
import { Send } from 'lucide-react';
import ChatMessageView from './ChatMessage';
import type { ChatMessage } from '../types';

interface Props {
  messages: ChatMessage[];
  isGenerating: boolean;
  hasDocuments: boolean;
  onSend: (text: string) => void;
}

export default function ChatPanel({ messages, isGenerating, hasDocuments, onSend }: Props) {
  const [input, setInput] = useState('');
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const submit = () => {
    const text = input.trim();
    if (!text || isGenerating) return;
    onSend(text);
    setInput('');
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-y-auto p-6 space-y-4">
        {messages.length === 0 && (
          <div className="h-full flex flex-col items-center justify-center text-center">
            <h2 className="font-display text-4xl font-bold text-accent">InvenioAI</h2>
            <p className="text-charcoal-muted mt-2">Ask anything about your knowledge base.</p>
          </div>
        )}
        {messages.map((m) => (
          <ChatMessageView key={m.id} message={m} />
        ))}
        <div ref={endRef} />
      </div>
      <div className="border-t border-line p-4">
        {!hasDocuments && (
          <p className="text-xs text-accent mb-2">Upload a PDF first to start asking questions.</p>
        )}
        <div className="flex items-end gap-2">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                submit();
              }
            }}
            rows={1}
            placeholder="Ask something about your documents..."
            className="flex-1 resize-none rounded-xl border border-line bg-cream-card px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-accent/40"
          />
          <button
            onClick={submit}
            disabled={isGenerating}
            className="p-3 rounded-xl bg-accent text-accent-fg hover:bg-accent-soft disabled:opacity-60"
            aria-label="Send"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Replace `src/pages/ChatPage.tsx`**

```tsx
import { useChat } from '../hooks/useChat';
import { useDocuments } from '../hooks/useDocuments';
import ChatPanel from '../components/ChatPanel';
import SourcesPanel from '../components/SourcesPanel';

export default function ChatPage() {
  const { messages, isGenerating, send } = useChat();
  const { data } = useDocuments();

  const lastAssistant = [...messages].reverse().find(
    (m) => m.role === 'assistant' && m.sources && m.sources.length > 0,
  );

  return (
    <div className="flex h-screen">
      <div className="flex-1 min-w-0">
        <ChatPanel
          messages={messages}
          isGenerating={isGenerating}
          hasDocuments={(data?.count ?? 0) > 0}
          onSend={send}
        />
      </div>
      <div className="w-96 shrink-0">
        <SourcesPanel citations={lastAssistant?.sources ?? []} />
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Verify build + manual E2E check**

Run: `cd frontend-react && npm run build && npm run dev` (backend running on `VITE_API_BASE_URL`)
Expected: upload a PDF, ask a question, tokens stream into the assistant bubble, sources populate the right panel, thought process collapsible works.

- [ ] **Step 4: Commit**

```bash
git add frontend-react/src/components/ChatPanel.tsx frontend-react/src/pages/ChatPage.tsx
git commit -m "feat(frontend): assemble chat page with streaming and sources"
```

---

### Task 13: MetricsDashboard + AnalyticsPage

**Files:**
- Create: `frontend-react/src/components/MetricsDashboard.tsx`
- Modify: `frontend-react/src/pages/AnalyticsPage.tsx`

**Interfaces:**
- Consumes: `useMetrics`, `MetricsResponse`, recharts.
- Produces: `MetricsDashboard({ metrics })` - KPI cards (HitRate, nDCG, avg response, indexed docs), a response-time line chart from `query_history`, and an advanced-metrics section (Precision/Recall/MRR). `AnalyticsPage` wires `useMetrics` with loading/empty/error states.

- [ ] **Step 1: Create `src/components/MetricsDashboard.tsx`**

```tsx
import {
  Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis, CartesianGrid,
} from 'recharts';
import type { MetricsResponse } from '../types';

function pct(v: number | undefined): string {
  return v == null ? '-' : `${(v * 100).toFixed(1)}%`;
}

function Card({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl border border-line bg-cream-card p-4">
      <p className="text-xs text-charcoal-muted">{label}</p>
      <p className="font-display text-2xl font-bold mt-1">{value}</p>
    </div>
  );
}

export default function MetricsDashboard({ metrics }: { metrics: MetricsResponse }) {
  const ir = metrics.ir_quality;
  const chartData = metrics.query_history.map((q, i) => ({
    name: `Q${i + 1}`,
    Total: q.response_time ?? 0,
    Retrieval: q.retrieval_time ?? 0,
    Generation: q.generation_time ?? 0,
  }));

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <Card label="HitRate@k" value={pct(ir?.hit_rate)} />
        <Card label="nDCG@k" value={pct(ir?.ndcg)} />
        <Card label="Avg Response" value={`${metrics.avg_response_time.toFixed(2)}s`} />
        <Card label="Indexed Docs" value={String(metrics.total_documents_indexed)} />
      </div>

      <div className="rounded-2xl border border-line bg-cream-card p-4">
        <h3 className="font-display text-sm font-semibold mb-3">Response Time Trend</h3>
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e4dccc" />
            <XAxis dataKey="name" fontSize={12} />
            <YAxis fontSize={12} unit="s" />
            <Tooltip />
            <Line type="monotone" dataKey="Total" stroke="#b4552d" strokeWidth={2} />
            <Line type="monotone" dataKey="Retrieval" stroke="#6b645b" strokeWidth={2} />
            <Line type="monotone" dataKey="Generation" stroke="#c9714b" strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <Card label="Precision@k" value={pct(ir?.precision)} />
        <Card label="Recall@k" value={pct(ir?.recall)} />
        <Card label="MRR" value={pct(ir?.mrr)} />
        <Card label="Avg Retrieval" value={`${metrics.avg_retrieval_time.toFixed(2)}s`} />
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Replace `src/pages/AnalyticsPage.tsx`**

```tsx
import { useMetrics } from '../hooks/useMetrics';
import MetricsDashboard from '../components/MetricsDashboard';

export default function AnalyticsPage() {
  const { data, isLoading, isError } = useMetrics();

  return (
    <div className="h-screen overflow-y-auto p-8">
      <div className="max-w-4xl mx-auto">
        <h2 className="font-display text-2xl font-bold mb-1">Analytics</h2>
        <p className="text-sm text-charcoal-muted mb-6">
          RAG retrieval quality and latency.
        </p>
        {isLoading && <p className="text-sm text-charcoal-muted">Loading metrics...</p>}
        {isError && <p className="text-sm text-accent">Could not load metrics.</p>}
        {data && data.total_queries === 0 && (
          <p className="text-sm text-charcoal-muted">
            No queries recorded yet. Ask a question on the Chat page.
          </p>
        )}
        {data && data.total_queries > 0 && <MetricsDashboard metrics={data} />}
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Verify build + manual check**

Run: `cd frontend-react && npm run build && npm run dev`
Expected: after asking some questions, `/analytics` shows KPI cards, a populated line chart, and advanced metric cards.

- [ ] **Step 4: Commit**

```bash
git add frontend-react/src/components/MetricsDashboard.tsx frontend-react/src/pages/AnalyticsPage.tsx
git commit -m "feat(frontend): add analytics dashboard"
```

---

### Task 14: Backend-only container image

**Files:**
- Create: `backend/Dockerfile.api`

**Interfaces:**
- Produces: a Docker image that runs only `uvicorn app.main:app` (no Streamlit), reading `PORT` (default 8000).

- [ ] **Step 1: Create `backend/Dockerfile.api`**

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./

ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
```

- [ ] **Step 2: Build the image to verify it is valid**

Run: `docker build -f backend/Dockerfile.api -t invenioai-api .`
Expected: image builds successfully.

- [ ] **Step 3: Smoke-test the container**

Run: `docker run --rm -e GROQ_API_KEY=dummy -p 8000:8000 invenioai-api` then in another shell `curl localhost:8000/`
Expected: `{"status":"InvenioAI API running"}`.

- [ ] **Step 4: Commit**

```bash
git add backend/Dockerfile.api
git commit -m "feat(backend): add backend-only container image"
```

---

### Task 15: Deployment docs (Azure + Vercel)

**Files:**
- Create: `frontend-react/DEPLOY.md`

**Interfaces:**
- Produces: step-by-step deploy notes. No code.

- [ ] **Step 1: Create `frontend-react/DEPLOY.md`**

````markdown
# Deployment

## Backend - Azure Container Apps

1. Build & push image:
   ```bash
   az acr build --registry <registry> --image invenioai-api:latest \
     --file backend/Dockerfile.api .
   ```
2. Create/update the Container App with env + secrets:
   - Secrets: `GROQ_API_KEY`, `QDRANT_URL` (and `QDRANT_API_KEY` if used).
   - Env: `INVENIOAI_ALLOWED_ORIGINS=https://<your-app>.vercel.app`
   - Do NOT set `INVENIOAI_API_KEY` (public, no-auth deployment).
   - Target port: 8000. Enable ingress (external).
3. Note the app FQDN, e.g. `https://invenioai-api.<region>.azurecontainerapps.io`.

## Frontend - Vercel

1. Import the repo in Vercel. Set **Root Directory** to `frontend-react`.
2. Framework preset: Vite. Build command `npm run build`, output `dist`.
3. Environment variable:
   - `VITE_API_BASE_URL=https://invenioai-api.<region>.azurecontainerapps.io`
4. Deploy. After the first deploy, copy the production domain and set it as
   `INVENIOAI_ALLOWED_ORIGINS` on the Azure Container App, then redeploy the backend.

## Old Streamlit HF Space

Left running, untouched, as an archived demo. Not maintained.
````

- [ ] **Step 2: Commit**

```bash
git add frontend-react/DEPLOY.md
git commit -m "docs(frontend): add azure + vercel deployment guide"
```

---

### Task 16: Full test + lint sweep and final verification

**Files:** none (verification only).

- [ ] **Step 1: Run the full frontend test suite**

Run: `cd frontend-react && npm run test`
Expected: all tests pass (apiClient, sse, useDocuments, useUploadJob, useChat).

- [ ] **Step 2: Run the type check / lint**

Run: `cd frontend-react && npm run lint`
Expected: no TypeScript errors.

- [ ] **Step 3: Production build**

Run: `cd frontend-react && npm run build`
Expected: clean build, `dist/` produced.

- [ ] **Step 4: Manual E2E against the real backend**

Start the backend (`cd backend && uvicorn app.main:app --reload`) and `npm run dev`. Verify:
upload a PDF -> appears in Knowledge Base; ask a question -> streaming tokens + sources; delete-all with confirm modal; `/analytics` shows metrics; theme toggle persists; refresh keeps chat history.

- [ ] **Step 5: Final commit (if any fixes were needed)**

```bash
git add -A
git commit -m "test(frontend): fix issues found in final verification sweep"
```

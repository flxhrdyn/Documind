import { useState } from 'react';
import { NavLink, Outlet } from 'react-router-dom';
import { MessageSquare, BarChart3, Menu, X } from 'lucide-react';
import Sidebar from './Sidebar';
import ThemeToggle from './ThemeToggle';

const tabClass = ({ isActive }: { isActive: boolean }) =>
  `px-4 py-2 text-xs font-semibold rounded-lg transition-all duration-200 flex items-center gap-2 ${
    isActive
      ? 'bg-surface text-ink shadow-sm'
      : 'text-ink-muted hover:text-ink'
  }`;

function Brand() {
  return (
    <div className="p-5 border-b border-line flex items-center gap-3">
      <div className="w-9 h-9 rounded-xl bg-accent flex items-center justify-center text-accent-fg font-display font-bold text-lg shrink-0">
        I
      </div>
      <div className="leading-none">
        <h1 className="font-display font-bold text-ink tracking-tight text-base">InvenioAI</h1>
        <span className="text-[11px] text-ink-muted block mt-0.5">Document Assistant</span>
      </div>
    </div>
  );
}

export default function Layout() {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <div className="h-screen flex bg-bg text-ink font-sans overflow-hidden">
      <aside className="hidden md:flex w-80 shrink-0 bg-surface/60 border-r border-line flex-col">
        <Brand />
        <div className="flex-1 overflow-y-auto p-5 space-y-7">
          <Sidebar />
        </div>
      </aside>

      {sidebarOpen && (
        <div className="md:hidden fixed inset-0 z-40 flex">
          <button
            aria-label="Close menu"
            onClick={() => setSidebarOpen(false)}
            className="absolute inset-0 bg-ink/40 motion-safe:animate-[fade-in_0.2s_ease-out]"
          />
          <aside className="relative w-80 max-w-[85vw] h-full bg-surface border-r border-line flex flex-col motion-safe:animate-[slide-in-left_0.25s_ease-out]">
            <div className="flex items-center justify-between border-b border-line">
              <Brand />
              <button
                aria-label="Close menu"
                onClick={() => setSidebarOpen(false)}
                className="p-2 mr-4 rounded-lg text-ink-muted hover:text-ink hover:bg-surface-2"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            <div className="flex-1 overflow-y-auto p-5 space-y-7">
              <Sidebar />
            </div>
          </aside>
        </div>
      )}

      <main className="flex-1 flex flex-col min-w-0">
        <nav className="flex items-center justify-between px-4 md:px-6 py-4 bg-surface/20 border-b border-line z-10">
          <div className="flex items-center gap-2">
            <button
              aria-label="Open menu"
              onClick={() => setSidebarOpen(true)}
              className="md:hidden p-2 rounded-lg border border-line bg-surface hover:bg-surface-2 text-ink-muted hover:text-ink transition-all"
            >
              <Menu className="w-4 h-4" />
            </button>
            <div className="flex bg-surface-2/60 p-1 rounded-xl">
              <NavLink to="/chat" className={tabClass}>
                <MessageSquare className="w-3.5 h-3.5 text-accent" />
                <span className="hidden sm:inline">Chat</span>
              </NavLink>
              <NavLink to="/analytics" className={tabClass}>
                <BarChart3 className="w-3.5 h-3.5 text-accent" />
                <span className="hidden sm:inline">Analytics</span>
              </NavLink>
            </div>
          </div>

          <ThemeToggle />
        </nav>

        <div className="flex-1 min-h-0 overflow-hidden">
          <Outlet />
        </div>
      </main>
    </div>
  );
}

import { NavLink, Outlet } from 'react-router-dom';
import { MessageSquare, BarChart3 } from 'lucide-react';
import Sidebar from './Sidebar';
import ThemeToggle from './ThemeToggle';

const tabClass = ({ isActive }: { isActive: boolean }) =>
  `px-4 py-2 text-xs font-semibold rounded-lg transition-all duration-200 flex items-center gap-2 ${
    isActive
      ? 'bg-surface text-ink shadow-sm'
      : 'text-ink-muted hover:text-ink'
  }`;

export default function Layout() {
  return (
    <div className="min-h-screen flex bg-bg text-ink font-sans">
      <aside className="w-80 shrink-0 bg-surface/60 border-r border-line flex flex-col">
        <div className="p-5 border-b border-line flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-accent flex items-center justify-center text-accent-fg font-display font-bold text-lg shrink-0">
            I
          </div>
          <div className="leading-none">
            <h1 className="font-display font-bold text-ink tracking-tight text-base">InvenioAI</h1>
            <span className="text-[11px] text-ink-muted block mt-0.5">Document Assistant</span>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-5 space-y-7">
          <Sidebar />
        </div>
      </aside>

      <main className="flex-1 flex flex-col min-w-0">
        <nav className="flex items-center justify-between px-6 py-4 bg-surface/20 border-b border-line z-10">
          <div className="flex bg-surface-2/60 p-1 rounded-xl">
            <NavLink to="/chat" className={tabClass}>
              <MessageSquare className="w-3.5 h-3.5 text-accent" />
              <span>Chat</span>
            </NavLink>
            <NavLink to="/analytics" className={tabClass}>
              <BarChart3 className="w-3.5 h-3.5 text-accent" />
              <span>Analytics</span>
            </NavLink>
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

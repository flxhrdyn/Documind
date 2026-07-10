import { NavLink, Outlet } from 'react-router-dom';
import { MessageSquare, BarChart3 } from 'lucide-react';
import Sidebar from './Sidebar';
import ThemeToggle from './ThemeToggle';

const navClass = ({ isActive }: { isActive: boolean }) =>
  `flex items-center gap-2.5 px-3.5 py-2 rounded-lg text-sm font-medium transition-colors ${
    isActive
      ? 'bg-accent text-accent-fg'
      : 'text-ink-muted hover:bg-surface hover:text-ink'
  }`;

export default function Layout() {
  return (
    <div className="min-h-screen flex bg-bg text-ink">
      <aside className="w-72 shrink-0 border-r border-line bg-surface-2 flex flex-col p-4 gap-6">
        <div className="flex items-center justify-between px-1">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-accent flex items-center justify-center text-accent-fg font-bold text-base shrink-0">
              I
            </div>
            <div className="leading-tight">
              <h1 className="font-bold text-sm tracking-tight">InvenioAI</h1>
              <p className="text-[11px] text-ink-muted">Document Intelligence</p>
            </div>
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
      <main className="flex-1 min-w-0 flex flex-col bg-bg">
        <Outlet />
      </main>
    </div>
  );
}

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

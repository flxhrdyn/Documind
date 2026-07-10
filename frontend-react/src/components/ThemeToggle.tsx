import { useEffect, useState } from 'react';
import { Moon, Sun } from 'lucide-react';

function getInitialDark(): boolean {
  const saved = localStorage.getItem('invenio_theme');
  if (saved === 'dark') return true;
  if (saved === 'light') return false;
  return window.matchMedia('(prefers-color-scheme: dark)').matches;
}

export default function ThemeToggle() {
  const [dark, setDark] = useState(getInitialDark);

  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark);
    localStorage.setItem('invenio_theme', dark ? 'dark' : 'light');
  }, [dark]);

  return (
    <button
      onClick={() => setDark((v) => !v)}
      aria-label={dark ? 'Switch to light theme' : 'Switch to dark theme'}
      className="p-2 rounded-lg text-ink-muted hover:bg-surface hover:text-ink transition-colors"
    >
      {dark ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
    </button>
  );
}

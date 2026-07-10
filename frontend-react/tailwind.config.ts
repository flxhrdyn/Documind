import type { Config } from 'tailwindcss';

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        bg: 'var(--color-bg)',
        surface: 'var(--color-surface)',
        'surface-2': 'var(--color-surface-2)',
        ink: { DEFAULT: 'var(--color-ink)', muted: 'var(--color-ink-muted)' },
        line: 'var(--color-line)',
        accent: {
          DEFAULT: 'var(--color-accent)',
          soft: 'var(--color-accent-soft)',
          ink: 'var(--color-accent-ink)',
          fg: 'var(--color-accent-fg)',
        },
      },
      fontFamily: {
        sans: ['"Manrope"', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'monospace'],
      },
    },
  },
} satisfies Config;

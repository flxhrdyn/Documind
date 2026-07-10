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

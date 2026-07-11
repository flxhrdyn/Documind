/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import { Sun, Moon } from 'lucide-react';

interface ThemeToggleProps {
  darkMode: boolean;
  onToggle: () => void;
}

export default function ThemeToggle({ darkMode, onToggle }: ThemeToggleProps) {
  return (
    <button
      id="theme-toggle"
      onClick={onToggle}
      aria-label={darkMode ? "Switch to light mode" : "Switch to dark mode"}
      className="relative p-2 rounded-xl border border-gray-200/80 bg-white hover:bg-gray-50 text-gray-700 dark:border-gray-800 dark:bg-gray-900 dark:hover:bg-gray-800/80 dark:text-gray-300 transition-all duration-200 shadow-sm focus-visible:ring-2 focus-visible:ring-teal-500/80 focus-visible:ring-offset-2"
    >
      <div className="relative w-5 h-5 flex items-center justify-center overflow-hidden">
        {/* Sun Icon */}
        <span
          className={`absolute transform transition-all duration-300 ease-out ${
            darkMode 
              ? 'translate-y-8 rotate-45 opacity-0' 
              : 'translate-y-0 rotate-0 opacity-100'
          }`}
        >
          <Sun className="w-5 h-5 text-teal-600" />
        </span>

        {/* Moon Icon */}
        <span
          className={`absolute transform transition-all duration-300 ease-out ${
            darkMode 
              ? 'translate-y-0 rotate-0 opacity-100' 
              : '-translate-y-8 -rotate-45 opacity-0'
          }`}
        >
          <Moon className="w-5 h-5 text-teal-400" />
        </span>
      </div>
    </button>
  );
}

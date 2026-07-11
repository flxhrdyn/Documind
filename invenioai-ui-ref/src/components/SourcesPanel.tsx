/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import { BookOpen, FileText, Compass, ExternalLink } from 'lucide-react';
import { Citation } from '../types';

interface SourcesPanelProps {
  citations: Citation[];
  activeCitationId: string | null;
  onCitationClick: (id: string) => void;
}

export default function SourcesPanel({
  citations,
  activeCitationId,
  onCitationClick
}: SourcesPanelProps) {
  
  const getScoreColor = (score: number) => {
    if (score >= 0.90) return 'text-emerald-600 bg-emerald-50 dark:text-emerald-400 dark:bg-emerald-950/30 border-emerald-100 dark:border-emerald-900/50';
    if (score >= 0.75) return 'text-teal-600 bg-teal-50 dark:text-teal-400 dark:bg-teal-950/30 border-teal-100 dark:border-teal-900/50';
    return 'text-gray-600 bg-gray-50 dark:text-gray-400 dark:bg-gray-900/50 border-gray-100 dark:border-gray-800/50';
  };

  return (
    <div className="h-full flex flex-col bg-white dark:bg-gray-950/20 border-l border-gray-100 dark:border-gray-900/80">
      {/* Panel Header */}
      <div className="p-4 border-b border-gray-100 dark:border-gray-900 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <BookOpen className="w-4 h-4 text-teal-600 dark:text-teal-400" />
          <h3 className="text-sm font-semibold text-gray-800 dark:text-gray-200 font-display">
            References ({citations.length})
          </h3>
        </div>
      </div>

      {/* Sources Body */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3.5">
        {citations.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-center py-20 px-4">
            <Compass className="w-8 h-8 text-gray-300 dark:text-gray-700 mb-2.5 stroke-[1.5]" />
            <p className="text-xs font-medium text-gray-600 dark:text-gray-400">
              No active sources
            </p>
            <p className="text-[10px] text-gray-400 dark:text-gray-500 mt-1 max-w-[200px] leading-relaxed">
              Ask a question or click a citation badge <span className="bg-gray-100 dark:bg-gray-900 px-1 py-0.5 rounded text-teal-600 font-semibold">[1]</span> to inspect source passages.
            </p>
          </div>
        ) : (
          <div className="space-y-4">
            <p className="text-[11px] text-gray-500 dark:text-gray-400 leading-relaxed font-sans">
              Passages supporting the current response:
            </p>
            
            {citations.map((cite) => {
              const isActive = activeCitationId === cite.id;
              
              return (
                <div
                  key={cite.id}
                  id={`source-card-${cite.id}`}
                  onClick={() => onCitationClick(cite.id)}
                  className={`group relative p-4 rounded-xl border text-left cursor-pointer transition-all duration-300 ${
                    isActive
                      ? 'border-teal-400 bg-teal-50/10 dark:border-teal-500/40 dark:bg-teal-950/10 shadow-md ring-1 ring-teal-400/30'
                      : 'border-gray-100 bg-white hover:border-gray-200 hover:shadow-sm dark:border-gray-900 dark:bg-gray-950/40 dark:hover:border-gray-800'
                  }`}
                >
                  {/* Active Indicator bar */}
                  {isActive && (
                    <div className="absolute top-0 bottom-0 left-0 w-1 bg-teal-600 rounded-l-xl" />
                  )}

                  <div className="flex items-start justify-between mb-2.5">
                    {/* Badge Citation ID */}
                    <div className="flex items-center space-x-2 min-w-0">
                      <span className={`flex items-center justify-center w-5 h-5 rounded-md text-[11px] font-mono font-bold shrink-0 ${
                        isActive 
                          ? 'bg-teal-600 text-white shadow-sm' 
                          : 'bg-gray-100 text-gray-600 dark:bg-gray-900 dark:text-gray-400 group-hover:bg-teal-100/60 dark:group-hover:bg-teal-950/30 group-hover:text-teal-600 dark:group-hover:text-teal-400 transition-colors duration-150'
                      }`}>
                        {cite.id}
                      </span>
                      <div className="min-w-0">
                        <div className="flex items-center space-x-1.5">
                          <FileText className="w-3 h-3 text-gray-400 shrink-0" />
                          <span className="text-xs font-semibold text-gray-800 dark:text-gray-200 truncate pr-1" title={cite.documentName}>
                            {cite.documentName}
                          </span>
                        </div>
                        <span className="text-[10px] font-mono text-gray-400 dark:text-gray-500">
                          Page {cite.pageNumber}
                        </span>
                      </div>
                    </div>

                    {/* Match Score */}
                    <div className={`px-2 py-0.5 rounded-full text-[9px] font-semibold border ${getScoreColor(cite.relevanceScore)}`}>
                      Match {(cite.relevanceScore * 100).toFixed(0)}%
                    </div>
                  </div>

                  {/* Snippet Quote block */}
                  <div className="relative text-[11.5px] leading-relaxed text-gray-600 dark:text-gray-300 font-sans break-words bg-gray-50/50 dark:bg-gray-900/20 p-2.5 rounded-lg border border-gray-50 dark:border-gray-900/50">
                    {isActive ? (
                      <span className="citation-highlight p-0.5 rounded">
                        {cite.snippet}
                      </span>
                    ) : (
                      cite.snippet
                    )}
                  </div>

                  {/* Bottom details */}
                  <div className="flex items-center justify-end mt-2 text-[10px] text-teal-600 dark:text-teal-400 font-medium opacity-0 group-hover:opacity-100 transition-opacity duration-150">
                    <span className="flex items-center space-x-1">
                      <span>Show original file</span>
                      <ExternalLink className="w-2.5 h-2.5" />
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

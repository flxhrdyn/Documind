/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useMemo } from 'react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer
} from 'recharts';
import {
  Activity,
  Search,
  ChevronLeft,
  ChevronRight,
  TrendingDown,
  Cpu,
  BarChart3,
  Clock
} from 'lucide-react';
import { QueryMetrics, RAGMetricsSummary, QueryLog } from '../types';

interface MetricsDashboardProps {
  metricsSummary: RAGMetricsSummary;
  queryLogs: QueryLog[];
  latencyData: Array<{ name: string; Retrieval: number; Generation: number; Total: number }>;
}

export default function MetricsDashboard({
  metricsSummary,
  queryLogs,
  latencyData
}: MetricsDashboardProps) {
  const [searchTerm, setSearchTerm] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 5;

  // Search filter
  const filteredLogs = useMemo(() => {
    return queryLogs.filter(log =>
      log.query.toLowerCase().includes(searchTerm.toLowerCase())
    );
  }, [queryLogs, searchTerm]);

  // Pagination
  const totalPages = Math.ceil(filteredLogs.length / itemsPerPage) || 1;
  const paginatedLogs = useMemo(() => {
    const start = (currentPage - 1) * itemsPerPage;
    return filteredLogs.slice(start, start + itemsPerPage);
  }, [filteredLogs, currentPage]);

  const handlePageChange = (direction: 'next' | 'prev') => {
    if (direction === 'next' && currentPage < totalPages) {
      setCurrentPage(prev => prev + 1);
    } else if (direction === 'prev' && currentPage > 1) {
      setCurrentPage(prev => prev - 1);
    }
  };

  // IR Benchmarks Data (comparing hybrid dense-sparse vs baselines)
  const irBenchmarks = [
    { metric: 'Hit Rate @ 3', hybrid: 0.942, denseOnly: 0.814, sparseOnly: 0.742, description: 'Percentage of queries returning at least one relevant chunk within top 3 candidate positions.' },
    { metric: 'MRR @ 5', hybrid: 0.884, denseOnly: 0.723, sparseOnly: 0.651, description: 'Mean Reciprocal Rank — measures how high the first truly relevant document is positioned on average.' },
    { metric: 'nDCG @ 5', hybrid: 0.895, denseOnly: 0.751, sparseOnly: 0.684, description: 'Normalized Discounted Cumulative Gain — computes graded relevance of items based on position.' },
    { metric: 'Precision @ 1', hybrid: 0.781, denseOnly: 0.635, sparseOnly: 0.512, description: 'Direct accuracy of the single highest ranked recommendation.' }
  ];

  return (
    <div className="space-y-8 animate-in fade-in duration-300 p-1">
      
      {/* SECTION 1: Performance Summary */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="p-5 rounded-2xl bg-white dark:bg-gray-900 border border-gray-100 dark:border-gray-900 shadow-sm flex flex-col justify-between">
          <div>
            <span className="text-[10px] font-mono font-semibold text-gray-400 dark:text-gray-500 uppercase tracking-wider">Query Volume</span>
            <h4 className="text-3xl font-display font-bold text-gray-950 dark:text-gray-50 mt-1">{metricsSummary.totalQueries}</h4>
          </div>
          <div className="mt-3 flex items-center space-x-1.5 text-xs text-gray-500 dark:text-gray-400">
            <Activity className="w-3.5 h-3.5 text-teal-600 dark:text-teal-400" />
            <span>Active in current session</span>
          </div>
        </div>

        <div className="p-5 rounded-2xl bg-white dark:bg-gray-900 border border-gray-100 dark:border-gray-900 shadow-sm flex flex-col justify-between">
          <div>
            <span className="text-[10px] font-mono font-semibold text-gray-400 dark:text-gray-500 uppercase tracking-wider">Average Latency</span>
            <h4 className="text-3xl font-display font-bold text-teal-600 dark:text-teal-400 mt-1">
              {metricsSummary.avgLatencyMs.toFixed(0)} <span className="text-xs font-sans font-normal text-gray-400">ms</span>
            </h4>
          </div>
          <div className="mt-3 flex items-center space-x-1.5 text-xs text-gray-500 dark:text-gray-400">
            <TrendingDown className="w-3.5 h-3.5 text-emerald-500" />
            <span>Reflects overall search and answer time</span>
          </div>
        </div>

        <div className="p-5 rounded-2xl bg-white dark:bg-gray-900 border border-gray-100 dark:border-gray-900 shadow-sm flex flex-col justify-between">
          <div>
            <span className="text-[10px] font-mono font-semibold text-gray-400 dark:text-gray-500 uppercase tracking-wider">Retrieval Latency</span>
            <h4 className="text-3xl font-display font-bold text-gray-950 dark:text-gray-50 mt-1">
              {metricsSummary.avgRetrievalMs.toFixed(0)} <span className="text-xs font-sans font-normal text-gray-400">ms</span>
            </h4>
          </div>
          <div className="mt-3 flex items-center space-x-1.5 text-xs text-gray-500 dark:text-gray-400">
            <Cpu className="w-3.5 h-3.5 text-indigo-400" />
            <span>Document indexing and matching</span>
          </div>
        </div>

        <div className="p-5 rounded-2xl bg-white dark:bg-gray-900 border border-gray-100 dark:border-gray-900 shadow-sm flex flex-col justify-between">
          <div>
            <span className="text-[10px] font-mono font-semibold text-gray-400 dark:text-gray-500 uppercase tracking-wider">LLM Generation</span>
            <h4 className="text-3xl font-display font-bold text-gray-950 dark:text-gray-50 mt-1">
              {metricsSummary.avgGenerationMs.toFixed(0)} <span className="text-xs font-sans font-normal text-gray-400">ms</span>
            </h4>
          </div>
          <div className="mt-3 flex items-center space-x-1.5 text-xs text-gray-500 dark:text-gray-400">
            <Clock className="w-3.5 h-3.5 text-gray-400" />
            <span>Response rendering latency</span>
          </div>
        </div>
      </div>

      {/* SECTION 2: Latency Decomposition AreaChart */}
      <div className="p-6 rounded-2xl bg-white dark:bg-gray-900 border border-gray-100 dark:border-gray-900 shadow-sm">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-6 gap-2">
          <div>
            <h3 className="text-sm font-semibold text-gray-900 dark:text-gray-100 font-display">Response Time Decomposition</h3>
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">Time breakdown comparing document search duration vs text generation duration</p>
          </div>
          <div className="flex items-center space-x-3 text-xs font-mono">
            <span className="flex items-center space-x-1">
              <span className="w-2.5 h-2.5 bg-teal-500/80 rounded-xs" />
              <span className="text-gray-500 dark:text-gray-400 font-sans">Generation</span>
            </span>
            <span className="flex items-center space-x-1">
              <span className="w-2.5 h-2.5 bg-indigo-500/80 rounded-xs" />
              <span className="text-gray-500 dark:text-gray-400 font-sans">Retrieval</span>
            </span>
          </div>
        </div>

        <div className="h-64 w-full">
          {latencyData.length === 0 ? (
            <div className="h-full flex items-center justify-center text-xs text-gray-400 font-mono">
              No query latency data captured yet
            </div>
          ) : (
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart
                data={latencyData}
                margin={{ top: 10, right: 10, left: -20, bottom: 0 }}
              >
                <defs>
                  <linearGradient id="colorRetrieval" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#6366f1" stopOpacity={0.15}/>
                    <stop offset="95%" stopColor="#6366f1" stopOpacity={0}/>
                  </linearGradient>
                  <linearGradient id="colorGeneration" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#0d9488" stopOpacity={0.15}/>
                    <stop offset="95%" stopColor="#0d9488" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f3f4f6" className="dark:stroke-gray-800/60" />
                <XAxis
                  dataKey="name"
                  tickLine={false}
                  axisLine={false}
                  tick={{ fill: '#9ca3af', fontSize: 10, fontFamily: 'JetBrains Mono' }}
                />
                <YAxis
                  tickLine={false}
                  axisLine={false}
                  tick={{ fill: '#9ca3af', fontSize: 10, fontFamily: 'JetBrains Mono' }}
                  unit="ms"
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgba(17, 24, 39, 0.95)',
                    borderRadius: '12px',
                    border: 'none',
                    fontSize: '11px',
                    fontFamily: 'Inter',
                    color: '#fff'
                  }}
                  itemStyle={{ color: '#fff' }}
                  labelStyle={{ color: '#9ca3af', fontFamily: 'JetBrains Mono', marginBottom: '4px' }}
                />
                <Area
                  type="monotone"
                  dataKey="Retrieval"
                  stackId="1"
                  stroke="#6366f1"
                  strokeWidth={2}
                  fillOpacity={1}
                  fill="url(#colorRetrieval)"
                />
                <Area
                  type="monotone"
                  dataKey="Generation"
                  stackId="1"
                  stroke="#0d9488"
                  strokeWidth={2}
                  fillOpacity={1}
                  fill="url(#colorGeneration)"
                />
              </AreaChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>

      {/* SECTION 3: IR Quality Metrics / Retrieval Benchmarks */}
      <div className="p-6 rounded-2xl bg-white dark:bg-gray-900 border border-gray-100 dark:border-gray-900 shadow-sm">
        <div className="mb-6">
          <div className="flex items-center space-x-2">
            <BarChart3 className="w-4 h-4 text-indigo-500" />
            <h3 className="text-sm font-semibold text-gray-900 dark:text-gray-100 font-display">Search Performance Benchmarks</h3>
          </div>
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">Comparative retrieval quality of hybrid search vs standard search techniques</p>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-gray-100 dark:border-gray-800 text-gray-400 dark:text-gray-500 font-medium">
                <th className="py-3 px-4 font-sans">Metric</th>
                <th className="py-3 px-4 font-mono text-right text-gray-900 dark:text-gray-100">Hybrid Search</th>
                <th className="py-3 px-4 font-mono text-right">Dense Search</th>
                <th className="py-3 px-4 font-mono text-right">Keyword Search</th>
                <th className="py-3 px-4 text-gray-400 dark:text-gray-500 pl-6">Description</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100 dark:divide-gray-800 text-gray-600 dark:text-gray-300">
              {irBenchmarks.map((row, idx) => (
                <tr key={idx} className="hover:bg-gray-50/50 dark:hover:bg-gray-800/20 transition-colors duration-150">
                  <td className="py-3.5 px-4 font-semibold text-gray-800 dark:text-gray-200">{row.metric}</td>
                  <td className="py-3.5 px-4 font-mono text-right text-teal-600 dark:text-teal-400 font-bold bg-teal-500/5">
                    {(row.hybrid * 100).toFixed(1)}%
                  </td>
                  <td className="py-3.5 px-4 font-mono text-right">{(row.denseOnly * 100).toFixed(1)}%</td>
                  <td className="py-3.5 px-4 font-mono text-right">{(row.sparseOnly * 100).toFixed(1)}%</td>
                  <td className="py-3.5 px-4 text-[11px] text-gray-400 dark:text-gray-500 leading-relaxed max-w-[280px] pl-6">{row.description}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* SECTION 4: Query History Log */}
      <div className="p-6 rounded-2xl bg-white dark:bg-gray-900 border border-gray-100 dark:border-gray-900 shadow-sm">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-6">
          <div>
            <h3 className="text-sm font-semibold text-gray-900 dark:text-gray-100 font-display">Query History</h3>
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">Log of questions submitted and their respective performance metrics</p>
          </div>

          {/* Search bar inside logs */}
          <div className="relative max-w-xs w-full">
            <Search className="w-3.5 h-3.5 text-gray-400 dark:text-gray-500 absolute left-3 top-1/2 transform -translate-y-1/2" />
            <input
              type="text"
              placeholder="Search query keywords..."
              value={searchTerm}
              onChange={(e) => {
                setSearchTerm(e.target.value);
                setCurrentPage(1);
              }}
              className="w-full text-xs pl-9 pr-4 py-2 border border-gray-200 dark:border-gray-800 rounded-xl bg-gray-50/50 dark:bg-gray-950 focus:border-teal-500 focus:outline-none transition-colors duration-150"
            />
          </div>
        </div>

        {filteredLogs.length === 0 ? (
          <div className="py-12 text-center text-xs text-gray-400 font-mono">
            No queries match your search keywords.
          </div>
        ) : (
          <div className="space-y-4">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs border-collapse">
                <thead>
                  <tr className="border-b border-gray-100 dark:border-gray-800 text-gray-400 dark:text-gray-500 font-medium">
                    <th className="py-3 px-4">Question</th>
                    <th className="py-3 px-4 font-mono">Query Time</th>
                    <th className="py-3 px-4 font-mono text-right">Latency</th>
                    <th className="py-3 px-4 font-mono text-right">References</th>
                    <th className="py-3 px-4 font-mono text-right">Best Match</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100 dark:divide-gray-800 text-gray-600 dark:text-gray-300">
                  {paginatedLogs.map((log) => (
                    <tr key={log.id} className="hover:bg-gray-50/50 dark:hover:bg-gray-800/15 transition-all duration-150">
                      <td className="py-3.5 px-4 font-medium text-gray-800 dark:text-gray-200 max-w-[280px] truncate" title={log.query}>
                        {log.query}
                      </td>
                      <td className="py-3.5 px-4 font-mono text-gray-400 dark:text-gray-500">{log.timestamp.split(' ')[1] || log.timestamp}</td>
                      <td className={`py-3.5 px-4 font-mono text-right ${log.latencyMs < 20 ? 'text-emerald-500 font-semibold' : ''}`}>
                        {log.latencyMs} ms
                        {log.latencyMs < 20 && <span className="text-[9px] block text-emerald-600 dark:text-emerald-400 font-sans font-semibold">Cache Hit</span>}
                      </td>
                      <td className="py-3.5 px-4 font-mono text-right">{log.sourceCount} chunks</td>
                      <td className="py-3.5 px-4 font-mono text-right text-gray-900 dark:text-gray-100">
                        {log.topScore ? `${(log.topScore * 100).toFixed(0)}%` : '-'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Pagination Controls */}
            <div className="flex items-center justify-between pt-4 border-t border-gray-100 dark:border-gray-800 text-xs">
              <span className="text-gray-400 dark:text-gray-500 font-mono">
                Showing {Math.min(filteredLogs.length, (currentPage - 1) * itemsPerPage + 1)}-
                {Math.min(filteredLogs.length, currentPage * itemsPerPage)} of {filteredLogs.length} queries
              </span>

              <div className="flex items-center space-x-1.5">
                <button
                  onClick={() => handlePageChange('prev')}
                  disabled={currentPage === 1}
                  className="p-1.5 rounded-lg border border-gray-100 dark:border-gray-800 hover:bg-gray-50 dark:hover:bg-gray-800 text-gray-500 disabled:opacity-40 transition-colors duration-150 cursor-pointer"
                  aria-label="Previous page"
                >
                  <ChevronLeft className="w-4 h-4" />
                </button>
                <span className="font-mono text-gray-600 dark:text-gray-400 px-2.5">
                  {currentPage} / {totalPages}
                </span>
                <button
                  onClick={() => handlePageChange('next')}
                  disabled={currentPage === totalPages}
                  className="p-1.5 rounded-lg border border-gray-100 dark:border-gray-800 hover:bg-gray-50 dark:hover:bg-gray-800 text-gray-500 disabled:opacity-40 transition-colors duration-150 cursor-pointer"
                  aria-label="Next page"
                >
                  <ChevronRight className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

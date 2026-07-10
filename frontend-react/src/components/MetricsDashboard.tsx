import type { ReactNode } from 'react';
import {
  Area, AreaChart, ResponsiveContainer, Tooltip, XAxis, YAxis, CartesianGrid,
} from 'recharts';
import { Activity, Cpu, Clock, TrendingDown, BarChart3 } from 'lucide-react';
import type { MetricsResponse } from '../types';

function pct(v: number | undefined): string {
  return v == null ? '-' : `${(v * 100).toFixed(1)}%`;
}

function ms(v: number | undefined): string {
  return v == null ? '-' : `${(v * 1000).toFixed(0)}`;
}

function StatCard({
  label, value, unit, icon, hint,
}: { label: string; value: string; unit?: string; icon: ReactNode; hint: string }) {
  return (
    <div className="p-5 rounded-2xl bg-surface border border-line flex flex-col justify-between">
      <div>
        <span className="text-[10px] font-mono font-semibold text-ink-muted uppercase tracking-wider">{label}</span>
        <h4 className="text-3xl font-display font-bold text-ink mt-1">
          {value} {unit && <span className="text-xs font-sans font-normal text-ink-muted">{unit}</span>}
        </h4>
      </div>
      <div className="mt-3 flex items-center gap-1.5 text-xs text-ink-muted">
        {icon}
        <span>{hint}</span>
      </div>
    </div>
  );
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl border border-line bg-surface p-4">
      <p className="text-[10px] text-ink-muted uppercase tracking-wide font-mono">{label}</p>
      <p className="font-mono text-xl font-semibold mt-1.5 tabular-nums text-accent-ink">{value}</p>
    </div>
  );
}

export default function MetricsDashboard({ metrics }: { metrics: MetricsResponse }) {
  const ir = metrics.ir_quality;
  const chartData = metrics.query_history.map((q, i) => ({
    name: `Query ${i + 1}`,
    Total: q.response_time != null ? q.response_time * 1000 : 0,
    Retrieval: q.retrieval_time != null ? q.retrieval_time * 1000 : 0,
    Generation: q.generation_time != null ? q.generation_time * 1000 : 0,
  }));

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          label="Query Volume"
          value={String(metrics.total_queries)}
          icon={<Activity className="w-3.5 h-3.5 text-accent-ink" />}
          hint="Questions asked this session"
        />
        <StatCard
          label="Average Latency"
          value={ms(metrics.avg_response_time)}
          unit="ms"
          icon={<TrendingDown className="w-3.5 h-3.5 text-emerald-500" />}
          hint="Reflects overall search and answer time"
        />
        <StatCard
          label="Retrieval Latency"
          value={ms(metrics.avg_retrieval_time)}
          unit="ms"
          icon={<Cpu className="w-3.5 h-3.5 text-indigo-400" />}
          hint="Document search and matching"
        />
        <StatCard
          label="Indexed Docs"
          value={String(metrics.total_documents_indexed)}
          icon={<Clock className="w-3.5 h-3.5 text-ink-muted" />}
          hint="Currently in the knowledge base"
        />
      </div>

      <div className="p-6 rounded-2xl bg-surface border border-line">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-6 gap-2">
          <div>
            <h3 className="text-sm font-semibold font-display text-ink">Response Time Decomposition</h3>
            <p className="text-xs text-ink-muted mt-0.5">Time breakdown comparing document search duration vs answer generation duration</p>
          </div>
          <div className="flex items-center gap-3 text-xs font-mono">
            <span className="flex items-center gap-1">
              <span className="w-2.5 h-2.5 bg-accent/80 rounded-sm" />
              <span className="text-ink-muted font-sans">Generation</span>
            </span>
            <span className="flex items-center gap-1">
              <span className="w-2.5 h-2.5 bg-indigo-400/80 rounded-sm" />
              <span className="text-ink-muted font-sans">Retrieval</span>
            </span>
          </div>
        </div>

        <div className="h-64 w-full">
          {chartData.length === 0 ? (
            <div className="h-full flex items-center justify-center text-xs text-ink-muted font-mono">
              No query latency data captured yet
            </div>
          ) : (
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <defs>
                  <linearGradient id="colorRetrieval" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#818cf8" stopOpacity={0.25} />
                    <stop offset="95%" stopColor="#818cf8" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="colorGeneration" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="var(--color-accent)" stopOpacity={0.25} />
                    <stop offset="95%" stopColor="var(--color-accent)" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--color-line)" />
                <XAxis dataKey="name" tickLine={false} axisLine={false} tick={{ fill: 'var(--color-ink-muted)', fontSize: 10 }} />
                <YAxis tickLine={false} axisLine={false} tick={{ fill: 'var(--color-ink-muted)', fontSize: 10 }} unit="ms" />
                <Tooltip
                  contentStyle={{
                    background: 'var(--color-surface)',
                    border: '1px solid var(--color-line)',
                    borderRadius: 12,
                    fontSize: 11,
                  }}
                />
                <Area type="monotone" dataKey="Retrieval" stackId="1" stroke="#818cf8" strokeWidth={2} fill="url(#colorRetrieval)" />
                <Area type="monotone" dataKey="Generation" stackId="1" stroke="var(--color-accent)" strokeWidth={2} fill="url(#colorGeneration)" />
              </AreaChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>

      <div className="p-6 rounded-2xl bg-surface border border-line">
        <div className="flex items-center gap-2 mb-4">
          <BarChart3 className="w-4 h-4 text-indigo-400" />
          <h3 className="text-sm font-semibold font-display text-ink">Information Retrieval Quality</h3>
        </div>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
          <MetricCard label="Hit Rate" value={pct(ir?.hit_rate)} />
          <MetricCard label="nDCG" value={pct(ir?.ndcg)} />
          <MetricCard label="MRR" value={pct(ir?.mrr)} />
          <MetricCard label="Precision" value={pct(ir?.precision)} />
        </div>
      </div>

      <div className="p-6 rounded-2xl bg-surface border border-line">
        <h3 className="text-sm font-semibold font-display text-ink mb-0.5">Query History</h3>
        <p className="text-xs text-ink-muted mb-4">Log of questions submitted and their respective performance metrics</p>

        {metrics.query_history.length === 0 ? (
          <div className="py-10 text-center text-xs text-ink-muted font-mono">No queries recorded yet.</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="border-b border-line text-ink-muted font-medium">
                  <th className="py-3 px-3">Question</th>
                  <th className="py-3 px-3 font-mono text-right">Latency</th>
                  <th className="py-3 px-3 font-mono text-right">Retrieval</th>
                  <th className="py-3 px-3 font-mono text-right">Generation</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-line text-ink-muted">
                {metrics.query_history.slice().reverse().map((q, i) => (
                  <tr key={i} className="hover:bg-surface-2/40 transition-colors">
                    <td className="py-3 px-3 font-medium text-ink max-w-[280px] truncate" title={q.question}>
                      {q.question}
                    </td>
                    <td className="py-3 px-3 font-mono text-right">{ms(q.response_time)} ms</td>
                    <td className="py-3 px-3 font-mono text-right">{ms(q.retrieval_time)} ms</td>
                    <td className="py-3 px-3 font-mono text-right">{ms(q.generation_time)} ms</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

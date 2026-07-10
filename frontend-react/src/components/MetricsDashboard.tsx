import {
  Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis, CartesianGrid,
} from 'recharts';
import type { MetricsResponse } from '../types';

function pct(v: number | undefined): string {
  return v == null ? '-' : `${(v * 100).toFixed(1)}%`;
}

function Card({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl border border-line bg-cream-card p-4">
      <p className="text-xs text-charcoal-muted">{label}</p>
      <p className="font-display text-2xl font-bold mt-1">{value}</p>
    </div>
  );
}

export default function MetricsDashboard({ metrics }: { metrics: MetricsResponse }) {
  const ir = metrics.ir_quality;
  const chartData = metrics.query_history.map((q, i) => ({
    name: `Q${i + 1}`,
    Total: q.response_time ?? 0,
    Retrieval: q.retrieval_time ?? 0,
    Generation: q.generation_time ?? 0,
  }));

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <Card label="HitRate@k" value={pct(ir?.hit_rate)} />
        <Card label="nDCG@k" value={pct(ir?.ndcg)} />
        <Card label="Avg Response" value={`${metrics.avg_response_time.toFixed(2)}s`} />
        <Card label="Indexed Docs" value={String(metrics.total_documents_indexed)} />
      </div>

      <div className="rounded-2xl border border-line bg-cream-card p-4">
        <h3 className="font-display text-sm font-semibold mb-3">Response Time Trend</h3>
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e4dccc" />
            <XAxis dataKey="name" fontSize={12} />
            <YAxis fontSize={12} unit="s" />
            <Tooltip />
            <Line type="monotone" dataKey="Total" stroke="#b4552d" strokeWidth={2} />
            <Line type="monotone" dataKey="Retrieval" stroke="#6b645b" strokeWidth={2} />
            <Line type="monotone" dataKey="Generation" stroke="#c9714b" strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <Card label="Precision@k" value={pct(ir?.precision)} />
        <Card label="Recall@k" value={pct(ir?.recall)} />
        <Card label="MRR" value={pct(ir?.mrr)} />
        <Card label="Avg Retrieval" value={`${metrics.avg_retrieval_time.toFixed(2)}s`} />
      </div>
    </div>
  );
}

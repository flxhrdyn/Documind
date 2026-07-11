import { useMetrics } from '../hooks/useMetrics';
import MetricsDashboard from '../components/MetricsDashboard';

export default function AnalyticsPage() {
  const { data, isLoading, isError } = useMetrics();

  return (
    <div className="h-full min-h-0 overflow-y-auto p-6 md:p-8 bg-bg">
      <div className="max-w-4xl mx-auto space-y-6">
        <div className="pb-4 border-b border-line">
          <h2 className="text-base font-semibold font-display text-ink">RAG Pipeline Analytics Dashboard</h2>
          <p className="text-xs text-ink-muted mt-0.5 max-w-md">
            Evaluate retrieval latency, semantic caching efficiency, and information retrieval precision.
          </p>
        </div>
        {isLoading && <p className="text-sm text-ink-muted">Loading metrics...</p>}
        {isError && <p className="text-sm text-accent-ink">Could not load metrics.</p>}
        {data && data.total_queries === 0 && (
          <p className="text-sm text-ink-muted">
            No queries recorded yet. Ask a question on the Chat page.
          </p>
        )}
        {data && data.total_queries > 0 && <MetricsDashboard metrics={data} />}
      </div>
    </div>
  );
}

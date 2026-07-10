import { useMetrics } from '../hooks/useMetrics';
import MetricsDashboard from '../components/MetricsDashboard';

export default function AnalyticsPage() {
  const { data, isLoading, isError } = useMetrics();

  return (
    <div className="h-screen overflow-y-auto p-8">
      <div className="max-w-4xl mx-auto">
        <h2 className="font-display text-2xl font-bold mb-1">Analytics</h2>
        <p className="text-sm text-charcoal-muted mb-6">
          RAG retrieval quality and latency.
        </p>
        {isLoading && <p className="text-sm text-charcoal-muted">Loading metrics...</p>}
        {isError && <p className="text-sm text-accent">Could not load metrics.</p>}
        {data && data.total_queries === 0 && (
          <p className="text-sm text-charcoal-muted">
            No queries recorded yet. Ask a question on the Chat page.
          </p>
        )}
        {data && data.total_queries > 0 && <MetricsDashboard metrics={data} />}
      </div>
    </div>
  );
}

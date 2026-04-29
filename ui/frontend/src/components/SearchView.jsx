import { useState } from 'react';
import { api } from '../api';

export default function SearchView() {
  const [query, setQuery] = useState('');
  const [topK, setTopK] = useState(10);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleSearch = async () => {
    if (!query.trim()) return;
    setLoading(true);
    const res = await api.search(query, parseInt(topK, 10));
    setResults(Array.isArray(res) ? res : []);
    setLoading(false);
  };

  return (
    <div className="space-y-4">
      <h2 className="text-2xl font-semibold text-white/90 mb-2">Search Memories</h2>
      <div className="flex gap-2">
        <input
          type="text"
          placeholder="Enter query..."
          className="glass-input flex-1 rounded-xl px-4 py-2.5 text-white/90 placeholder-white/30"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
        />
        <select
          className="glass-input rounded-xl px-3 py-2.5 text-white/90 w-24"
          value={topK}
          onChange={(e) => setTopK(e.target.value)}
        >
          <option value={5} className="bg-neutral-900">Top 5</option>
          <option value={10} className="bg-neutral-900">Top 10</option>
          <option value={25} className="bg-neutral-900">Top 25</option>
          <option value={50} className="bg-neutral-900">Top 50</option>
        </select>
        <button className="glass-btn glass-btn-primary rounded-xl px-5 py-2.5 text-white font-medium" onClick={handleSearch} disabled={loading}>
          {loading ? <span className="inline-block w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" /> : 'Search'}
        </button>
      </div>

      <div className="space-y-2">
        {results.length === 0 && !loading && (
          <div className="text-center text-white/30 py-8">No results yet</div>
        )}
        {results.map((r, i) => (
          <div key={i} className="glass rounded-xl">
            <div className="p-4">
              <div className="flex justify-between items-start">
                <div className="text-sm font-mono text-blue-300/80">ID: {r.id}</div>
                <div className="glass px-2 py-0.5 rounded-full text-xs text-white/70">Score: {(r.score * 100).toFixed(2)}%</div>
              </div>
              <p className="mt-3 text-white/80 whitespace-pre-wrap leading-relaxed">{r.text}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

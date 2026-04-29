import { api } from '../api';

export default function StatsView({ stats, onRefresh }) {
  const handleCheckpoint = async () => {
    await api.checkpoint();
    if (onRefresh) onRefresh();
  };

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-2xl font-semibold text-white/90">Index Statistics</h2>
        <div className="flex gap-2">
          <button className="glass-btn rounded-xl px-4 py-2 text-sm text-white/80" onClick={onRefresh}>
            Refresh
          </button>
          <button className="glass-btn glass-btn-primary rounded-xl px-4 py-2 text-sm text-white" onClick={handleCheckpoint}>
            Checkpoint
          </button>
        </div>
      </div>

      {!stats && (
        <div className="flex justify-center py-8">
          <span className="inline-block w-8 h-8 border-2 border-white/20 border-t-white/80 rounded-full animate-spin" />
        </div>
      )}

      {stats && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="glass rounded-xl p-5">
            <div className="text-sm text-white/40 mb-1">Memories</div>
            <div className="text-3xl font-semibold text-white/90">{stats.memoryCount ?? 0}</div>
          </div>
          <div className="glass rounded-xl p-5">
            <div className="text-sm text-white/40 mb-1">Fingerprints</div>
            <div className="text-3xl font-semibold text-white/90">{stats.fingerprintCount ?? 0}</div>
          </div>
          <div className="glass rounded-xl p-5">
            <div className="text-sm text-white/40 mb-1">Chunks</div>
            <div className="text-3xl font-semibold text-white/90">{stats.chunkCount ?? 0}</div>
          </div>
          <div className="glass rounded-xl p-5">
            <div className="text-sm text-white/40 mb-1">Lexical Terms</div>
            <div className="text-3xl font-semibold text-white/90">{stats.lexicalTermCount ?? 0}</div>
          </div>
        </div>
      )}
    </div>
  );
}

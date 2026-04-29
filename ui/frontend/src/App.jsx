import { useState, useEffect } from 'react';
import { api } from './api';
import SearchView from './components/SearchView';
import InsertView from './components/InsertView';
import StatsView from './components/StatsView';

function App() {
  const [activeTab, setActiveTab] = useState('search');
  const [stats, setStats] = useState(null);

  const refreshStats = async () => {
    const result = await api.stats();
    setStats(result);
  };

  useEffect(() => {
    refreshStats();
  }, []);

  return (
    <div className="min-h-screen p-4">
      <div className="max-w-4xl mx-auto">
        <div className="glass-strong navbar rounded-2xl mb-6 px-6">
          <div className="flex-1">
            <a className="text-xl font-semibold tracking-tight text-white/90">Adaline</a>
          </div>
          <div className="flex-none gap-2">
            <div className="glass px-3 py-1 rounded-full text-sm text-white/70">
              {stats ? `${stats.memoryCount ?? 0} memories` : '...'}
            </div>
          </div>
        </div>

        <div className="glass rounded-2xl mb-6 p-2 flex gap-1">
          <button
            className={`flex-1 py-2.5 rounded-xl text-sm font-medium transition-all ${activeTab === 'search' ? 'glass-tab-active text-white' : 'glass-tab text-white/50 hover:text-white/80'}`}
            onClick={() => setActiveTab('search')}
          >
            Search
          </button>
          <button
            className={`flex-1 py-2.5 rounded-xl text-sm font-medium transition-all ${activeTab === 'insert' ? 'glass-tab-active text-white' : 'glass-tab text-white/50 hover:text-white/80'}`}
            onClick={() => setActiveTab('insert')}
          >
            Insert
          </button>
          <button
            className={`flex-1 py-2.5 rounded-xl text-sm font-medium transition-all ${activeTab === 'stats' ? 'glass-tab-active text-white' : 'glass-tab text-white/50 hover:text-white/80'}`}
            onClick={() => setActiveTab('stats')}
          >
            Stats
          </button>
        </div>

        <div className="glass rounded-2xl p-6">
          {activeTab === 'search' && <SearchView />}
          {activeTab === 'insert' && <InsertView onInsert={refreshStats} />}
          {activeTab === 'stats' && <StatsView stats={stats} onRefresh={refreshStats} />}
        </div>
      </div>
    </div>
  );
}

export default App;

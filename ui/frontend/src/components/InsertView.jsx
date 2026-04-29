import { useState } from 'react';
import { api } from '../api';

export default function InsertView({ onInsert }) {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState(null);

  const handleInsert = async () => {
    if (!text.trim()) return;
    setLoading(true);
    const res = await api.insert(text);
    setLoading(false);
    setText('');
    setMessage({ type: 'success', text: `Inserted memory with ID: ${res?.id ?? '?'}` });
    if (onInsert) onInsert();
    setTimeout(() => setMessage(null), 3000);
  };

  return (
    <div className="space-y-4">
      <h2 className="text-2xl font-semibold text-white/90 mb-2">Insert Memory</h2>
      <textarea
        className="glass-input w-full h-48 rounded-xl p-4 text-white/90 placeholder-white/30 resize-none"
        placeholder="Enter text to store..."
        value={text}
        onChange={(e) => setText(e.target.value)}
      />
      <div className="flex justify-between items-center">
        <div className="text-sm text-white/40">{text.length} characters</div>
        <button className="glass-btn glass-btn-primary rounded-xl px-5 py-2.5 text-white font-medium" onClick={handleInsert} disabled={loading}>
          {loading ? <span className="inline-block w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" /> : 'Insert'}
        </button>
      </div>
      {message && (
        <div className={`glass rounded-xl px-4 py-3 text-sm ${message.type === 'success' ? 'text-emerald-300 border-emerald-400/20' : 'text-rose-300 border-rose-400/20'}`}>
          <span>{message.text}</span>
        </div>
      )}
    </div>
  );
}

let requestId = 0;
const pending = new Map();

window.dispatchResponse = (id, result) => {
  const resolver = pending.get(id);
  if (resolver) {
    pending.delete(id);
    resolver(result);
  }
};

export function callNim(method, args = {}) {
  return new Promise((resolve) => {
    requestId += 1;
    const id = requestId;
    pending.set(id, resolve);
    const payload = JSON.stringify({ id, method, args });
    if (window.external && window.external.invoke) {
      window.external.invoke(payload);
    } else {
      console.warn('Nim bridge not available');
      resolve(null);
    }
  });
}

export const api = {
  insert: (text) => callNim('insert', { text }),
  search: (query, topK = 10) => callNim('search', { query, topK }),
  delete: (id) => callNim('delete', { id }),
  update: (id, text) => callNim('update', { id, text }),
  stats: () => callNim('stats', {}),
  checkpoint: () => callNim('checkpoint', {}),
};

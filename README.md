# Adaline

A Nim vector search engine that turns text into **10240-bit sparse fingerprints** and searches them with HNSW + LSH, plus a lexical index for token-based matching. Results are fused with Reciprocal Rank Fusion and reranked by term coverage.

- **Conditional chunking** — long text splits into sentence-aware chunks when fingerprints approach saturation
- **Delete & reuse** — `deleteMemory()` removes from all indexes and returns slots to a freelist
- **Checkpoint / fast restart** — `checkpoint()` persists indexes to disk; restarts only replay WAL after the checkpoint
- **Single binary, no dependencies** — memory-mapped files, no external services

---

## How it works

**Insert:** Text → optional chunking → SDR encoder → 1280-byte fingerprint. The fingerprint goes into an HNSW graph, an LSH index, and a lexical inverted index. Parent text is logged to a WAL.

**Search:** Query → same SDR encoder → fingerprint. Semantic candidates come from LSH seeds + HNSW layer-0 descent. Lexical candidates come from an inverted index scored with Query Likelihood Model + Dirichlet smoothing. The two lists are merged with RRF, deduplicated to parent memories, and reranked by query term coverage.

**Storage:** Flat memory-mapped files with 256-byte headers (`ADLN` magic). Slots are dense indices (0, 1, 2…) rather than byte offsets. Pre-allocated 64 MiB chunks minimize remap frequency.

| File | Purpose |
|------|---------|
| `wal.bin` | Append-only text + metadata |
| `fingerprints.bin` | Fingerprint store (header + 1280-byte slots) |
| `graph.bin` | HNSW node store (header + 2056-byte slots) |
| `chunks.bin` | Parent→chunk mappings |
| `lsh.bin` | Persisted LSH index (checkpoint) |
| `lexical.bin` | Persisted lexical postings (checkpoint) |
| `corpus.bin` | Persisted corpus stats (checkpoint) |

---

## Quick start

```bash
# Build
nimble build_release

# Insert
./adaline insert "The quick brown fox"
./adaline insert "Nim is a systems programming language"

# Search
./adaline search "quick fox" 5

# Stats
./adaline stats

# Tests
nimble test

# Benchmarks
nimble benchmark
./benchmarks/benchmark_beir scifact
./benchmarks/benchmark_beir nfcorpus
./benchmarks/benchmark_longmemeval
```

---

## Benchmarks

Apple MacBook Air M2 (16 GB). Full tables and methodology in [`BENCHMARK.md`](BENCHMARK.md).

| Dataset | Corpus | Indexing | Query (top-100) | nDCG@10 | R@5 |
|---------|--------|----------|-----------------|---------|-----|
| SciFact | 5,183 docs | 2,452 docs/s | 226 q/s | 0.561 | — |
| NFCorpus | 3,633 docs | 2,318 docs/s | 366 q/s | 0.277 | — |
| LongMemEval-S | 500 questions | — | — | — | 94.6% |

SciFact: Recall@1 = 43.3%, MRR = 0.54. P50 latency ~4.4 ms for top-100.
NFCorpus: Precision@1 = 38.7%, MRR = 0.47. Hard medical retrieval task with sparse labels.
LongMemEval-S: R@1 = 78.2%, R@5 = 94.6%. Conversational memory retrieval.

---

## Configuration

`domain/entities/config.nim`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fingerprintBytes` | 1280 | Fingerprint size (10240 bits) |
| `tokenWeight` | 0.50 | Jaccard weight for token block |
| `bigramWeight` | 0.25 | Jaccard weight for char-bigram block |
| `contextWeight` | 0.25 | Jaccard weight for XOR-context block |
| `lshBands` / `lshRows` | 50 / 2 | Fingerprint LSH banding |
| `hnswMaxLayers` | 8 | HNSW layer count |
| `hnswMaxNeighbors` | 32 | Max edges per layer |
| `hnswEfConstruction` | 200 | HNSW build beam width |
| `hnswEfSearch` | 64 | HNSW query beam width |
| `dirichletMu` | 2000.0 | QLM smoothing parameter |
| `rrfK` | 60 | RRF constant |
| `rerankCoverageWeight` | 0.5 | Term-coverage boost weight |
| `chunkSaturationThreshold` | 0.6 | Chunk when any block exceeds this saturation |

---

## Design

Dependency flow: `Use Cases ← Domain ← Infrastructure`. Domain services call into `infrastructure/` directly. Not Clean Architecture — simplicity and performance over strict layers.

**Delete:** `deleteMemory()` removes from LSH, lexical, and HNSW. The slot goes to a freelist. HNSW nodes are marked `layerCount = 0` so traversal skips them. WAL does not persist tombstones; deleted memories reappear on restart until `checkpoint()` is called.

**Chunking:** Long documents are split into sentence-aware chunks with one-sentence overlap. Each chunk gets its own fingerprint. Prevents saturation and keeps fingerprints sparse. Short documents stay single-chunk.

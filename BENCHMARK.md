# Adaline Benchmarks

Run against BEIR datasets with the LSH → HNSW layer-0 search path and term-coverage reranking.

## Build

```bash
nimble benchmark
```

## Running Benchmarks

### BEIR (insert + query + quality)

```bash
./benchmarks/beir scifact    # ~5K docs, fast
./benchmarks/beir nfcorpus   # ~3.6K docs, fast
./benchmarks/beir msmarco    # ~8.8M docs, very slow, 1GB+ download
```

Datasets auto-download on first run and cache in `benchmarks/<name>/`.

### LongMemEval

```bash
python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='xiaowu0162/longmemeval-cleaned', filename='longmemeval_s_cleaned.json', repo_type='dataset', local_dir='benchmarks/data')"
./benchmarks/longmemeval
```

### CRUD (delete + update throughput)

```bash
nim c -d:release -o:benchmarks/crud benchmarks/crud.nim
./benchmarks/crud scifact 1000 1000
```

---

## Hardware

Apple MacBook Air M2 (16 GB), macOS, Apple SSD.

---

## SciFact

5,183 documents. 1,109 queries (300 with qrels).

### Indexing

| Statistic | Value |
|-----------|-------|
| Total time | 2.21 s |
| Throughput | 2,350 docs/s |
| P50 latency | 0.41 ms |
| P95 latency | 0.74 ms |
| P99 latency | 1.03 ms |

### Query (top-100)

| Statistic | Value |
|-----------|-------|
| Total time | 4.98 s |
| Throughput | 223 q/s |
| P50 latency | 4.50 ms |
| P95 latency | 5.72 ms |
| P99 latency | 6.37 ms |

### Retrieval Quality (300 judged queries)

| Metric | Value |
|--------|-------|
| Recall@1 | 42.92% |
| Recall@5 | 61.58% |
| Recall@10 | 67.83% |
| Recall@100 | 88.04% |
| Precision@1 | 44.33% |
| Precision@5 | 13.20% |
| Precision@10 | 7.47% |
| Precision@100 | 1.14% |
| MRR | 0.5331 |
| MAP | 0.5214 |
| nDCG@10 | 0.5566 |

### CRUD Throughput (on already-indexed SciFact)

| Operation | Count | Total | Throughput | P50 | P95 |
|-----------|-------|-------|------------|-----|-----|
| Delete | 1,000 | 0.86 s | 1,158 docs/s | 0.92 ms | 1.69 ms |
| Delete | 2,500 | 1.86 s | 1,348 docs/s | 0.74 ms | 1.34 ms |
| Update | 1,000 | 0.91 s | 1,099 docs/s | 0.89 ms | 1.64 ms |
| Update | 2,500 | 1.41 s | 1,769 docs/s | 0.51 ms | 0.95 ms |

Delete/update speed varies with graph density; sparser graphs heal faster.

### Notes

- P50 query latency is ~4.5 ms for top-100 semantic+lexical fusion.
- Term-coverage reranker contributes significantly — without it, nDCG@10 was ~0.36 in earlier experiments.
- Memory footprint: ~6.5 MB for fingerprints plus a few MB for in-memory indexes.
- SciFact documents are short; most do not trigger chunking.

---

## NFCorpus

3,633 documents. 3,237 queries (323 with qrels).

### Indexing

| Statistic | Value |
|-----------|-------|
| Total time | 1.67 s |
| Throughput | 2,175 docs/s |
| P50 latency | 0.46 ms |
| P95 latency | 0.72 ms |
| P99 latency | 1.03 ms |

### Query (top-100)

| Statistic | Value |
|-----------|-------|
| Total time | 8.96 s |
| Throughput | 361 q/s |
| P50 latency | 2.45 ms |
| P95 latency | 3.93 ms |
| P99 latency | 4.34 ms |

### Retrieval Quality (323 judged queries)

| Metric | Value |
|--------|-------|
| Recall@1 | 5.26% |
| Recall@5 | 11.27% |
| Recall@10 | 13.19% |
| Recall@100 | 23.41% |
| Precision@1 | 38.39% |
| Precision@5 | 25.82% |
| Precision@10 | 19.44% |
| Precision@100 | 6.51% |
| MRR | 0.4717 |
| MAP | 0.1276 |
| nDCG@10 | 0.2767 |

### Notes

- Medical literature with sparse relevance judgments and longer, more lexically diverse documents.
- Low recall because the task is passage retrieval from long abstracts with very few labeled positives per query.
- Query throughput is higher than SciFact because the corpus is smaller. HNSW search time scales sub-linearly with corpus size.
- Some medical abstracts are long enough to trigger chunking.

---

## LongMemEval-S

500 questions. Each has ~53 conversation sessions (~115K tokens).

> **Note:** The numbers below measure only the **retrieval component** — whether the correct session(s) appear in the top-k results. The official LongMemEval benchmark is a generation + LLM-as-judge task that scores answer correctness, not retrieval recall. These results show how well Adaline retrieves relevant context, but they are not directly comparable to published LongMemEval accuracy scores.

### Retrieval Quality

| Metric | Value |
|--------|-------|
| R@1 | 78.20% |
| R@5 | 94.60% |
| R@10 | 96.80% |

### Per-Category R@5

| Category | R@5 | Count |
|----------|-----|-------|
| knowledge-update | 100.00% | 78/78 |
| single-session-user | 98.57% | 69/70 |
| multi-session | 96.99% | 129/133 |
| temporal-reasoning | 95.49% | 127/133 |
| single-session-assistant | 91.07% | 51/56 |
| single-session-preference | 63.33% | 19/30 |

### Notes

- Correct session is in the top 5 for 473/500 questions.
- Knowledge updates are perfect (100% R@5).
- Single-session preference is the weak spot (63.3%). Preferences are often implicit and require inference beyond literal text matching.
- Conditional chunking is essential here. Sessions average ~115K tokens, which would saturate a single fingerprint.

---

## MS MARCO

Supported but not benchmarked due to scale (~8.8M passages, ~1 GB download, ~11 GB fingerprints, ~18 GB graph).

```bash
./benchmarks/beir msmarco
```

At 8.8M docs, each query visits roughly `64 × log(N)` ≈ 1,000–2,000 nodes. The bottleneck shifts to memory bandwidth (cold node access) and WAL replay time on startup.

---

## Methodology

- **Distance**: `1.0 - weightedJaccard` with block weights 50% (tokens), 25% (bigrams), 25% (context).
- **Search**: LSH seeds → HNSW layer-0 descent (efSearch=64).
- **Lexical**: Query Likelihood Model with Dirichlet smoothing (μ=2000).
- **Fusion**: RRF (k=60).
- **Reranking**: Term-coverage boost (weight=0.5) on top-K merged results.
- **Metrics**: Computed against BEIR qrels (binary relevance), averaged only over queries with judgments. nDCG uses standard `1/log2(rank+1)` gain.

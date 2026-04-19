# Adaline Benchmarks

Run against BEIR datasets with the LSH → HNSW layer-0 search path and term-coverage reranking.

## Build

```bash
nimble benchmark
```

## Running Benchmarks

### BEIR

```bash
./benchmarks/benchmark_beir scifact    # ~5K docs, fast
./benchmarks/benchmark_beir nfcorpus   # ~3.6K docs, fast
./benchmarks/benchmark_beir msmarco    # ~8.8M docs, very slow, 1GB+ download
```

Datasets auto-download on first run and cache in `benchmarks/<name>/`.

### LongMemEval

```bash
python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='xiaowu0162/longmemeval-cleaned', filename='longmemeval_s_cleaned.json', repo_type='dataset', local_dir='benchmarks/data')"
./benchmarks/benchmark_longmemeval
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
| Total time | 2.11 s |
| Throughput | 2,452 docs/s |
| P50 latency | 0.40 ms |
| P95 latency | 0.71 ms |
| P99 latency | 0.99 ms |

### Query (top-100)

| Statistic | Value |
|-----------|-------|
| Total time | 4.90 s |
| Throughput | 226 q/s |
| P50 latency | 4.44 ms |
| P95 latency | 5.62 ms |
| P99 latency | 6.09 ms |

### Retrieval Quality (300 judged queries)

| Metric | Value |
|--------|-------|
| Recall@1 | 43.25% |
| Recall@5 | 61.58% |
| Recall@10 | 68.50% |
| Recall@100 | 88.04% |
| Precision@1 | 44.67% |
| Precision@5 | 13.20% |
| Precision@10 | 7.53% |
| Precision@100 | 1.14% |
| MRR | 0.5358 |
| MAP | 0.5241 |
| nDCG@10 | 0.5605 |

### Notes

- P50 query latency is ~4.4 ms for top-100 semantic+lexical fusion.
- Term-coverage reranker contributes significantly — without it, nDCG@10 was ~0.36 in earlier experiments.
- Memory footprint: ~6.5 MB for fingerprints plus a few MB for in-memory indexes.
- SciFact documents are short; most do not trigger chunking.

---

## NFCorpus

3,633 documents. 3,237 queries (323 with qrels).

### Indexing

| Statistic | Value |
|-----------|-------|
| Total time | 1.57 s |
| Throughput | 2,318 docs/s |
| P50 latency | 0.44 ms |
| P95 latency | 0.66 ms |
| P99 latency | 0.97 ms |

### Query (top-100)

| Statistic | Value |
|-----------|-------|
| Total time | 8.85 s |
| Throughput | 366 q/s |
| P50 latency | 2.45 ms |
| P95 latency | 3.86 ms |
| P99 latency | 4.16 ms |

### Retrieval Quality (323 judged queries)

| Metric | Value |
|--------|-------|
| Recall@1 | 5.26% |
| Recall@5 | 11.26% |
| Recall@10 | 13.19% |
| Recall@100 | 23.42% |
| Precision@1 | 38.70% |
| Precision@5 | 25.82% |
| Precision@10 | 19.50% |
| Precision@100 | 6.51% |
| MRR | 0.4729 |
| MAP | 0.1276 |
| nDCG@10 | 0.2773 |

### Notes

- Medical literature with sparse relevance judgments and longer, more lexically diverse documents.
- Low recall because the task is passage retrieval from long abstracts with very few labeled positives per query.
- Query throughput is higher than SciFact because the corpus is smaller. HNSW search time scales sub-linearly with corpus size.
- Some medical abstracts are long enough to trigger chunking.

---

## LongMemEval-S

500 questions. Each has ~53 conversation sessions (~115K tokens).

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
./benchmarks/benchmark_beir msmarco
```

At 8.8M docs, each query visits roughly `64 × log(N)` ≈ 1,000–2,000 nodes. The bottleneck shifts to memory bandwidth (cold node access) and WAL replay time on startup.

---

## Methodology

- **Distance**: `1.0 - weightedJaccard` with block weights 50% (tokens), 25% (bigrams), 25% (context).
- **Search**: LSH seeds → HNSW layer-0 descent (efSearch=64).
- **Lexical**: Query Likelihood Model with Dirichlet smoothing (μ=2000).
- **Fusion**: Weighted RRF (k=60, semanticWeight=0.5, lexicalWeight=1.0).
- **Reranking**: Term-coverage boost (weight=0.5) on top-K merged results.
- **Metrics**: Computed against BEIR qrels (binary relevance), averaged only over queries with judgments. nDCG uses standard `1/log2(rank+1)` gain.

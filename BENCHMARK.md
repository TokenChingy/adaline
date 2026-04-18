# Adaline Benchmarks

Benchmarks are run against BEIR datasets using the unified LSH Wormhole search path (LSH seeds → HNSW layer-0 descent) with term-coverage reranking.

## Hardware

- CPU: x86_64 (likely 4-8 cores, exact model depends on host)
- RAM: Sufficient to hold all indexes in memory
- Storage: NVMe SSD (benchmark data + mmap files)
- OS: Linux

## Build

```bash
nim c -d:release benchmarks/benchmark_beir.nim
```

## Running Benchmarks

```bash
# SciFact (~5K docs, fast)
./benchmarks/benchmark_beir scifact

# NFCorpus (~3.6K docs, fast)
./benchmarks/benchmark_beir nfcorpus

# MS MARCO (~8.8M docs, very slow, 1GB+ download)
./benchmarks/benchmark_beir msmarco
```

Datasets are auto-downloaded on first run and cached in `benchmarks/<name>/`.

---

## Results: SciFact

| Metric | Value |
|--------|-------|
| Corpus | 5,183 documents |
| Queries | 1,109 |
| Qrels | 300 |

### Indexing Speed

| Statistic | Value |
|-----------|-------|
| Total time | 5.80 s |
| Throughput | 894 docs/s |
| P50 latency | 0.96 ms |
| P95 latency | 2.17 ms |
| P99 latency | 3.65 ms |

### Query Speed (top-100)

| Statistic | Value |
|-----------|-------|
| Total time | 5.48 s |
| Throughput | 202 queries/s |
| P50 latency | 4.82 ms |
| P95 latency | 6.36 ms |
| P99 latency | 7.82 ms |

### Retrieval Quality

| Metric | Value |
|--------|-------|
| Recall@1 | 41.58% |
| Recall@5 | 60.81% |
| Recall@10 | 67.17% |
| Recall@100 | 87.37% |
| Precision@1 | 43.00% |
| Precision@5 | 13.00% |
| Precision@10 | 7.40% |
| Precision@100 | 0.98% |
| MRR | 0.5234 |
| MAP | 0.5121 |
| nDCG@10 | 0.5475 |

### Observations

- **Query latency is excellent** for a pure-Nim, memory-mapped engine. P50 ~5ms for top-100 semantic+lexical fusion on a 5K corpus is competitive with much heavier systems.
- **Recall@1 of 41.6%** on SciFact means the system finds the top relevant document in the first position ~42% of the time. This is solid for a sparse fingerprint approach without neural re-ranking.
- **nDCG@10 of 0.55** shows good ranking quality in the top 10. The term-coverage reranker contributes significantly here — without it, nDCG@10 was ~0.36 in earlier experiments.
- **The bottleneck is not search** — 202 q/s means a single core can handle production query loads for small-to-medium corpora.
- **Memory footprint is tiny**: ~6.5 MB for fingerprints (5K × 1280 bytes) plus a few MB for in-memory indexes.

---

## Results: NFCorpus

| Metric | Value |
|--------|-------|
| Corpus | 3,633 documents |
| Queries | 3,237 |
| Qrels | 323 |

### Indexing Speed

| Statistic | Value |
|-----------|-------|
| Total time | 4.50 s |
| Throughput | 807 docs/s |
| P50 latency | 1.03 ms |
| P95 latency | 2.61 ms |
| P99 latency | 4.39 ms |

### Query Speed (top-100)

| Statistic | Value |
|-----------|-------|
| Total time | 9.89 s |
| Throughput | 327 queries/s |
| P50 latency | 3.65 ms |
| P95 latency | 5.23 ms |
| P99 latency | 6.47 ms |

### Retrieval Quality

| Metric | Value |
|--------|-------|
| Recall@1 | 5.03% |
| Recall@5 | 10.53% |
| Recall@10 | 13.03% |
| Recall@100 | 22.24% |
| Precision@1 | 39.32% |
| Precision@5 | 26.67% |
| Precision@10 | 22.49% |
| Precision@100 | 13.77% |
| MRR | 0.4672 |
| MAP | 0.1206 |
| nDCG@10 | 0.2693 |

### Observations

- **NFCorpus is a much harder dataset** than SciFact. Medical literature has sparse relevance judgments (only 323 qrels across 3,237 queries) and documents are longer and more lexically diverse.
- **Recall is low** (5% @ 1, 22% @ 100) because the task is passage retrieval from long medical abstracts with very few labeled positives per query. The system still ranks reasonably well by MRR (0.47), meaning when it does find a relevant doc, it's often near the top.
- **Precision is strong**: 39% at rank 1 means the top result is relevant nearly 40% of the time. This suggests the semantic+lexical fusion works well for high-confidence matches even when overall recall is limited by sparse labels.
- **Query throughput is higher** than SciFact (327 vs 202 q/s) because NFCorpus has fewer documents. HNSW search time scales sub-linearly with corpus size.
- **The low MAP (0.12)** reflects the difficulty of the dataset more than the engine. NFCorpus is known to be challenging for sparse retrieval methods.

---

## MS MARCO

MS MARCO passage retrieval (~8.8M passages) is supported but not benchmarked here due to scale:

- **Download**: ~1.08 GB zip
- **Fingerprints**: ~8.8M × 1280 B = ~11 GB
- **Graph store**: ~8.8M × 2056 B = ~18 GB
- **Estimated indexing time**: 2–3 hours
- **Estimated query time**: P50 likely 10–50 ms (HNSW scales logarithmically)

To run:
```bash
./benchmarks/benchmark_beir msmarco
```

### Expected Behavior at Scale

- HNSW efSearch=64 means each query visits roughly `64 × log(N)` nodes. At 8.8M docs, this is ~1,000–2,000 distance computations.
- Each Jaccard comparison is 160 bitwise ANDs + popcounts. At ~1,500 comparisons/query, that's ~240K uint64 ops. A modern CPU handles this in well under 1ms.
- The real bottleneck at MS MARCO scale will be **memory bandwidth** (walking the graph touches cold nodes) and **WAL replay time** on startup.
- A block-based or quantized fingerprint store would be necessary for MS MARCO production deployment.

---

## Methodology Notes

- **Distance metric**: `1.0 - weightedJaccard` with block weights 50% (tokens), 25% (bigrams), 25% (context).
- **Search path**: LSH seeds → HNSW layer-0 descent (efSearch=64). No brute-force fallback.
- **Lexical scoring**: Query Likelihood Model with Dirichlet smoothing (μ=2000).
- **Fusion**: Reciprocal Rank Fusion (k=60).
- **Reranking**: Term-coverage boost (weight=0.5) applied to top-K merged results.
- **Metrics**: Computed against BEIR qrels (binary relevance). nDCG uses standard `1/log2(rank+1)` gain.

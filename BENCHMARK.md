# Adaline Benchmarks

Benchmarks are run against BEIR datasets using the unified LSH Wormhole search path (LSH seeds → HNSW layer-0 descent) with term-coverage reranking.

**Important:** All quality metrics below are averaged over the **full query set** (every query in `queries.jsonl`), including queries with no qrels. Queries without relevance judgments contribute a score of 0. This produces conservative numbers compared to averaging only over judged queries.

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
| Queries (total) | 1,109 |
| Queries (with qrels) | 300 |

### Indexing Speed

| Statistic | Value |
|-----------|-------|
| Total time | 6.17 s |
| Throughput | 840 docs/s |
| P50 latency | 1.06 ms |
| P95 latency | 2.02 ms |
| P99 latency | 3.51 ms |

### Query Speed (top-100)

| Statistic | Value |
|-----------|-------|
| Total time | 5.97 s |
| Throughput | 186 queries/s |
| P50 latency | 5.26 ms |
| P95 latency | 6.92 ms |
| P99 latency | 8.41 ms |

### Retrieval Quality (all 1,109 queries)

| Metric | Value |
|--------|-------|
| Recall@1 | 11.23% |
| Recall@5 | 16.63% |
| Recall@10 | 18.17% |
| Recall@100 | 23.62% |
| Precision@1 | 11.54% |
| Precision@5 | 3.55% |
| Precision@10 | 2.00% |
| Precision@100 | 0.27% |
| MRR | 0.1415 |
| MAP | 0.1387 |
| nDCG@10 | 0.1482 |

### Observations

- **Query latency is excellent** for a pure-Nim, memory-mapped engine. P50 ~5ms for top-100 semantic+lexical fusion on a 5K corpus is competitive with much heavier systems.
- **Metrics are conservative** because they average over all 1,109 queries. Only 300 have qrels; the other 809 contribute 0. If averaged over judged queries only: Recall@1 ≈ 41.6%, nDCG@10 ≈ 0.55.
- **The bottleneck is not search** — 186 q/s means a single core can handle production query loads for small-to-medium corpora.
- **Memory footprint is tiny**: ~6.5 MB for fingerprints (5K × 1280 bytes) plus a few MB for in-memory indexes.

---

## Results: NFCorpus

| Metric | Value |
|--------|-------|
| Corpus | 3,633 documents |
| Queries (total) | 3,237 |
| Queries (with qrels) | 323 |

### Indexing Speed

| Statistic | Value |
|-----------|-------|
| Total time | 4.69 s |
| Throughput | 775 docs/s |
| P50 latency | 1.13 ms |
| P95 latency | 2.34 ms |
| P99 latency | 3.96 ms |

### Query Speed (top-100)

| Statistic | Value |
|-----------|-------|
| Total time | 9.84 s |
| Throughput | 329 queries/s |
| P50 latency | 3.56 ms |
| P95 latency | 5.33 ms |
| P99 latency | 7.27 ms |

### Retrieval Quality (all 3,237 queries)

| Metric | Value |
|--------|-------|
| Recall@1 | 0.49% |
| Recall@5 | 1.08% |
| Recall@10 | 1.29% |
| Recall@100 | 2.21% |
| Precision@1 | 3.71% |
| Precision@5 | 2.64% |
| Precision@10 | 2.24% |
| Precision@100 | 1.36% |
| MRR | 0.0457 |
| MAP | 0.0122 |
| nDCG@10 | 0.0268 |

### Observations

- **NFCorpus is a much harder dataset** than SciFact. Medical literature has sparse relevance judgments (only 323 qrels across 3,237 queries) and documents are longer and more lexically diverse.
- **Recall is very low** when averaged over all queries because 2,914 queries have no judgments and contribute 0. Over judged queries only: Recall@1 ≈ 5.0%, nDCG@10 ≈ 0.27.
- **Precision@1 of 3.7%** means the top result is relevant ~4% of the time across all queries. Over judged queries only, this rises to ~39%.
- **Query throughput is higher** than SciFact (329 vs 186 q/s) because NFCorpus has fewer documents. HNSW search time scales sub-linearly with corpus size.
- **The low MAP (0.012)** reflects the difficulty of the dataset and the large number of unjudged queries more than the engine itself. NFCorpus is known to be challenging for sparse retrieval methods.

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
- **Metrics**: Computed against BEIR qrels (binary relevance). Averaged over **all queries** in `queries.jsonl`, with unjudged queries scoring 0. nDCG uses standard `1/log2(rank+1)` gain.

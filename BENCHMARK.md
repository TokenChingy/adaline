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

### BEIR

```bash
# SciFact (~5K docs, fast)
./benchmarks/benchmark_beir scifact

# NFCorpus (~3.6K docs, fast)
./benchmarks/benchmark_beir nfcorpus

# MS MARCO (~8.8M docs, very slow, 1GB+ download)
./benchmarks/benchmark_beir msmarco
```

Datasets are auto-downloaded on first run and cached in `benchmarks/<name>/`.

### LongMemEval

```bash
# Download dataset first
python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='xiaowu0162/longmemeval-cleaned', filename='longmemeval_s_cleaned.json', repo_type='dataset', local_dir='benchmarks/data')"

# Run retrieval benchmark
nim c -d:release benchmarks/benchmark_longmemeval.nim
./benchmarks/benchmark_longmemeval
```

---

## Results: SciFact

| Metric | Value |
|--------|-------|
| Corpus | 5,183 documents |
| Queries | 1,109 total / 300 with qrels |

### Indexing Speed

| Statistic | Value |
|-----------|-------|
| Total time | 7.64 s |
| Throughput | 679 docs/s |
| P50 latency | 1.22 ms |
| P95 latency | 2.93 ms |
| P99 latency | 5.97 ms |

### Query Speed (top-100)

| Statistic | Value |
|-----------|-------|
| Total time | 6.56 s |
| Throughput | 169 queries/s |
| P50 latency | 5.62 ms |
| P95 latency | 8.36 ms |
| P99 latency | 10.05 ms |

### Retrieval Quality (over 300 judged queries)

| Metric | Value |
|--------|-------|
| Recall@1 | 41.92% |
| Recall@5 | 61.31% |
| Recall@10 | 67.50% |
| Recall@100 | 88.37% |
| Precision@1 | 43.33% |
| Precision@5 | 13.13% |
| Precision@10 | 7.43% |
| Precision@100 | 0.99% |
| MRR | 0.5263 |
| MAP | 0.5146 |
| nDCG@10 | 0.5503 |

### Observations

- **Query latency is excellent** for a pure-Nim, memory-mapped engine. P50 ~5.6ms for top-100 semantic+lexical fusion is competitive with much heavier systems.
- **Recall@1 of 41.9%** means the system finds the top relevant document first ~42% of the time. Solid for a sparse fingerprint approach without neural re-ranking.
- **nDCG@10 of 0.55** shows good ranking quality in the top 10. The term-coverage reranker contributes significantly — without it, nDCG@10 was ~0.36 in earlier experiments.
- **The bottleneck is not search** — 169 q/s means a single core can handle production query loads for small-to-medium corpora.
- **Memory footprint is tiny**: ~6.5 MB for fingerprints (5K × 1280 bytes) plus a few MB for in-memory indexes.

---

## Results: NFCorpus

| Metric | Value |
|--------|-------|
| Corpus | 3,633 documents |
| Queries | 3,237 total / 323 with qrels |

### Indexing Speed

| Statistic | Value |
|-----------|-------|
| Total time | 4.23 s |
| Throughput | 859 docs/s |
| P50 latency | 0.99 ms |
| P95 latency | 2.31 ms |
| P99 latency | 4.01 ms |

### Query Speed (top-100)

| Statistic | Value |
|-----------|-------|
| Total time | 9.83 s |
| Throughput | 329 queries/s |
| P50 latency | 3.63 ms |
| P95 latency | 5.17 ms |
| P99 latency | 6.49 ms |

### Retrieval Quality (over 323 judged queries)

| Metric | Value |
|--------|-------|
| Recall@1 | 5.24% |
| Recall@5 | 10.78% |
| Recall@10 | 13.14% |
| Recall@100 | 22.39% |
| Precision@1 | 40.25% |
| Precision@5 | 26.55% |
| Precision@10 | 22.58% |
| Precision@100 | 13.76% |
| MRR | 0.4774 |
| MAP | 0.1236 |
| nDCG@10 | 0.2730 |

### Observations

- **NFCorpus is a much harder dataset** than SciFact. Medical literature has sparse relevance judgments and longer, more lexically diverse documents.
- **Recall is low** (5.2% @ 1, 22.4% @ 100) because the task is passage retrieval from long medical abstracts with very few labeled positives per query. The system still ranks reasonably well by MRR (0.48), meaning when it does find a relevant doc, it's often near the top.
- **Precision@1 of 40%** means the top result is relevant 40% of the time. This suggests the semantic+lexical fusion works well for high-confidence matches even when overall recall is limited by sparse labels.
- **Query throughput is higher** than SciFact (329 vs 169 q/s) because NFCorpus has fewer documents. HNSW search time scales sub-linearly with corpus size.
- **The low MAP (0.12)** reflects the difficulty of the dataset more than the engine. NFCorpus is known to be challenging for sparse retrieval methods.

---

## Results: LongMemEval-S

LongMemEval-S tests long-term conversational memory retrieval. Each of the 500 questions has its own haystack of ~53 conversation sessions (~115K tokens total). The benchmark evaluates whether the correct session is retrieved when searching with the question.

| Metric | Value |
|--------|-------|
| Questions | 500 |
| Avg sessions per question | ~53 |
| Avg tokens per question | ~115,000 |

### Retrieval Quality

| Metric | Value |
|--------|-------|
| R@1 | 78.60% |
| R@5 | 91.60% |
| R@10 | 93.00% |

### Per-Category R@5

| Category | R@5 | Count |
|----------|-----|-------|
| knowledge-update | 98.72% | 77/78 |
| single-session-assistant | 98.21% | 55/56 |
| single-session-user | 95.71% | 67/70 |
| multi-session | 93.23% | 124/133 |
| temporal-reasoning | 90.98% | 121/133 |
| single-session-preference | 46.67% | 14/30 |

### Observations

- **R@5 of 91.6%** is excellent for a pure sparse retrieval system. The correct session is in the top 5 for over 9 out of 10 questions.
- **R@1 of 78.6%** means Adaline finds the exact right session first ~79% of the time without any LLM re-ranking.
- **Knowledge updates and single-session recall are nearly perfect** (>95%). These are the easiest categories — the answer is directly stated in one session.
- **Multi-session reasoning at 93.2%** is surprisingly strong. Adaline's partitioned SDR (tokens + bigrams + XOR context) captures enough semantic signal to retrieve sessions that contain distributed facts.
- **Temporal reasoning at 91.0%** is also strong. Even though Adaline has no explicit date parsing, the lexical index picks up temporal references and the semantic fingerprint captures event sequences.
- **Single-session preference is the weak spot at 46.7%.** Preferences are often implicit or indirectly stated ("I don't like spicy food" vs "the food was too spicy"). These require inference beyond literal text matching, which is harder for sparse fingerprints without an LLM reader.
- **Comparison to other systems:** Published LongMemEval retrieval scores (R@5) for other systems range from ~85% (BM25) to ~96% (vector search with all-MiniLM-L6-v2). Adaline's 91.6% sits squarely in the competitive range, despite using no neural embeddings at all.

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
- **Search path**: LSH seeds → HNSW layer-0 descent (efSearch=64).
- **Lexical scoring**: Query Likelihood Model with Dirichlet smoothing (μ=2000).
- **Fusion**: Reciprocal Rank Fusion (k=60).
- **Reranking**: Term-coverage boost (weight=0.5) applied to top-K merged results.
- **Metrics**: Computed against BEIR qrels (binary relevance), averaged only over queries that have judgments. nDCG uses standard `1/log2(rank+1)` gain.

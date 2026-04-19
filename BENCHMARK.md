# Adaline Benchmarks

Benchmarks are run against BEIR datasets using the unified LSH Wormhole search path (LSH seeds → HNSW layer-0 descent) with term-coverage reranking.

## Build

```bash
nimble benchmark
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
./benchmarks/benchmark_longmemeval
```

---

## Hardware: Apple MacBook Air M2 (16 GB)

- CPU: Apple M2 (8 cores: 4P + 4E)
- RAM: 16 GB unified memory
- Storage: Apple SSD (benchmark data + mmap files)
- OS: macOS

## Results: SciFact

| Metric | Value |
|--------|-------|
| Corpus | 5,183 documents |
| Queries | 1,109 total / 300 with qrels |

### Indexing Speed

| Statistic | Value |
|-----------|-------|
| Total time | 1.94 s |
| Throughput | 2,673 docs/s |
| P50 latency | 0.37 ms |
| P95 latency | 0.65 ms |
| P99 latency | 0.89 ms |

### Query Speed (top-100)

| Statistic | Value |
|-----------|-------|
| Total time | 4.81 s |
| Throughput | 231 queries/s |
| P50 latency | 4.38 ms |
| P95 latency | 5.48 ms |
| P99 latency | 6.01 ms |

### Retrieval Quality (over 300 judged queries)

| Metric | Value |
|--------|-------|
| Recall@1 | 43.25% |
| Recall@5 | 61.47% |
| Recall@10 | 68.06% |
| Recall@100 | 88.04% |
| Precision@1 | 44.67% |
| Precision@5 | 13.13% |
| Precision@10 | 7.47% |
| Precision@100 | 1.13% |
| MRR | 0.5361 |
| MAP | 0.5242 |
| nDCG@10 | 0.5591 |

### Observations

- **The M2 is ~4.4× faster at indexing** than the Intel i7-8559U (2,673 vs 608 docs/s), thanks to deterministic hashing (no MinHash overhead) and efficient ARM64 codegen.
- **Query latency is excellent** for a pure-Nim, memory-mapped engine. P50 ~4.2ms for top-100 semantic+lexical fusion is competitive with much heavier systems.
- **Recall@1 of 43.3%** means the system finds the top relevant document first ~43% of the time. Solid for a sparse fingerprint approach without neural re-ranking.
- **nDCG@10 of 0.56** shows good ranking quality in the top 10. The term-coverage reranker contributes significantly — without it, nDCG@10 was ~0.36 in earlier experiments.
- **The bottleneck is not search** — 241 q/s means a single core can handle production query loads for small-to-medium corpora.
- **Memory footprint is tiny**: ~6.5 MB for fingerprints (5K × 1280 bytes) plus a few MB for in-memory indexes.
- **Chunking overhead**: SciFact documents are short, so most do not trigger chunking.

---

## Results: NFCorpus

| Metric | Value |
|--------|-------|
| Corpus | 3,633 documents |
| Queries | 3,237 total / 323 with qrels |

### Indexing Speed

| Statistic | Value |
|-----------|-------|
| Total time | 2.29 s |
| Throughput | 1,590 docs/s |
| P50 latency | 0.61 ms |
| P95 latency | 1.15 ms |
| P99 latency | 1.80 ms |

### Query Speed (top-100)

| Statistic | Value |
|-----------|-------|
| Total time | 6.21 s |
| Throughput | 521 queries/s |
| P50 latency | 2.10 ms |
| P95 latency | 3.53 ms |
| P99 latency | 3.76 ms |

### Retrieval Quality (over 323 judged queries)

| Metric | Value |
|--------|-------|
| Recall@1 | 4.86% |
| Recall@5 | 11.09% |
| Recall@10 | 13.11% |
| Recall@100 | 21.26% |
| Precision@1 | 35.29% |
| Precision@5 | 26.61% |
| Precision@10 | 22.55% |
| Precision@100 | 13.75% |
| MRR | 0.4470 |
| MAP | 0.1203 |
| nDCG@10 | 0.2666 |

### Observations

- **NFCorpus is a much harder dataset** than SciFact. Medical literature has sparse relevance judgments and longer, more lexically diverse documents.
- **Recall is low** (4.9% @ 1, 21.3% @ 100) because the task is passage retrieval from long medical abstracts with very few labeled positives per query. The system still ranks reasonably well by MRR (0.45), meaning when it does find a relevant doc, it's often near the top.
- **Precision@1 of 35%** means the top result is relevant 35% of the time. This suggests the semantic+lexical fusion works well for high-confidence matches even when overall recall is limited by sparse labels.
- **Query throughput is higher** than SciFact (521 vs 241 q/s) because NFCorpus queries run against a smaller corpus. HNSW search time scales sub-linearly with corpus size.
- **The low MAP (0.12)** reflects the difficulty of the dataset more than the engine. NFCorpus is known to be challenging for sparse retrieval methods.
- **Chunking impact**: Some medical abstracts are long enough to trigger chunking, which adds insert overhead but can improve precision by retrieving specific passages rather than whole abstracts.

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
| R@1 | 78.00% |
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

### Observations

- **R@5 of 94.6%** is excellent for a pure sparse retrieval system. The correct session is in the top 5 for 95 out of 100 questions.
- **R@1 of 78.0%** means Adaline finds the exact right session first ~78% of the time without any LLM re-ranking.
- **Knowledge updates are perfect** (100% R@5). These are the easiest categories — the answer is directly stated in one session.
- **Multi-session reasoning at 96.2%** is surprisingly strong. Adaline's partitioned SDR (tokens + bigrams + XOR context) captures enough semantic signal to retrieve sessions that contain distributed facts.
- **Temporal reasoning at 94.7%** is also strong. Even though Adaline has no explicit date parsing, the lexical index picks up temporal references and the semantic fingerprint captures event sequences.
- **Single-session preference is the weak spot at 63.3%.** Preferences are often implicit or indirectly stated ("I don't like spicy food" vs "the food was too spicy"). These require inference beyond literal text matching, which is harder for sparse fingerprints without an LLM reader.
- **Chunking impact:** This is where conditional chunking shines. LongMemEval sessions average ~115K tokens, which would saturate a single fingerprint. By splitting long sessions into chunks, each chunk maintains a sparse, distinctive fingerprint.
- **Comparison to other systems:** Published LongMemEval retrieval scores (R@5) for other systems range from ~85% (BM25) to ~96% (vector search with all-MiniLM-L6-v2). Adaline's 94.4% sits squarely in the competitive range, despite using no neural embeddings at all.

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
- **Search path**: Fingerprint LSH seeds → HNSW layer-0 descent (efSearch=64).
- **Lexical scoring**: Query Likelihood Model with Dirichlet smoothing (μ=2000).
- **Fusion**: Weighted Reciprocal Rank Fusion (k=60, semanticWeight=0.5, lexicalWeight=1.0).
- **Reranking**: Term-coverage boost (weight=0.5) applied to top-K merged results.
- **Metrics**: Computed against BEIR qrels (binary relevance), averaged only over queries that have judgments. nDCG uses standard `1/log2(rank+1)` gain.

---

## Historical: Intel Core i7-8559U (Linux, 32 GB)

For comparison, here are earlier results from an Intel Core i7-8559U (4 cores / 8 threads, 2.7 GHz base / 4.5 GHz turbo) with 32 GB RAM and NVMe SSD on Linux.

### SciFact (Intel)

| Statistic | Value |
|-----------|-------|
| Indexing throughput | 608 docs/s |
| Query throughput | 148 queries/s |
| Recall@1 | 40.72% |
| Recall@5 | 60.08% |
| Recall@10 | 67.83% |
| nDCG@10 | 0.5455 |
| MRR | 0.5166 |
| MAP | 0.5080 |

### NFCorpus (Intel)

| Statistic | Value |
|-----------|-------|
| Indexing throughput | 594 docs/s |
| Query throughput | 307 queries/s |
| Recall@1 | 4.79% |
| Recall@5 | 10.95% |
| Recall@10 | 13.04% |
| nDCG@10 | 0.2649 |
| MRR | 0.4424 |
| MAP | 0.1205 |

### LongMemEval-S (Intel)

| Metric | Value |
|--------|-------|
| R@1 | 79.00% |
| R@5 | 94.00% |
| R@10 | 96.00% |

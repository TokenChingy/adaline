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
./benchmarks/beir arguana    # ~8.7K docs, adversarial counter-arguments
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
| Total time | 2.10 s |
| Throughput | 2,465 docs/s |
| P50 latency | 0.39 ms |
| P95 latency | 0.74 ms |
| P99 latency | 1.17 ms |

### Query (top-100)

| Statistic | Value |
|-----------|-------|
| Total time | 4.99 s |
| Throughput | 222 q/s |
| P50 latency | 4.49 ms |
| P95 latency | 5.74 ms |
| P99 latency | 7.03 ms |

### Retrieval Quality (300 judged queries)

| Metric | Value |
|--------|-------|
| Recall@1 | 43.47% |
| Recall@5 | 63.28% |
| Recall@10 | 72.07% |
| Recall@100 | 86.49% |
| Precision@1 | 44.67% |
| Precision@5 | 13.53% |
| Precision@10 | 7.90% |
| Precision@100 | 1.05% |
| MRR | 0.5475 |
| MAP | 0.5371 |
| nDCG@10 | 0.5794 |

### CRUD Throughput (on already-indexed SciFact)

| Operation | Count | Total | Throughput | P50 | P95 |
|-----------|-------|-------|------------|-----|-----|
| Delete | 1,000 | 0.78 s | 1,282 docs/s | 0.84 ms | 1.52 ms |
| Update | 1,000 | 0.83 s | 1,200 docs/s | 0.84 ms | 1.44 ms |

Delete/update speed varies with graph density; sparser graphs heal faster.

### Notes

- P50 query latency is ~4.5 ms for top-100 semantic+lexical fusion.
- Term-coverage reranker contributes significantly — without it, nDCG@10 was ~0.36 in earlier experiments.
- Lowering `rrfK` from 60 to 10 improved SciFact nDCG@10 by ~0.02 (sharper rank discrimination).
- Memory footprint: ~6.5 MB for fingerprints plus a few MB for in-memory indexes.
- SciFact documents are short; most do not trigger chunking.

---

## NFCorpus

3,633 documents. 3,237 queries (323 with qrels).

### Indexing

| Statistic | Value |
|-----------|-------|
| Total time | 1.49 s |
| Throughput | 2,445 docs/s |
| P50 latency | 0.42 ms |
| P95 latency | 0.63 ms |
| P99 latency | 0.90 ms |

### Query (top-100)

| Statistic | Value |
|-----------|-------|
| Total time | 8.69 s |
| Throughput | 372 q/s |
| P50 latency | 2.36 ms |
| P95 latency | 3.80 ms |
| P99 latency | 4.21 ms |

### Retrieval Quality (323 judged queries)

| Metric | Value |
|--------|-------|
| Recall@1 | 5.40% |
| Recall@5 | 11.20% |
| Recall@10 | 13.27% |
| Recall@100 | 21.71% |
| Precision@1 | 38.70% |
| Precision@5 | 25.82% |
| Precision@10 | 19.69% |
| Precision@100 | 5.40% |
| MRR | 0.4720 |
| MAP | 0.1256 |
| nDCG@10 | 0.2791 |

### Notes

- Medical literature with sparse relevance judgments and longer, more lexically diverse documents.
- Low recall because the task is passage retrieval from long abstracts with very few labeled positives per query.
- Query throughput is higher than SciFact because the corpus is smaller. HNSW search time scales sub-linearly with corpus size.
- Some medical abstracts are long enough to trigger chunking.

---

## ArguAna

8,674 documents. 1,406 queries (all with qrels). Adversarial counter-argument retrieval — queries are counter-arguments to the target document, so relevant docs are semantically near-identical but opposite in stance.

### Indexing

| Statistic | Value |
|-----------|-------|
| Total time | 2.33 s |
| Throughput | 3,724 docs/s |
| P50 latency | 0.22 ms |
| P95 latency | 0.57 ms |
| P99 latency | 0.88 ms |

### Query (top-100)

| Statistic | Value |
|-----------|-------|
| Total time | 32.37 s |
| Throughput | 43 q/s |
| P50 latency | 21.01 ms |
| P95 latency | 39.58 ms |
| P99 latency | 52.42 ms |

### Retrieval Quality (1,406 judged queries)

| Metric | Value |
|--------|-------|
| Recall@1 | 0.00% |
| Recall@5 | 33.29% |
| Recall@10 | 50.43% |
| Recall@100 | 95.45% |
| Precision@1 | 0.00% |
| Precision@5 | 6.66% |
| Precision@10 | 5.04% |
| Precision@100 | 1.01% |
| MRR | 0.1584 |
| MAP | 0.1584 |
| nDCG@10 | 0.2264 |

### Notes

- ArguAna is deliberately adversarial: the correct document is topically identical but stance-opposed. Jaccard-based semantic similarity cannot distinguish "for" from "against," so R@1 is zero.
- The lexical lane carries most of the signal here (R@100 = 95.5%), but without stance-aware reranking the correct document rarely cracks the top 10.
- Query latency is higher than SciFact/NFCorpus because the corpus is larger (~8.7K docs) and queries are longer, more lexically complex arguments.

---

## LongMemEval-S

500 questions. Each has ~53 conversation sessions (~115K tokens).

> **Note:** The numbers below measure only the **retrieval component** — whether the correct session(s) appear in the top-k results. The official LongMemEval benchmark is a generation + LLM-as-judge task that scores answer correctness, not retrieval recall. These results show how well Adaline retrieves relevant context, but they are not directly comparable to published LongMemEval accuracy scores.

### Retrieval Quality

| Metric | Value |
|--------|-------|
| R@1 | 76.80% |
| R@5 | 93.60% |
| R@10 | 95.60% |

### Per-Category R@5

| Category | R@5 | Count |
|----------|-----|-------|
| knowledge-update | 100.00% | 78/78 |
| single-session-user | 97.14% | 68/70 |
| multi-session | 95.49% | 127/133 |
| temporal-reasoning | 92.48% | 123/133 |
| single-session-assistant | 96.43% | 54/56 |
| single-session-preference | 60.00% | 18/30 |

### Notes

- Correct session is in the top 5 for 468/500 questions.
- Knowledge updates are perfect (100% R@5).
- Single-session preference is the weak spot (60.0%). Preferences are often implicit and require inference beyond literal text matching.
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
- **Fusion**: RRF (k=10).
- **Reranking**: Term-coverage boost (weight=0.5) on top-K merged results.
- **Metrics**: Computed against BEIR qrels (binary relevance), averaged only over queries with judgments. nDCG uses standard `1/log2(rank+1)` gain.

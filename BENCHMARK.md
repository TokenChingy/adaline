# Adaline Benchmarks

## Build

```bash
nimble release
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

### Lane Ablation & Contribution (SciFact)

```bash
nim c -d:release -o:benchmarks/beir_ablation benchmarks/beir_ablation.nim
./benchmarks/beir_ablation
```

### CRUD (delete + update throughput)

```bash
nim c -d:release -o:benchmarks/crud benchmarks/crud.nim
./benchmarks/crud scifact 1000 1000
```

### Vision / Dense-Vector (Python, requires PyTorch)

```bash
python3 benchmarks/benchmark.py --dataset cifar10 --benchmark all
python3 benchmarks/benchmark.py --dataset imagenet --benchmark comparison --backbone resnet50
```

To also benchmark the actual compiled Adaline Engine (LSH) via Python bindings:

```bash
nimble python
python3 benchmarks/benchmark.py --dataset cifar10 --benchmark classify --engine
```

### LongMemEval

```bash
python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='xiaowu0162/longmemeval-cleaned', filename='longmemeval_s_cleaned.json', repo_type='dataset', local_dir='benchmarks/data')"
nim c -d:release -o:benchmarks/longmemeval benchmarks/longmemeval.nim
./benchmarks/longmemeval
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
| Total insert time | 1.11 s |
| Insert throughput | 4,669 docs/s |
| Insert P50 | 0.20 ms |
| Insert P95 | 0.30 ms |

### Query (top-100)

| Statistic | Value |
|-----------|-------|
| Query throughput | 273 q/s |
| Query P50 | 3.7 ms |
| Query P95 | 4.2 ms |

### Retrieval Quality (300 judged queries)

| Metric | Value |
|--------|-------|
| Recall@1 | 52.22% |
| Recall@5 | 72.11% |
| Recall@10 | 77.68% |
| Recall@100 | 87.72% |
| Precision@1 | 52.67% |
| Precision@5 | 15.40% |
| Precision@10 | 8.50% |
| Precision@100 | 0.99% |
| MRR | 0.6246 |
| MAP | 0.6140 |
| nDCG@10 | 0.6535 |

### CRUD Throughput

| Operation | Throughput | P50 |
|-----------|-----------|-----|
| Insert | 4,395 docs/s | 0.22 ms |
| Delete | 1,169 ops/s | 0.82 ms |
| Update | 1,203 ops/s | 0.82 ms |
| Post-CRUD query | 410 q/s | — |

### Lane Ablation

| Metric | Dual Lane | Semantic Only | Lexical Only |
|--------|-----------|---------------|--------------|
| Recall@1 | 52.22% | 27.00% | 51.22% |
| Recall@5 | 72.11% | 31.00% | 72.53% |
| Recall@10 | 77.68% | 32.00% | 78.18% |
| Recall@100 | 87.72% | 40.55% | 88.59% |
| MRR | 0.6246 | 0.2926 | 0.6234 |
| MAP | 0.6140 | 0.2899 | 0.6142 |
| nDCG@10 | 0.6535 | 0.2962 | 0.6540 |

The lexical lane is critical on SciFact: removing it cuts recall@100 in half and drops nDCG@10 by 0.36. Interestingly, lexical-only now slightly outperforms dual-lane on nDCG@10 (0.6540 vs 0.6535), suggesting the phrase-aware reranker is almost entirely lexical-driven on this dataset.

### Lane Contribution

Of the top-k merged results (before reranking), the origin breakdown is:

| Source | % of results |
|--------|-------------|
| Semantic only | 48.4% |
| Lexical only | 46.0% |
| Both lanes | 5.6% |

Each lane contributes roughly half of the final results independently.

---

## NFCorpus

3,633 documents. 3,237 queries (323 with qrels).

### Indexing

| Statistic | Value |
|-----------|-------|
| Total insert time | 0.78 s |
| Insert throughput | 4,646 docs/s |
| Insert P50 | 0.21 ms |
| Insert P95 | 0.31 ms |

### Query (top-100)

| Statistic | Value |
|-----------|-------|
| Query throughput | 460 q/s |
| Query P50 | 2.05 ms |
| Query P95 | 2.62 ms |

### Retrieval Quality (323 judged queries)

| Metric | Value |
|--------|-------|
| Recall@1 | 5.16% |
| Recall@5 | 11.52% |
| Recall@10 | 14.13% |
| Recall@100 | 22.22% |
| Precision@1 | 38.39% |
| Precision@5 | 26.56% |
| Precision@10 | 20.19% |
| Precision@100 | 5.16% |
| MRR | 0.4860 |
| MAP | 0.1232 |
| nDCG@10 | 0.2857 |

---

## ArguAna

8,674 documents. 1,406 queries (all with qrels). Adversarial counter-argument retrieval — queries are counter-arguments to the target document, so relevant docs are semantically near-identical but opposite in stance.

### Indexing

| Statistic | Value |
|-----------|-------|
| Total insert time | 1.66 s |
| Insert throughput | 5,216 docs/s |
| Insert P50 | 0.18 ms |
| Insert P95 | 0.29 ms |

### Query (top-100)

| Statistic | Value |
|-----------|-------|
| Query throughput | 95 q/s |
| Query P50 | 9.3 ms |
| Query P95 | 18.4 ms |

### Retrieval Quality (1,406 judged queries)

| Metric | Value |
|--------|-------|
| Recall@1 | 0.00% |
| Recall@5 | 50.00% |
| Recall@10 | 72.05% |
| Recall@100 | 97.30% |
| Precision@1 | 0.00% |
| Precision@5 | 10.00% |
| Precision@10 | 7.20% |
| Precision@100 | 0.98% |
| MRR | 0.2300 |
| MAP | 0.2300 |
| nDCG@10 | 0.3368 |

### Notes

- ArguAna is deliberately adversarial: the correct document is topically identical but stance-opposed. Jaccard-based semantic similarity cannot distinguish "for" from "against," so R@1 is zero.
- The lexical lane carries most of the signal here (R@100 = 97.3%). The phrase-aware reranker pushes the correct mirrored-argument document into the top 5 for 50.00% of queries (up from 38.12%) and into the top 10 for 72.05% of queries (up from 55.76%).
- Query latency is higher than SciFact/NFCorpus because the corpus is larger (~8.7K docs) and queries are longer, more lexically complex arguments.

---

## FIQA

57K financial QA pairs from the BEIR collection.

### Indexing

| Statistic | Value |
|-----------|-------|
| Total insert time | 10.0 s |
| Insert throughput | 5,677 docs/s |
| Insert P50 | 0.14 ms |
| Insert P95 | 0.32 ms |

### Query (top-100)

| Statistic | Value |
|-----------|-------|
| Query throughput | 14.6 q/s |
| Query P50 | 66 ms |
| Query P95 | 82 ms |

### Retrieval Quality

| Metric | Value |
|--------|-------|
| nDCG@10 | 0.1680 |

---

## Vision — Dense-Vector Retrieval (CIFAR-10 / MobileNetV2)

200 training vectors (1280-dim, 10 classes, 20 per class). 500 test vectors.

Features extracted from CIFAR-10 using pretrained MobileNetV2 (1280-dim,
L2-normalised). The `encodeDense` k-WTA encoder converts these float32
vectors into Adaline fingerprints.

Pure-Python benchmarks use brute-force Jaccard for comparison.  With the
`--engine` flag (requires compiled Python bindings: `nimble python`), the
benchmarks route search through the actual Nim LSH index.

```bash
python3 benchmarks/benchmark.py --dataset cifar10 --benchmark all
python3 benchmarks/benchmark.py --dataset cifar10 --benchmark classify --engine
```

Or run the Nim benchmark directly (uses `insertDense` / `searchDense` use-cases):

```bash
python3 benchmarks/dump_features.py
nim c -d:release -o:benchmarks/vision_bench benchmarks/vision_bench.nim
./benchmarks/vision_bench benchmarks/data/cifar10_features.bin
```

### Results

#### Footprint

| Encoder | Active bits | Sparse bytes | Ratio vs dense |
|---------|-------------|--------------|----------------|
| kwta_k128 | 127.3 | 255 | 20.1× |

#### 1-Shot Classification (50 queries/class)

| Method | Accuracy | ms/query |
|--------|----------|----------|
| dense (cosine) | **44.4%** | 0.018 |
| sparse (LSH) | 42.2% | 0.163 |

#### Few-Shot Scaling (accuracy vs prototypes/class)

| Shots | dense | sparse (LSH) |
|-------|-------|---------------|
| 1 | 34.6% | 35.8% |
| 2 | 50.4% | 42.0% |
| 5 | 53.2% | 52.6% |
| 10 | 54.6% | 55.8% |
| 20 | 62.6% | 61.8% |

#### Incremental Class Addition (5 prototypes/class)

| Classes | dense | sparse (LSH) | ms/query |
|---------|-------|---------------|----------|
| 2 | 88.0% | 89.0% | 0.149 |
| 4 | 78.5% | 68.5% | 0.161 |
| 6 | 57.3% | 68.3% | 0.182 |
| 8 | 54.5% | 58.5% | 0.202 |
| 10 | 52.8% | 50.2% | 0.217 |

#### Open-Set Detection (6 known classes, query all 10)

| Method | AUROC | Known acc |
|--------|-------|-----------|
| dense (cosine) | 0.552 | **67.7%** |
| sparse (LSH) | **0.578** | 59.3% |

### Key Findings

1. **Dense cosine and sparse LSH are competitive on real CNN features.**
   At 1-shot dense wins (44.4% vs 42.2%), but at 20 prototypes they converge
   (62.6% vs 61.8%). The k-WTA encoder preserves enough signal for Jaccard
   retrieval to track brute-force cosine similarity.
2. **Sparse storage is 20× smaller** than dense float32 (255 bytes active-bit
   list vs 5,120 bytes), while delivering comparable accuracy.
3. **Open-set detection is viable.** AUROC of 0.578 on real features shows
   the similarity score separates known from novel inputs better than random
   chance, though less cleanly than on synthetic structured data.
4. **Query latency scales gracefully.** At 10 classes × 5 prototypes = 50
   items, sparse retrieval averages ~0.16–0.22 ms/query through the full LSH engine.
   The `vision_bench` binary exercises the actual `insertDense` / `searchDense` use-cases.
5. **`--engine` bridges Python and Nim.** The same Python benchmark suite can
   now exercise the actual compiled LSH index, giving approximate search
   instead of brute-force Jaccard while keeping the PyTorch feature-extraction
   pipeline unchanged.

---

## LongMemEval-S

500 questions. Each has ~53 conversation sessions (~115K tokens).

> **Note:** The numbers below measure only the **retrieval component** — whether the correct session(s) appear in the top-k results. The official LongMemEval benchmark is a generation + LLM-as-judge task that scores answer correctness, not retrieval recall. These results show how well Adaline retrieves relevant context, but they are not directly comparable to published LongMemEval accuracy scores.

### Retrieval Quality

| Metric | Value |
|--------|-------|
| R@1 | 83.60% |
| R@5 | 94.80% |
| R@10 | 96.20% |

### Per-Category R@5

| Category | R@5 | Count |
|----------|-----|-------|
| knowledge-update | 98.72% | 77/78 |
| single-session-assistant | 100.00% | 56/56 |
| single-session-user | 95.71% | 67/70 |
| multi-session | 95.49% | 127/133 |
| temporal-reasoning | 93.98% | 125/133 |
| single-session-preference | 73.33% | 22/30 |

### Notes

- Correct session is in the top 5 for 474/500 questions.
- Knowledge updates are near-perfect (98.7% R@5).
- Single-session preference is the weak spot (73.3%). Preferences are often implicit and require inference beyond literal text matching.
- Conditional chunking is essential here. Sessions average ~115K tokens, which would saturate a single fingerprint.

---

## MS MARCO

Supported but not benchmarked due to scale (~8.8M passages, ~1 GB download, ~11 GB fingerprints).

```bash
./benchmarks/beir msmarco
```

At 8.8M docs, each query scans all LSH candidates with brute-force Jaccard. The bottleneck shifts to memory bandwidth (cold fingerprint access) and WAL replay time on startup.

---

## Observations

### Semantic vs Lexical Lane

On text datasets where queries and documents share heavy lexical overlap (SciFact, NFCorpus, ArguAna), the lexical lane (QLM + Dirichlet smoothing) carries the majority of the signal. The semantic lane (LSH + brute-force weighted Jaccard) contributes value for paraphrase and conceptual-neighbor retrieval, but its standalone recall lags behind the lexical lane on these benchmarks.

**SciFact ablation:**

| | Dual Lane | Semantic Only | Lexical Only |
|---|---|---|---|
| R@1 | 52.22% | 27.00% | 51.22% |
| R@100 | 87.72% | 40.55% | 88.59% |
| nDCG@10 | 0.6535 | 0.2962 | 0.6540 |

**Lane contribution** (top-k merged results, SciFact):

| Source | % |
|--------|---|
| Semantic only | 48.4% |
| Lexical only | 46.0% |
| Both lanes | 5.6% |

### LSH Configuration

Current defaults use 80 LSH bands with 2 rows each, providing full coverage of all 160 fingerprint segments. This strikes a balance between candidate recall and index size:

- **SciFact:** 4,669 docs/s insert, 273 q/s query, 87.72% R@100, nDCG@10 = 0.6535
- **NFCorpus:** 4,646 docs/s insert, 460 q/s query
- **ArguAna:** 5,216 docs/s insert, 95 q/s query
- **FIQA:** 5,677 docs/s insert, 14.6 q/s query

More bands increase candidate recall but grow the index; fewer bands reduce memory at the cost of recall. For corpora under ~10K docs, 80 bands is the practical sweet spot. At 57K docs (FIQA), brute-force Jaccard over LSH candidates becomes the bottleneck, dropping query throughput to ~15 q/s.

### Query Performance

The LSH + brute-force Jaccard semantic lane was optimized with:

- **Epoch-array dedup** in `queryLsh` (replaces `HashSet` allocation per query)
- **AND-construction** (min 2 band hits required) to filter weak collisions
- **Fast-path `bandHash`** (unrolled, no `mod` for default config)
- **In-place `removeLsh`** (no allocation churn on delete)
- **Pre-sized table** on `loadLsh`

These optimizations more than doubled query throughput on SciFact (125 → 273 q/s) with no meaningful recall loss.

### Reranker Impact

Replacing the uniform term-coverage boost with the phrase-aware lexical reranker yielded the following quality deltas:

| Dataset | Baseline nDCG@10 | New nDCG@10 | Δ |
|---------|-----------------|-------------|---|
| SciFact | 0.6044 | 0.6535 | **+8.1%** |
| ArguAna | 0.2520 | 0.3368 | **+33.7%** |
| NFCorpus | 0.2788 | 0.2857 | **+2.5%** |
| LongMemEval R@5 | 93.60% | 94.80% | **+1.2 pts** |

The gains are largest on ArguAna because exact phrase matching is the best lexical proxy for detecting mirrored counter-arguments. On SciFact, the gains come from matching multi-word scientific compounds (e.g., "treatment-resistant depression"). The reranker adds ~0.1 ms on SciFact but ~4.5 ms on ArguAna because ArguAna queries are longer and more lexically complex; limiting the reranker to the top-30 candidates keeps SciFact latency within ~3% of baseline.

### Chunking

LongMemEval demonstrates the importance of conditional chunking. Sessions average ~115K tokens — far beyond what a single 10,240-bit fingerprint can encode without saturation. The sentence-aware splitter with overlap ensures no semantic information is lost at chunk boundaries.

### Dense-Vector Encoding

The `encodeDense` k-WTA encoder converts arbitrary L2-normalised float32 vectors into Adaline fingerprints using the same `hashFeature`/`probeBlock` primitives as the text SDR encoder. On synthetic vision data, sparse LSH retrieval outperforms dense cosine similarity at 1-shot classification (54.4% vs 32.2%) and scales efficiently with more prototypes.

---

## Methodology

- **Distance:** `1.0 - weightedJaccard` with block weights 50% (tokens), 25% (bigrams), 25% (context).
- **Search:** LSH seeds → brute-force weighted Jaccard scoring.
- **Lexical:** Query Likelihood Model with Dirichlet smoothing (μ=2000).
- **Fusion:** RRF-S (score-aware RRF, k=10). Multiplies each reciprocal-rank contribution by the normalised raw lane score, preserving RRF robustness while using score magnitude as a tie-breaker.
- **Reranking:** Phrase-aware lexical reranker on the top-30 candidates. Blends IDF-weighted unigram coverage, exact substring phrase matching, and soft document-length normalization (weights 0.30 / 0.20 / 0.05).
- **Metrics:** Computed against BEIR qrels (binary relevance), averaged only over queries with judgments. nDCG uses standard `1/log2(rank+1)` gain.

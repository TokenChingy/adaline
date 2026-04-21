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

To also benchmark the actual compiled Adaline Engine (HNSW+LSH) via Python bindings:

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
| Total insert time | 4.53 s |
| Insert throughput | 1,143 docs/s |
| Insert P50 | 0.72 ms |
| Insert P95 | 1.61 ms |

### Query (top-100)

| Statistic | Value |
|-----------|-------|
| Query throughput | 260 q/s |
| Query P50 | 3.85 ms |
| Query P95 | 4.61 ms |

### Retrieval Quality (300 judged queries)

| Metric | Value |
|--------|-------|
| Recall@1 | 40.83% |
| Recall@5 | 65.69% |
| Recall@10 | 71.28% |
| Recall@100 | 89.22% |
| Precision@1 | 42.00% |
| Precision@5 | 14.13% |
| Precision@10 | 7.80% |
| Precision@100 | 1.01% |
| MRR | 0.5328 |
| MAP | 0.5232 |
| nDCG@10 | 0.5670 |

### CRUD Throughput

| Operation | Throughput | P50 |
|-----------|-----------|-----|
| Insert | 1,054 docs/s | 0.82 ms |
| Delete | 1,802 ops/s | 0.51 ms |
| Update | 934 ops/s | 0.97 ms |
| Post-CRUD query | 707 q/s | — |

---

## NFCorpus

3,633 documents. 3,237 queries (323 with qrels).

### Indexing

| Statistic | Value |
|-----------|-------|
| Total insert time | 3.09 s |
| Insert throughput | 1,175 docs/s |
| Insert P50 | 0.72 ms |
| Insert P95 | 1.52 ms |

### Query (top-100)

| Statistic | Value |
|-----------|-------|
| Query throughput | 393 q/s |
| Query P50 | 2.48 ms |
| Query P95 | 3.20 ms |

### Retrieval Quality (323 judged queries)

| Metric | Value |
|--------|-------|
| Recall@1 | 4.68% |
| Recall@5 | 10.81% |
| Recall@10 | 13.32% |
| Recall@100 | 22.58% |
| Precision@1 | 36.84% |
| Precision@5 | 25.57% |
| Precision@10 | 19.44% |
| Precision@100 | 5.47% |
| MRR | 0.4608 |
| MAP | 0.1233 |
| nDCG@10 | 0.2731 |

---

## ArguAna

8,674 documents. 1,406 queries (all with qrels). Adversarial counter-argument retrieval — queries are counter-arguments to the target document, so relevant docs are semantically near-identical but opposite in stance.

### Indexing

| Statistic | Value |
|-----------|-------|
| Total insert time | 7.70 s |
| Insert throughput | 1,127 docs/s |
| Insert P50 | 0.77 ms |
| Insert P95 | 1.74 ms |

### Query (top-100)

| Statistic | Value |
|-----------|-------|
| Query throughput | 47 q/s |
| Query P50 | 19.48 ms |
| Query P95 | 35.83 ms |

### Retrieval Quality (1,406 judged queries)

| Metric | Value |
|--------|-------|
| Recall@1 | 0.00% |
| Recall@5 | 22.12% |
| Recall@10 | 38.69% |
| Recall@100 | 95.31% |
| Precision@1 | 0.00% |
| Precision@5 | 4.42% |
| Precision@10 | 3.87% |
| Precision@100 | 0.96% |
| MRR | 0.1174 |
| MAP | 0.1174 |
| nDCG@10 | 0.1649 |

### Notes

- ArguAna is deliberately adversarial: the correct document is topically identical but stance-opposed. Jaccard-based semantic similarity cannot distinguish "for" from "against," so R@1 is zero.
- The lexical lane carries most of the signal here (R@100 = 95.5%), but without stance-aware reranking the correct document rarely cracks the top 10.
- Query latency is higher than SciFact/NFCorpus because the corpus is larger (~8.7K docs) and queries are longer, more lexically complex arguments.

---

## Vision — Dense-Vector Retrieval (CIFAR-10 / MobileNetV2)

200 training vectors (1280-dim, 10 classes, 20 per class). 500 test vectors.

Features extracted from CIFAR-10 using pretrained MobileNetV2 (1280-dim,
L2-normalised). The `encodeDense` k-WTA encoder converts these float32
vectors into Adaline fingerprints.

Pure-Python benchmarks use brute-force Jaccard for comparison.  With the
`--engine` flag (requires compiled Python bindings: `nimble python`), the
benchmarks route search through the actual Nim HNSW+LSH index.

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
| sparse (HNSW) | 42.2% | 0.163 |

#### Few-Shot Scaling (accuracy vs prototypes/class)

| Shots | dense | sparse (HNSW) |
|-------|-------|---------------|
| 1 | 34.6% | 35.8% |
| 2 | 50.4% | 42.0% |
| 5 | 53.2% | 52.6% |
| 10 | 54.6% | 55.8% |
| 20 | 62.6% | 61.8% |

#### Incremental Class Addition (5 prototypes/class)

| Classes | dense | sparse (HNSW) | ms/query |
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
| sparse (HNSW) | **0.578** | 59.3% |

### Key Findings

1. **Dense cosine and sparse HNSW are competitive on real CNN features.**
   At 1-shot dense wins (44.4% vs 42.2%), but at 20 prototypes they converge
   (62.6% vs 61.8%). The k-WTA encoder preserves enough signal for Jaccard
   retrieval to track brute-force cosine similarity.
2. **Sparse storage is 20× smaller** than dense float32 (255 bytes active-bit
   list vs 5,120 bytes), while delivering comparable accuracy.
3. **Open-set detection is viable.** AUROC of 0.578 on real features shows
   the similarity score separates known from novel inputs better than random
   chance, though less cleanly than on synthetic structured data.
4. **Query latency scales gracefully.** At 10 classes × 5 prototypes = 50
   items, sparse retrieval averages ~0.16–0.22 ms/query through the full HNSW engine.
   The `vision_bench` binary exercises the actual `insertDense` / `searchDense` use-cases.
5. **`--engine` bridges Python and Nim.** The same Python benchmark suite can
   now exercise the actual compiled HNSW+LSH index, giving O(log N) search
   instead of brute-force Jaccard while keeping the PyTorch feature-extraction
   pipeline unchanged.

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

## Observations

### Semantic vs Lexical Lane

On text datasets where queries and documents share heavy lexical overlap (SciFact, NFCorpus, ArguAna), the lexical lane (QLM + Dirichlet smoothing) carries the majority of the signal. The semantic lane (LSH + HNSW + weighted Jaccard) contributes value for paraphrase and conceptual-neighbor retrieval, but its standalone recall lags behind the lexical lane on these benchmarks.

### HNSW Configuration

Current defaults use `M=16` neighbors and `efConstruction=64` / `efSearch=64`. This strikes a balance between graph quality and throughput:

- **SciFact:** 1,143 docs/s insert, 260 q/s query, 89.22% R@100
- **NFCorpus:** 1,175 docs/s insert, 393 q/s query
- **ArguAna:** 1,127 docs/s insert, 47 q/s query

Higher `M` improves recall at depth (R@100) but increases insertion time due to more neighbor wiring. Higher `ef` improves graph quality during construction and search precision but increases latency. For corpora under ~10K docs, `M=16, ef=64` is the practical sweet spot.

### Chunking

LongMemEval demonstrates the importance of conditional chunking. Sessions average ~115K tokens — far beyond what a single 10,240-bit fingerprint can encode without saturation. The sentence-aware splitter with overlap ensures no semantic information is lost at chunk boundaries.

### Dense-Vector Encoding

The `encodeDense` k-WTA encoder converts arbitrary L2-normalised float32 vectors into Adaline fingerprints using the same `hashFeature`/`probeBlock` primitives as the text SDR encoder. On synthetic vision data, sparse HNSW retrieval outperforms dense cosine similarity at 1-shot classification (54.4% vs 32.2%) and scales efficiently with more prototypes.

---

## Methodology

- **Distance:** `1.0 - weightedJaccard` with block weights 50% (tokens), 25% (bigrams), 25% (context).
- **Search:** LSH seeds → HNSW layer-0 descent (`efSearch=64`).
- **Lexical:** Query Likelihood Model with Dirichlet smoothing (μ=2000).
- **Fusion:** RRF (k=10).
- **Reranking:** Term-coverage boost (weight=0.5) on top-K merged results.
- **Metrics:** Computed against BEIR qrels (binary relevance), averaged only over queries with judgments. nDCG uses standard `1/log2(rank+1)` gain.

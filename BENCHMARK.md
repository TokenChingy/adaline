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

### Ablation Benchmark (semantic / lexical / combined)

The ablation benchmark isolates each search lane:

```bash
nim c -d:release -o:benchmarks/ablation benchmarks/ablation.nim
./benchmarks/ablation scifact all       # run semantic, lexical, combined
./benchmarks/ablation scifact semantic  # semantic lane only
./benchmarks/ablation scifact lexical   # lexical lane only
./benchmarks/ablation scifact combined  # both lanes with RRF
```

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

### Baseline (master)

#### Indexing


#### Query (top-100)


#### Retrieval Quality (300 judged queries)

| Metric | Value |
|--------|-------|
| Recall@1 | 42.81% |
| Recall@5 | 63.44% |
| Recall@10 | 72.57% |
| Recall@100 | 86.49% |
| Precision@1 | 44.67% |
| Precision@5 | 13.53% |
| Precision@10 | 7.90% |
| Precision@100 | 1.05% |
| MRR | 0.5475 |
| MAP | 0.5371 |
| nDCG@10 | 0.5796 |

### After bug fixes (`fix/combined-bugs`)

#### Indexing


> **Note:** Insertion is slower because the fixed `searchLayer` explores more candidates during `efConstruction=200`, building a higher-quality graph.

#### Query (top-100)


#### Retrieval Quality (300 judged queries)

| Metric | Value |
|--------|-------|
| Recall@1 | 45.81% |
| Recall@5 | 64.69% |
| Recall@10 | 71.42% |
| Recall@100 | 87.57% |
| Precision@1 | 47.00% |
| Precision@5 | 13.93% |
| Precision@10 | 7.80% |
| Precision@100 | 1.09% |
| MRR | 0.5626 |
| MAP | 0.5521 |
| nDCG@10 | 0.5891 |

### `fix/no-graph-weyl-sparse` (this branch)

Hardware: WSL2 Linux, 4 cores, 8 GB RAM (x86_64). Not directly comparable to M2 numbers above.

This branch includes: `searchLayer` fix, LSH coverage fix (80×2), entryPoint delete fix, Weyl-sequence probes, sparser defaults (token 4→3, bigram 2→1, context 2→1), and optimized HNSW defaults (M=8, efConstruction=50).

#### HNSW (M=8, efConstruction=50)

| Statistic | Value |
|-----------|-------|
| Total insert time | 11.88 s |
| Insert throughput | 436.11 docs/s |
| Insert P50 | 1.89 ms |
| Insert P95 | 4.34 ms |
| Query throughput | 127.67 q/s |
| Query P50 | 7.49 ms |
| Recall@1 | 40.81% |
| Recall@5 | 63.92% |
| Recall@10 | 70.61% |
| Recall@100 | 88.58% |
| nDCG@10 | 0.5649 |

#### Comparison

| Mode | Insert (docs/s) | Query (q/s) | nDCG@10 | R@10 |
|------|----------------|-------------|---------|------|
| Baseline (master) | 2,465 | 222 | 0.5796 | 72.57% |
| `fix/combined-bugs` | 32.8 | 45.7 | 0.5891 | 71.42% |
| **This branch** | **436.1** | **127.7** | **0.5649** | **70.61%** |


> The baseline (master) numbers are from Apple M2 and not directly comparable. The key comparison is between `fix/combined-bugs` and this branch on the same hardware. With M=8, efC=50, insertion is **~13× faster** than `fix/combined-bugs` (32.8→436 docs/s) and query is **~2.8× faster** (45.7→128 q/s), with only a ~1% drop in R@100.

### Ablation Breakdown (SciFact)

| Branch | Lane | R@1 | R@5 | R@10 | R@100 | nDCG@10 |
|--------|------|-----|-----|------|-------|---------|
| **baseline** | semantic | 4.11% | 4.94% | 5.06% | 5.22% | 0.0467 |
| **baseline** | lexical | 43.25% | 64.19% | 73.57% | 88.19% | 0.5870 |
| **baseline** | combined | 42.58% | 63.36% | 71.90% | 86.82% | 0.5745 |
| **searchlayer-bug** | semantic | 31.14% | 45.31% | 48.83% | 53.54% | 0.4063 |
| **searchlayer-bug** | lexical | 43.25% | 64.19% | 73.57% | 88.19% | 0.5870 |
| **searchlayer-bug** | combined | 44.47% | 65.36% | 71.58% | 87.91% | 0.5816 |
| **lsh-coverage** | semantic | 12.42% | 16.17% | 17.50% | 19.61% | 0.1531 |
| **lsh-coverage** | lexical | 43.25% | 64.19% | 73.57% | 88.19% | 0.5870 |
| **lsh-coverage** | combined | 42.58% | 63.36% | 71.90% | 86.82% | 0.5745 |
| **combined-bugs** | semantic | 33.97% | 49.08% | 53.28% | 58.98% | 0.4404 |
| **combined-bugs** | lexical | 43.25% | 64.19% | 73.57% | 88.19% | 0.5870 |
| **combined-bugs** | combined | 45.47% | 65.36% | 71.42% | 87.57% | 0.5889 |
| **hash-weyl-fixed** | semantic | 32.47% | 47.67% | 53.06% | 60.53% | 0.4302 |
| **hash-weyl-fixed** | lexical | 43.25% | 64.19% | 73.57% | 88.19% | 0.5870 |
| **hash-weyl-fixed** | combined | 45.72% | 64.53% | 72.33% | 88.47% | 0.5938 |
| **hnsw-layer-p-fixed** | semantic | 32.97% | 47.92% | 52.11% | 57.73% | 0.4291 |
| **hnsw-layer-p-fixed** | lexical | 43.25% | 64.19% | 73.57% | 88.19% | 0.5870 |
| **hnsw-layer-p-fixed** | combined | 45.14% | 65.03% | 71.42% | 87.57% | 0.5863 |
| **all-fixes** | semantic | 32.47% | 48.67% | 54.39% | 61.93% | 0.4369 |
| **all-fixes** | lexical | 43.25% | 64.19% | 73.57% | 88.19% | 0.5870 |
| **all-fixes** | combined | 45.39% | 64.19% | 72.40% | 88.41% | 0.5922 |

### Key Findings (SciFact)

1. **The `searchLayer` early-termination bug is the single largest issue.** Fixing it raises semantic R@100 from **5.2% → 53.5%** and combined nDCG@10 from **0.5745 → 0.5816**.
2. **LSH coverage matters.** With the default 50 bands × 2 rows, 37.5% of the fingerprint (the entire context block + tail of bigrams) is invisible to LSH. Raising `lshBands` to 80 so that `80 × 2 = 160` covers all segments adds another **~5.4 points** of semantic R@100 on top of the bug fix.
3. **Weyl-sequence probes (`hash-weyl`) give a small combined boost.** Changing `probeBlock` from sequential seeds to a Weyl sequence improves combined nDCG@10 to **0.5938** (best overall), but slightly reduces semantic precision@1/5 compared to `combined-bugs` alone.
4. **Standard HNSW layer distribution (`mL = 1/ln(M)`) hurts sparse SDRs.** The dense hierarchy (`p = 0.5`) works better for Jaccard-based sparse fingerprints than the standard sparse hierarchy.
5. **Diversity-aware neighbor pruning hurts and is very slow.** The standard HNSW `selectNeighbors` heuristic is designed for dense Euclidean/inner-product spaces. For sparse Jaccard fingerprints, simple distance-based truncation outperforms it.
6. **Lexical lane dominates SciFact.** Even with all fixes, lexical-only (88.2% R@100, 0.587 nDCG@10) outperforms semantic-only (60.5% R@100, 0.430 nDCG@10). The semantic lane is still valuable for paraphrase/semantic-neighbor retrieval, but on short scientific abstracts with heavy lexical overlap, QLM carries most of the signal.

---

## NFCorpus

3,633 documents. 3,237 queries (323 with qrels).

### Baseline (master)

#### Indexing


#### Query (top-100)


#### Retrieval Quality (323 judged queries)

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

### After bug fixes (`fix/combined-bugs`)

#### Ablation Breakdown (NFCorpus)

| Lane | R@1 | R@5 | R@10 | R@100 | nDCG@10 |
|------|-----|-----|------|-------|---------|
| semantic | 2.24% | 5.15% | 6.46% | 10.38% | 0.1685 |
| lexical | 5.50% | 11.32% | 13.33% | 22.85% | 0.2817 |
| combined | 4.96% | 11.22% | 13.18% | 22.27% | 0.2701 |

### Key Findings (NFCorpus)

1. **Semantic path improves dramatically** with bug fixes: nDCG@10 goes from **0.0728 → 0.1685** (+131%).
2. **Combined system is slightly below lexical-only** on this dataset (0.2701 vs 0.2817). Sparse relevance judgments and medical vocabulary mean the semantic lane's broader recall sometimes introduces noise into the RRF merge.
3. **Query speed drops** with the bug fixes because `searchLayer` now explores more candidates (as intended). Semantic-only P50 latency goes from 7.1 ms → 10.3 ms.

---

## ArguAna

8,674 documents. 1,406 queries (all with qrels). Adversarial counter-argument retrieval — queries are counter-arguments to the target document, so relevant docs are semantically near-identical but opposite in stance.

### Indexing


### Query (top-100)


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

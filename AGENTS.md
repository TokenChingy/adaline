# Adaline — Agent Notes

## What this project is

Adaline is a Nim library for generating and querying **Sparse Distributed Representations** via **Sparse Fingerprints**. Each fingerprint is a fixed-size **10240-bit bitmap**. These fingerprints are stored and searched inside a **Hierarchical Navigable Small World (HNSW) Graph** with a **Fingerprint LSH** (GoldFinger-style) seed layer. A **Lexical Sidecar** (Query Likelihood with Dirichlet Smoothing) runs in parallel, and results are merged via **Reciprocal Rank Fusion**.

Long memories are automatically **chunked** into multiple fingerprints when any block approaches saturation. Chunk-to-parent linkage is persisted in `chunks.bin`.

The entire index is laid out as contiguous byte arrays on disk with 256-byte self-describing headers (`ADLN` magic). Store files use dense slot addressing (0, 1, 2…) rather than byte offsets, with pre-allocated 64 MiB growth chunks to minimize mmap remaps. A freelist enables slot reuse on delete. Persisted indexes (`lsh.bin`, `lexical.bin`, `corpus.bin`) allow fast startup by skipping full WAL replay. The CLI casts pointers via `mmap` and immediately begins execution.

## Folder layout

```
use_cases/          <- One file per use-case. Each file declares its own
                       input/output ports (contracts). No business logic here;
                       only wiring and orchestration.
                       (insert, search, update, delete)
domain/entities/    <- Core types: Fingerprint, HNSW node, Memory, Config, Chunk.
domain/algorithms/  <- The math:
                       - SDR encoder (partitioned Tokens / Bigrams / XOR Context)
                       - Dense encoder (k-WTA float32 vector → fingerprint)
                       - Regional weighted Jaccard
                       - Banded Fingerprint LSH (GoldFinger-style)
                       - HNSW construction and greedy search
                       - Lexical index (QLM + Dirichlet smoothing)
                       - Corpus index (IDF tracking)
                       - RRF merger
                       - Reranker (term-coverage boost)
                       - Chunker (conditional sentence-aware splitting)
domain/services/    <- Pure domain orchestration. Each operation lives in
                       `memory/` (types, init, insert, delete, update, search,
                       checkpoint, insert_dense, search_dense, delete_dense).
                       Import the specific file; no umbrella re-export.
 infrastructure/     <- Concrete adapters: mmapped storage (WAL, fingerprint store,
                       graph store, chunks mapping store). Imported by domain
                       services when needed.
bindings/           <- Python bindings (nimpy) exposing the same use cases.
benchmarks/         <- BEIR benchmark runner, LongMemEval runner, CRUD benchmark,
                       Python vision suite, `vision_bench.nim` (Nim dense-vector
                       benchmark using insertDense / searchDense use-cases).
tests/              <- Unit tests mirroring the folder layout.
adaline.nim         <- The CLI entry point.
```

## Design constraints

This is **not** Clean Architecture. The dependency flow is:

```
Use Cases ← Domain ← Infrastructure
```

- **Use cases** only orchestrate. They do not contain domain rules.
- **Domain** code (especially `MemoryService`) calls into `infrastructure/` (e.g., `MmappedStorage`).
- **Infrastructure** sits at the bottom of the stack and is imported by domain services when needed.

## Stack

- **Language:** Nim
- **Fingerprint size:** 10240 bits (1280 bytes)
- **Storage:** Memory-mapped flat files with self-describing headers (WAL, fingerprint store, graph store, chunks mapping store, persisted LSH/lexical/corpus indexes)
- **Index / search structure:** Banded Fingerprint LSH + HNSW Graph
- **Lexical lane:** Query Likelihood Model with Dirichlet Smoothing
- **Merger:** Reciprocal Rank Fusion (RRF)
- **Chunking:** Sentence-aware conditional splitting with overlap; threshold configurable via `chunkSaturationThreshold`
- **Delete / Update:** `deleteMemory()` and `updateMemory()` use an in-memory reverse edge index to heal HNSW neighbor lists without tombstones or full rebuilds
- **Checkpoint:** `checkpoint()` serializes in-memory indexes to disk for fast restart
- **Python bindings:** `bindings/adaline.nim` exposes `Engine` (insert, search, update, delete, stats, checkpoint) via nimpy. Build with `nimble python`.
- **Dense-vector encoding:** `domain/algorithms/dense_encoder.nim` provides `encodeDense()` for converting L2-normalised float32 vectors into fingerprints via k-WTA, enabling vision / signal / tabular use cases.

## Semantic Path Diagnostics

The following issues were discovered and fixed during a systematic ablation study (see `BENCHMARK.md` for full numbers). The branches below exist in this repo for reference.

### Critical: `searchLayer` early-termination bug

**File:** `domain/algorithms/hnsw_graph.nim`  
**Fix:** `results.len > 0` → `results.len >= ef`

The original code broke the best-first search **even when the result buffer was not full** (`results.len < ef`). This caused premature stopping during both insertion (`efConstruction=200`) and query (`efSearch=64`). The bug destroyed graph navigability: with `ef=1` during greedy upper-layer descent, the loop stopped after exploring a single neighbor.

**Impact on SciFact semantic path:** R@100 rose from **5.2% → 53.5%** after this fix alone.

### Critical: LSH ignored 37.5% of the fingerprint

**File:** `domain/entities/config.nim`  
**Fix:** `lshBands: 50` → `lshBands: 80`

With the default config (`lshBands=50`, `lshRows=2`), only segments 0..99 of the 160 uint64 segments were hashed by `bandHash`. The entire XOR-context block (segments 112..159) and the tail of the bigram block (segments 100..111) were invisible to LSH candidate generation.

**Impact:** Raising `lshBands` to 80 covers all 160 segments and adds **~5 points** of semantic R@100 on top of the bug fix.

### Safety: dangling entry point after delete

**File:** `domain/services/memory/delete.nim`  
**Fix:** After deleting chunks, scan remaining nodes and update `hnswEntryPoint` / `maxHnswLayer` if the entry point was removed.

Without this fix, a search after deleting the entry-point node dereferences a freed/reused slot.

### Tested but NOT recommended

| Experiment | Branch | Result |
|------------|--------|--------|
| **Weyl-sequence probes** | `exp/hash-weyl-fixed` | Marginal combined nDCG@10 gain, but changes fingerprint generation (breaks existing indexes). **Adopted on `fix/no-graph-weyl-sparse`.** |
| **Standard HNSW layer distribution (`mL = 1/ln(M)`)** | `exp/hnsw-layer-p-fixed` | **Hurts** semantic recall. The dense hierarchy (`p=0.5`) works better for sparse Jaccard fingerprints than the standard sparse hierarchy. |
| **Diversity-aware neighbor pruning** | `exp/diversity-heuristic` | **Hurts** semantic recall and is **~10× slower** at insertion. The standard HNSW `selectNeighbors` heuristic is designed for dense Euclidean spaces; simple distance truncation is superior for sparse SDRs. |

### Branch: `fix/no-graph-weyl-sparse`

This branch includes the three critical fixes (searchLayer, LSH coverage, entryPoint delete), replaces correlated `probeBlock` hashing with a Weyl sequence, and increases fingerprint sparsity.

**Changes:**
- **Config:** `hnswMaxNeighbors: 16`, `hnswEfConstruction: 64`, `hnswEfSearch: 64`.
- **Search (`domain/services/memory/search.nim`):** LSH seeds + HNSW approximate search.
- **Insert (`domain/services/memory/insert.nim`):** WAL → corpus → chunk → fingerprint → LSH → lexical → HNSW.
- **Delete (`domain/services/memory/delete.nim`):** Heals HNSW edges via reverse index; zeros fingerprint so deleted slots are skipped.
- **SDR encoder (`domain/algorithms/sdr_encoder.nim`):** `probeBlock` now uses `h0 + i * 0x9e3779b97f4a7c15` (Weyl sequence) instead of `hashFeature(feature, uint64(i))`, guaranteeing independent probes.
- **Sparsity defaults:** `tokenProbes: 4→3`, `bigramProbes: 2→1`, `contextProbes: 2→1`.
- **LSH defaults:** `lshBands: 50→80`, `lshRows: 3→2` (full coverage of all 160 segments).
- **Storage (`infrastructure/mmapped_storage.nim`):** `freeId` zeros the fingerprint so brute-force search skips deleted slots.
- **Dense encoder (`domain/algorithms/dense_encoder.nim`):** New k-WTA encoder for float32 vectors (vision / signal / tabular).
- **Python vision suite (`benchmarks/benchmark.py`, `dense_encoder.py`, `feature_extractor.py`, `dump_features.py`):** CIFAR-10 / ImageNet feature extraction and benchmarking.  `benchmark.py` can optionally exercise the actual compiled Adaline Engine (`--engine`) via Python bindings for HNSW+LSH search.
- **Dense-vector pipeline (`domain/services/memory/insert_dense.nim`, `search_dense.nim`, `delete_dense.nim`):** Bypasses text chunking and lexical indexing to encode float32 vectors directly into fingerprints for vision / signal / tabular use cases.

**Trade-off:** Compared to baseline (master), query throughput is **~17% higher** (222→260 q/s) with better R@100 (86.49%→89.22%). The denser graph (M=16) trades ~2× insertion speed for stronger recall at depth, while ef=64 keeps query latency competitive.

For small corpora where insertion speed and semantic recall matter more than query throughput, the HNSW graph can be disabled by setting `hnswMaxLayers: 0`.

## Agent hygiene

Before finishing any task:

1. **Remove dead code and files.** Delete design docs, scratch scripts, or temporary binaries created during exploration. Do not leave stale documentation that no longer matches the implementation.
2. **Clean generated artifacts.** Remove benchmark data directories, test temp dirs, and compiled binaries that can be regenerated. (These are already in `.gitignore`, but purge them from the working tree too.)
3. **Update living docs.** `AGENTS.md` and `BENCHMARK.md` must reflect the current codebase only. If you added a feature, benchmark, or test, add it to the relevant doc. If you removed something, remove it from the doc.
4. **Run the full test suite.** `nimble test` must pass before you consider a task complete.

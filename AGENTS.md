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
                       - Regional weighted Jaccard
                       - Banded Fingerprint LSH (GoldFinger-style)
                       - HNSW construction and greedy search
                       - Lexical index (QLM + Dirichlet smoothing)
                       - Corpus index (IDF tracking)
                       - RRF merger
                       - Reranker (term-coverage boost)
                       - Chunker (conditional sentence-aware splitting)
domain/services/    <- Pure domain orchestration. `memory_service.nim` is the umbrella
                       re-export; each operation lives in `memory/` (types, init,
                       insert, delete, update, search, checkpoint). Import the
                       specific file when working on a single use case.
 infrastructure/     <- Concrete adapters: mmapped storage (WAL, fingerprint store,
                       graph store, chunks mapping store). Imported by domain
                       services when needed.
bindings/           <- Python bindings (nimpy) exposing the same use cases.
benchmarks/         <- BEIR benchmark runner, LongMemEval runner, CRUD benchmark.
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

## Agent hygiene

Before finishing any task:

1. **Remove dead code and files.** Delete design docs, scratch scripts, or temporary binaries created during exploration. Do not leave stale documentation that no longer matches the implementation.
2. **Clean generated artifacts.** Remove benchmark data directories, test temp dirs, and compiled binaries that can be regenerated. (These are already in `.gitignore`, but purge them from the working tree too.)
3. **Update living docs.** `AGENTS.md` and `BENCHMARK.md` must reflect the current codebase only. If you added a feature, benchmark, or test, add it to the relevant doc. If you removed something, remove it from the doc.
4. **Run the full test suite.** `nimble test` must pass before you consider a task complete.

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
domain/services/    <- Pure domain orchestration (MemoryService).
 infrastructure/     <- Concrete adapters: mmapped storage (WAL, fingerprint store,
                       graph store, chunks mapping store). Imported by domain
                       services when needed.
adaline.nim         <- The CLI entry point.
```

## Design constraints

This is **not** Clean Architecture. The dependency flow is:

```
Use Cases ← Domain ← Infrastructure
```

- **Use cases** only orchestrate. They do not contain domain rules.
- **Domain** code (especially `MemoryService`) calls into `infrastructure/` (e.g. `MmappedStorage`).
- **Infrastructure** sits at the bottom of the stack and is imported by domain services when needed.

## Stack

- **Language:** Nim
- **Fingerprint size:** 10240 bits (1280 bytes)
- **Storage:** Memory-mapped flat files with self-describing headers (WAL, fingerprint store, graph store, chunks mapping store, persisted LSH/lexical/corpus indexes)
- **Index / search structure:** Banded Fingerprint LSH + HNSW Graph
- **Lexical lane:** Query Likelihood Model with Dirichlet Smoothing
- **Merger:** Reciprocal Rank Fusion (RRF)
- **Chunking:** Sentence-aware conditional splitting with overlap; threshold configurable via `chunkSaturationThreshold`
- **Delete:** `deleteMemory()` removes from all indexes and returns slots to freelist
- **Checkpoint:** `checkpoint()` serializes in-memory indexes to disk for fast restart

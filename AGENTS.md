# Adaline — Agent Notes

## What this project is

Adaline is a Nim library for generating and querying **Sparse Distributed Representations** via **Sparse Fingerprints**. Each fingerprint is a fixed-size **10240-bit bitmap**. These fingerprints are stored and searched inside a **Hierarchical Navigable Small World (HNSW) Graph** with a **MinHash LSH** seed layer. A **Lexical Sidecar** (Query Likelihood with Dirichlet Smoothing) runs in parallel, and results are merged via **Reciprocal Rank Fusion**.

The entire index is laid out as contiguous byte arrays on disk. The CLI casts pointers via `mmap` and immediately begins execution.

## Folder layout

```
use_cases/          <- One file per use-case. Each file declares its own
                       input/output ports (contracts). No business logic here;
                       only wiring and orchestration.
domain/entities/    <- Core types: Fingerprint, HNSW node, Memory, Config.
domain/algorithms/  <- The math:
                       - SDR encoder (partitioned Tokens / Bigrams / XOR Context)
                       - Regional weighted Jaccard
                       - Banded MinHash LSH
                       - HNSW construction and greedy search
                       - Lexical index (QLM + Dirichlet smoothing)
                       - RRF merger
domain/services/    <- Pure domain orchestration (MemoryService).
infrastructure/     <- Concrete adapters: mmapped storage (WAL, fingerprint store,
                       graph store). Domain code has no knowledge of this layer.
cli.nim             <- The CLI entry point.
```

## Design constraints

- **Use cases** only orchestrate. They do not contain domain rules.
- **Domain** code has no knowledge of `infrastructure/` or `cli.nim`.
- **Infrastructure** depends on domain interfaces, never the other way around.

## Stack

- **Language:** Nim
- **Fingerprint size:** 10240 bits (1280 bytes)
- **Storage:** Memory-mapped flat files (WAL, fingerprint store, graph store)
- **Index / search structure:** Banded MinHash LSH + HNSW Graph
- **Lexical lane:** Query Likelihood Model with Dirichlet Smoothing
- **Merger:** Reciprocal Rank Fusion (RRF)

# Adaline — Agent Notes

## What this project is

Adaline is a Nim library for generating and querying **Sparse Distributed Representations** via **Sparse Fingerprints**. Each fingerprint is a fixed-size **10240-bit bitmap**. These fingerprints are stored and searched via a **Fingerprint LSH** (GoldFinger-style) seed layer with brute-force Jaccard scoring. A **Lexical Sidecar** (Query Likelihood with Dirichlet Smoothing) runs in parallel, and results are merged via **Reciprocal Rank Fusion**.

Long memories are automatically **chunked** into multiple fingerprints when any block approaches saturation. Chunk-to-parent linkage is persisted in `chunks.bin`.

The entire index is laid out as contiguous byte arrays on disk with 256-byte self-describing headers (`ADLN` magic). `fingerprints.bin` stores append-only variable-length compressed fingerprints addressed by byte offset; `fingerprints.idx` provides a dense slot-addressed table (0, 1, 2…) mapping each ID to its `(offset, size)` pair. A freelist enables slot reuse on delete. Files grow in pre-allocated 64 MiB chunks to minimize mmap remaps. Persisted indexes (`lsh.bin`, `lexical.bin`, `corpus.bin`) allow fast startup by skipping full WAL replay. The CLI casts pointers via `mmap` and immediately begins execution.

## Folder layout

```
use_cases/          <- One file per use-case. Each file declares its own
                       input/output ports (contracts). No business logic here;
                       only wiring and orchestration.
                       (insert, search, update, delete)
domain/entities/    <- Core types: Fingerprint, Memory, Config, Chunk.
domain/algorithms/  <- The math:
                       - SDR encoder (partitioned Tokens / Bigrams / XOR Context)
                       - Dense encoder (k-WTA float32 vector → fingerprint)
                       - Regional weighted Jaccard
                       - Banded Fingerprint LSH (GoldFinger-style)
                       - Lexical index (QLM + Dirichlet smoothing)
                       - Corpus index (IDF tracking)
                       - RRF merger
                       - Reranker (term-coverage boost)
                       - Chunker (conditional sentence-aware splitting)
domain/services/    <- Pure domain orchestration. Each operation lives in
                       `memory/` (types, init, insert, delete, update, search,
                       checkpoint, insert_dense, search_dense, delete_dense).
                       Import the specific file; no umbrella re-export.
 infrastructure/    <- Concrete adapters: mmapped storage (WAL, fingerprint store,
                       chunks mapping store). Imported by domain services when needed.
bindings/           <- Python bindings (nimpy) exposing the same use cases.
benchmarks/         <- BEIR benchmark runner, LongMemEval runner, CRUD benchmark,
                       Python vision suite, `vision_bench.nim` (Nim dense-vector
                       benchmark using insertDense / searchDense use-cases).
tests/              <- Unit tests mirroring the folder layout.
ui/                 <- Desktop UI: Nim webview backend (`ui/backend.nim`) plus
                       React frontend (`ui/frontend/`) built with Vite,
                       TailwindCSS, and DaisyUI. Served via webview.h.
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
- **Storage:** Memory-mapped flat files with self-describing headers (WAL, fingerprint store, chunks mapping store, persisted LSH/lexical/corpus indexes)
- **Index / search structure:** Banded Fingerprint LSH + brute-force Jaccard
- **Lexical lane:** Query Likelihood Model with Dirichlet Smoothing
- **Merger:** Reciprocal Rank Fusion (RRF)
- **Chunking:** Sentence-aware conditional splitting with overlap; threshold configurable via `chunkSaturationThreshold`
- **Delete / Update:** `deleteMemory()` and `updateMemory()` remove entries from LSH and lexical indexes, then free the slot for reuse.
- **Checkpoint:** `checkpoint()` serializes in-memory indexes to disk for fast restart
- **Python bindings:** `bindings/adaline.nim` exposes `Engine` (insert, search, update, delete, stats, checkpoint) via nimpy. Build with `nimble python`.
- **Desktop UI:** `adaline ui` launches a cross-platform webview window running a React frontend. The frontend communicates with the Nim backend via `window.external.invoke` JSON messages. Build with `nimble ui`.
- **Dense-vector encoding:** `domain/algorithms/dense_encoder.nim` provides `encodeDense()` for converting L2-normalised float32 vectors into fingerprints via k-WTA, enabling vision / signal / tabular use cases.
- **Adaptive fingerprint compression:** Three formats selected by active-segment count: sparse (≤20), bitmap (21–157), or raw (≥158). Text SDR fingerprints average ~220 bytes with top-K filtering; dense-vector fingerprints compress to ~19 bytes.
- **Top-K token filtering:** Documents encode only the top-K tokens by IDF (default 12). This cuts fingerprint size to ~220 bytes (~83% savings) while improving nDCG@10 and MRR by ~4 points on SciFact.
- **LSH query optimizations:** Epoch-array dedup (replaces `HashSet` per query), AND-construction (min 2 band hits), fast-path `bandHash` (unrolled, no `mod` for default config), in-place `removeLsh`, and pre-sized table on `loadLsh`. These optimizations more than doubled SciFact query throughput (125 → 277 q/s) with no recall loss.
- **Lexical lane `seq[float]` scoring:** Replaced `Table[uint64, float]` docScores with a dense `seq[float>` + touched-list, eliminating hash-table overhead in the posting-loop hot path.
- **Pre-tokenized rerank cache:** Added `tokenCache: Table[uint64, HashSet[string]]` to `MemoryService`, populated on insert/update. Reranker now does HashSet lookups instead of per-query tokenization.

## Historical Notes

Older branches (`fix/no-graph-weyl-sparse`, `master`) included an HNSW graph layer on top of LSH. This was removed after systematic benchmarking showed that with sparse fingerprints (top-K filtering, k-WTA dense vectors), HNSW graph edges were too weak to provide navigability benefits. LSH + brute-force Jaccard is faster to insert, equally fast (or faster) to query, and yields identical recall across all tested corpus sizes (3K–57K docs).

**Lane ablation (SciFact):** Disabling the lexical lane drops recall@100 from 87.7% to 40.6% and nDCG@10 from 0.604 to 0.348. The lexical lane is critical on text datasets with heavy lexical overlap.

**Lane contribution (SciFact top-k):** Semantic-only 47.3%, Lexical-only 47.1%, Both lanes 5.6%.

## Comment Style

All documentation **must** use Nim doc comments (`## `):

- **Module-level docs:** every `.nim` file starts with a `## ` block describing its purpose.
- **Proc-level docs:** exported procs are preceded by a `## ` block explaining inputs, outputs, and behaviour.
- **No regular `#` comments** for documentation. Use `#` only for inline explanations of non-obvious logic (and keep them minimal).
- **One `#` per line.** Never use `###`, `####`, or other multi-hash decorative styles.

Example:
```nim
## Dense-vector insert service.
## Bypasses text chunking and lexical indexing.
## Encodes a float32 vector directly to a fingerprint and inserts into LSH.

proc insertDense*(service: var MemoryService; vec: seq[float32]): uint64 =
  ...
```

## Agent hygiene

Before finishing any task:

1. **Remove dead code and files.** Delete design docs, scratch scripts, or temporary binaries created during exploration. Do not leave stale documentation that no longer matches the implementation.
2. **Clean generated artifacts.** Remove benchmark data directories, test temp dirs, and compiled binaries that can be regenerated. (These are already in `.gitignore`, but purge them from the working tree too.)
3. **Update living docs.** `AGENTS.md` and `BENCHMARK.md` must reflect the current codebase only. If you added a feature, benchmark, or test, add it to the relevant doc. If you removed something, remove it from the doc.
4. **Run the full test suite.** `nimble test` must pass before you consider a task complete.

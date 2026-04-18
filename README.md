# Adaline

A Nim engine for Sparse Distributed Representations (SDR) using memory-mapped flat files, MinHash LSH, HNSW graph search, and a lexical sidecar with Reciprocal Rank Fusion.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Insert Flow                                                                │
│                                                                             │
│  Input Text ──► SDR Encoder ──► Fingerprint Store (mmap)                    │
│       │              │                                                       │
│       │              └──► MinHash LSH Index (in-memory)                     │
│       │              └──► HNSW Graph (mmap)                                 │
│       │              └──► Lexical Index (in-memory)                         │
│       │                                                                     │
│       └──► WAL (append-only binary, mmap-backed)                            │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  Search Flow                                                                │
│                                                                             │
│  Query ──► SDR Encoder ──► MinHash LSH ──► Candidate Seeds                  │
│       │                         │                                          │
│       │                         └──► HNSW Graph Search (layer-0 descent)    │
│       │                                         │                          │
│       │                                         └──► Top-K Semantic         │
│       │                                                                    │
│       └──► Lexical Index (QLM + Dirichlet) ──► Top-K Lexical              │
│                                                                             │
│                          RRF Merge ──► Term-Coverage Rerank ──► Final     │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Insert Flow (Step-by-Step)

1. **WAL Append**
   - The raw text and a `uint64` MemoryID are appended to `wal.bin`.
   - MemoryID equals the byte offset in the fingerprint store (multiples of 1280).

2. **Corpus Index Update**
   - Term document frequencies are updated incrementally.
   - IDF values are recomputed for seen terms.

3. **SDR Encoding**
   - Text is lowercased and tokenized.
   - **Block A (Tokens, 4096 bits):** Each token is hashed to `scaledProbes` positions based on its IDF. Token bigrams (adjacent word pairs) are also hashed here.
   - **Block B (Character Bigrams, 3072 bits):** A 2-character sliding window hashes to positions. Only letter/digit pairs are kept.
   - **Block C (XOR Context, 3072 bits):** For each token, its hash is XOR-folded with its left and right neighbor hashes, then probed.
   - Probe counts scale with rarity: rare terms get more probes, common terms get fewer.

4. **Fingerprint Store Write**
   - The 1280-byte fingerprint is written to `fingerprints.bin` at offset `MemoryID` via `mmap`.

5. **MinHash LSH Insert**
   - 100 independent MinHash values are computed from active bit positions.
   - The signature is divided into 25 bands of 4 rows.
   - Each band is hashed to a bucket; `MemoryID` is appended to that bucket.

6. **Lexical Index Insert**
   - Tokens are counted per document.
   - The inverted index adds `(MemoryID, frequency)` to each token's postings list.
   - Corpus-wide term frequencies and total token count are updated.

7. **HNSW Graph Insert**
   - A random level is assigned (exponential decay, max 7).
   - The graph store (`graph.bin`) is grown if needed.
   - For layers above the target layer, greedy descent finds the closest neighbor.
   - At each target layer, `efConstruction=200` nearest neighbors are found.
   - Bidirectional edges are created and pruned to `maxNeighbors=32` per layer.

## Search Flow (Step-by-Step)

1. **Query Encoding**
   - The query text runs through the same SDR encoder with IDF scaling.

2. **MinHash LSH Query**
   - The query's MinHash signature is banded and hashed.
   - Any bucket that shares a band hash with the query returns its MemoryIDs.
   - Results are deduplicated.

3. **LSH Wormhole (Semantic Search)**
   - Hash the query through MinHash LSH to retrieve seed MemoryIDs.
   - Drop those seeds into the HNSW graph's lower layers.
   - Execute greedy search outward (`efSearch=64`) to find local optimums.
   - LSH seeds are scored directly and merged with HNSW results.
   - Distance metric: `1.0 - weightedJaccard`.

4. **Lexical Search**
   - Query tokens are looked up in the inverted index.
   - Documents are scored using Query Likelihood with Dirichlet smoothing:
     ```
     Score = sum_q ln(1 + TF(q,D) / (mu * P(q|Corpus))) + |Q| * ln(mu / (|D| + mu))
     ```
   - The accumulator iterates postings directly (no nested per-document lookups).

5. **RRF Merge**
   - Semantic top-K and lexical top-K are merged via Reciprocal Rank Fusion:
     ```
     RRF = 1/(60 + rank_semantic) + 1/(60 + rank_lexical)
     ```

6. **Term-Coverage Rerank**
   - Each candidate document is tokenized and checked for exact query term coverage.
   - A coverage boost (`coverageRatio * 0.5`) is added to the RRF score.
   - Results are re-sorted by the boosted score descending.

## Storage Layout

| File | Purpose | Format |
|------|---------|--------|
| `data/wal.bin` | Append-only text + metadata | `[MemoryID: u64][len: u32][text: bytes]` |
| `data/fingerprints.bin` | Flat fingerprint array | 1280 bytes per fingerprint |
| `data/graph.bin` | Flat HNSW node array | 2056 bytes per node |

All stores are memory-mapped via `mmap` for zero-copy reads.

## Building & Testing

```bash
# Run CLI demo
nim c -d:release cli.nim && ./cli

# Run tests
nim c -r tests/domain/test_fingerprint.nim
nim c -r tests/domain/test_memory_service.nim
nim c -r tests/use_cases/test_search_memories.nim

# Run BEIR benchmark
nim c -d:release benchmarks/benchmark_beir.nim && ./benchmarks/benchmark_beir
```

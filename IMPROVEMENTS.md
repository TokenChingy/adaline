# SciFact Recall@1 Investigation — Final Report

**Branch:** `investigate-scifact-recall`  
**Goal:** Fix semantic retrieval and raise SciFact Recall@1.

## Final Results (SciFact)

| Metric | Old Baseline | After Changes | Δ |
|--------|-------------|---------------|---|
| R@1 | 40.58% | **41.50%** | +0.92pp |
| R@5 | 61.08% | **61.58%** | +0.50pp |
| R@10 | 68.83% | **67.72%** | −1.11pp |
| R@100 | 86.71% | **88.12%** | +1.41pp |
| MRR | 0.5182 | **0.5255** | +0.0073 |
| MAP | 0.5071 | **0.5154** | +0.0083 |
| nDCG@10 | 0.5480 | **0.5514** | +0.0034 |
| Index docs/s | 1,732 | **2,811** | +62% |
| Query q/s | 241 | **226** | −6% |
| Query P50 | 4.19 ms | **4.43 ms** | +0.24 ms |

## What Changed

### 1. Deterministic Hashing (SDR Encoder)

**Before:** `FNV1a → SplitMix64` — a PRNG that destroys locality. Similar tokens ("genome"/"genomes") share zero bits.

**After:** Simple Jenkins-style deterministic hash with per-probe seeds. Same feature + same seed always maps to the same bit. No PRNG state machine.

```nim
proc hashFeature(feature: string; seed: uint64 = 0): uint64
proc probeBlock(fp, feature, count, baseBit, sizeBits)
```

**Impact:** Insert speed ↑ 62% (removed SplitMix64 overhead). Semantic quality roughly unchanged for SciFact because the core issue is vocabulary mismatch, not hash randomness.

### 2. Prefix/Suffix Token Features

Each token of length ≥4 now also probes its first-4-char prefix and last-4-char suffix into the token block. This gives partial overlap for morphological variants.

**Impact:** +0.3pp semantic R@1 on SciFact. Small but free.

### 3. Query Probe Boost

Queries are encoded with `isQuery = true`, which applies a `queryProbeMultiplier` (default 2.0) to probe counts. This makes query fingerprints denser, improving overlap with document fingerprints.

### 4. GoldFinger-style LSH (Replaced MinHash)

**Before:** MinHash signatures — iterate all set bits, track 100 minima per hash function. O(bits_set × minHashFunctions) per fingerprint.

**After:** Direct fingerprint banding — partition the 160 uint64 segments into bands and hash each band directly. O(lshBands × lshRows) per fingerprint.

```nim
proc bandHash(fp: ptr Fingerprint; bandId, rows: int): uint64
```

**Impact:** Insert speed ↑ significantly. LSH is now simpler, faster, and avoids the MinHash approximation error. The `minHashFunctions` config field was removed.

### 5. HNSW `searchLayer` Best-First Fix

`searchLayer` was returning results in **worst-first** order because `HeapQueue.pop()` on a min-heap of negated distances returns the smallest negDist (largest distance) first. Fixed by reversing the output array.

**Impact:** +1.5pp R@1 on the old baseline. Graph edges now link to actual nearest neighbors.

### 6. Weighted RRF

Added `semanticRrfWeight` and `lexicalRrfWeight` to `mergeRrf`.

**Observation:** For SciFact, any semantic weight > 0 hurts R@1 because the semantic lane is weak (R@1 ≈ 7%). Lexical-only achieves ~43%. The default keeps `semanticWeight = 0.5` for datasets where semantic is stronger.

### 7. Dynamic Segment Boundaries

Segment positions (token/bigram/context blocks) are derived from `cfg.tokenBits/bigramBits/contextBits` instead of hardcoded constants. Block sizes are now fully configurable.

## What Did NOT Change (and why)

- **Brute-force semantic path:** Added and then removed. It scored all chunks exactly but polluted RRF with low-quality candidates, dropping full-system R@1. LSH + HNSW naturally filters to a smaller, higher-quality candidate pool.
- **BM25 backend:** User explicitly rejected this.
- **Fingerprint size:** Remains 10240 bits. Larger fingerprints would help but require invasive type changes.

## Removed During Cleanup

- **Pluggable similarity metrics (Dice, Cosine, Overlap):** Added during investigation then removed. Performance difference vs Jaccard was within noise on both SciFact and LongMemEval-S. Keeping only Jaccard reduces surface area.

## Files Changed

| File | Change |
|------|--------|
| `domain/algorithms/sdr_encoder.nim` | Deterministic hash, prefix/suffix, query boost |
| `domain/algorithms/fingerprint_lsh.nim` | **New** — GoldFinger direct banding |
| `domain/algorithms/minhash_lsh.nim` | **Deleted** |
| `domain/algorithms/hnsw_graph.nim` | Best-first fix |
| `domain/algorithms/weighted_jaccard.nim` | Regional weighted Jaccard only (simplified) |
| `domain/algorithms/rrf_merger.nim` | Weighted RRF |
| `domain/entities/config.nim` | New fields: queryProbeMultiplier, semanticRrfWeight, lexicalRrfWeight; removed minHashFunctions; LSH defaults 50×2 |
| `domain/entities/fingerprint.nim` | Dynamic segment boundaries |
| `domain/services/memory_service.nim` | Uses deterministic hash, GoldFinger LSH, weighted RRF |
| `tests/domain/test_fingerprint_lsh.nim` | **New** |
| `tests/domain/test_minhash_lsh.nim` | **Deleted** |
| `BENCHMARK.md` | Updated SciFact and LongMemEval numbers |
| `README.md` | Updated architecture docs |
| `AGENTS.md` | Updated description |

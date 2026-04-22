# Storage Efficiency Investigation

## Branch: `exp/storage-efficiency`

**Goal:** Identify and prototype storage strategies that are better (higher recall / consistency), faster (lower latency, faster startup), and more memory efficient (lower RAM footprint) than the current mmap-based flat-file approach.

---

## Current Baseline (master)

| Metric | Value | Notes |
|--------|-------|-------|
| Fingerprint size | 1,280 bytes | `array[160, uint64]` = 10,240 bits |
| Graph node size | 1,032 bytes | Hardcoded 8 layers × 32 neighbors |
| Text storage | All in RAM (`textCache: Table[uint64, string]`) | Unbounded growth |
| Timestamp storage | All in RAM (`timestampCache`) | Redundant with WAL |
| Reverse edges | All in RAM (`hnswReverseIndex`) | Doubles edge memory |
| WAL | Append-only, never truncated | Full replay every boot |
| chunks.bin | Append-only, never rewritten | Dead mappings replayed |
| Graph store header | Unused | Fingerprint header is sole source of truth |
| Growth | 64 MiB chunks, remap on grow | Close + reopen `MemFile` |

### Estimated waste at default config (`M=16`, `maxLayers=8`)
- Graph node allocates 256 `uint32` neighbor slots
- Default uses 16 neighbors/layer × 8 layers = 128 slots
- **Waste: 50% of graph.bin disk space and mmap cache**

---

## Hypotheses & Experiments

### H1: Variable-Size Graph Nodes ✅ PROTOTYPED
**Hypothesis:** Making `HnswNode` record size configurable at runtime (based on `hnswMaxLayers` and `hnswMaxNeighbors`) will cut graph storage roughly in half at default settings, with better cache locality and no functional change.

**Status:** Implemented on this branch. All tests pass.

**Results:**
- Default config (`M=16`, `maxLayers=8`): record size drops from **1,032 bytes → 520 bytes**
- **Disk savings: ~49.6%** of graph.bin footprint
- At 64 MiB growth chunks, slots per chunk rise from ~65,028 → ~129,055
- No regression in search, insert, or delete behaviour (validated via `benchmarks/validate_variable_nodes.nim`)

**Implementation notes:**
- Added `HnswNodeView` type that wraps a raw pointer + `maxNeighbors`/`maxLayers`
- All production code now uses views; legacy `ptr HnswNode` API retained for unit tests
- Graph store header extended with `hnswMaxLayers` and `hnswMaxNeighbors` fields (validated on open)
- `hnsw_graph.nim` procs now accept `graphRecordSize: uint64`

**Risks addressed:**
- Pointer arithmetic is isolated in `getHnswNodeView` and `getHnswNodePtr`
- Header validation prevents opening a store with incompatible config
- `removeNeighbor` shift cost is unchanged (still O(M) per layer)

---

### H2: Unified Record Store
**Hypothesis:** Storing fingerprint + graph node in a single contiguous record improves cache locality during HNSW search (which touches both), halves the number of mmap regions, and simplifies growth logic.

**Risks:**
- Larger minimum record size means more padding if graph node is small
- Fingerprint and graph have different access patterns (fingerprints read heavily, graph written heavily)
- Coupled growth means fingerprint store can't grow independently

**Approach:**
1. Single `records.bin` with header + `(fingerprint + graphNode)` pairs.
2. One mmap, one freelist, one growth path.

---

### H3: On-Demand Text Store (Paged / Slab)
**Hypothesis:** Eliminating `textCache` by storing texts in a separate paged file and loading on demand during result materialization will dramatically reduce RAM usage, especially for large corpora.

**Risks:**
- Search result materialization becomes I/O bound
- Need a separate index (B-tree, radix tree, or simple offset table) to map `memoryId → (offset, length)`
- Deletion leaves holes; need compaction or a freelist for text blobs

**Approach:**
1. `texts.bin`: append-only or paged file storing `(length, bytes)` blobs.
2. `textIndex.bin`: dense array of `(offset: uint64, length: uint32)` indexed by `memoryId`, stored in its own mmap or at the start of `texts.bin`.
3. On search, only load texts for the top-K results.
4. Delete: zero the index entry, add text offset to a freelist.

**Memory impact estimate:**
- Current: `sum(len(text) for all memories)` in RAM
- Proposed: `8 + 4 bytes × maxId` in RAM for index; texts on disk

---

### H4: WAL Compaction + Checkpoint Integration
**Hypothesis:** Truncating the WAL after checkpoint (by rewriting only live entries) will make startup time O(live memories) instead of O(total history).

**Risks:**
- Adds write amplification during checkpoint
- Need to ensure crash safety (write new WAL, fsync, atomically swap)

**Approach:**
1. During `checkpoint()`, iterate `textCache` (or new text store) and write a compact WAL.
2. Atomically replace `wal.bin`.
3. On restart, if checkpoint is fresh, skip WAL replay entirely by loading text index directly.

---

### H5: Embedded Reverse Edges
**Hypothesis:** Storing reverse edges in the graph file (or eliminating the in-memory reverse index) removes the RAM doubling of edges.

**Risks:**
- Reverse edges are only needed on delete, which is rare
- Storing them on disk means delete becomes I/O heavy
- Rebuilding reverse index on demand for delete is O(N × M)

**Approach:**
- Option A: Store reverse edges in a separate mmap'd file, loaded on demand for delete.
- Option B: Accept O(N × M) rebuild on delete (rare operation).

---

### H6: Inline Timestamps & Metadata
**Hypothesis:** Storing timestamps (and other small metadata) in the fingerprint or graph header area eliminates the `timestampCache` Table.

**Approach:**
- Reserve 8 bytes in each fingerprint record for `createdAt` timestamp.
- Or store in graph node (currently has 6 reserved bytes).

---

## Measurement Plan

For each experiment, measure:

1. **Disk usage** (`du -h data/` after inserting N items)
2. **RSS memory** (`/usr/bin/time -l` or `ps` during search)
3. **Insert throughput** (items/sec)
4. **Search latency** (p50, p99)
5. **Startup time** (ms from init to first query)
6. **Recall@K** on SciFact (must not regress)

Baseline benchmark corpus: **SciFact** (5,183 docs) via existing BEIR runner.

---

## Proposed Execution Order

1. **Quick win: H1 (Variable graph nodes)** — Low risk, clear benefit, minimal API change.
2. **Big win: H3 (Paged text store)** — Highest RAM impact, requires more design.
3. **Synergy: H4 (WAL compaction)** — Builds on H3, fixes startup time.
4. **Integration: H2 (Unified store)** — Revisit after H1 and H3 are understood.
5. **Polish: H5 + H6** — Clean up remaining RAM overhead.

---

## Notes

- All changes must preserve `nimble test` pass rate.
- Fingerprint format (1280 bytes) is sacred — it is the public data contract.
- Checkpoint format can evolve if needed, but prefer backward compatibility or clear migration path.
- Keep `## ` doc comment style per AGENTS.md.

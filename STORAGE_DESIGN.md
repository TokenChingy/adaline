# Adaline Storage Redesign — SQLite-Inspired

## Current Problems

| Problem | Impact |
|---------|--------|
| **mmap close/reopen on grow** | Every capacity expansion calls `munmap` + `mmap`. Invalidates pointers, expensive syscall storm during bulk insert. |
| **memoryId = byte offset** | IDs are 0, 1280, 2560, 3840… Fragile, wastes address space, prevents dense freelist reuse. |
| **No file headers** | No magic bytes, version, or metadata. Corruption is silent. No tooling possible without source. |
| **No freelist** | Delete/update impossible. Holes are permanent. |
| **Full index rebuild on startup** | LSH, lexical, HNSW rebuilt from WAL replay = O(n) on every open. |
| **Direct mmap writes** | `copyMem` into mmap = SIGBUS risk on full disk, uncommitted state leaks to other processes. |

## SQLite Patterns to Adopt

### 1. Pre-Allocated Growth Regions

Instead of closing/reopening the mmap on every grow, **pre-allocate in large chunks** (e.g., 64 MB) and remap only when the chunk is exhausted. `munmap` + `mmap` happens rarely because a single 64 MB chunk holds ~52K fingerprints.

```
Current:  grow from 1024 → 2048 → 4096 → 8192 slots (4 remaps for 8× growth)
SQLite-inspired: grow from 1024 → 1024 + 64MB_chunk (1 remap for same growth)
```

### 2. Slot Addressing (Dense IDs)

Replace `memoryId = byte_offset` with `memoryId = slot_index`. Offset computed as `slot_index * record_size`.

```
Before: memoryId ∈ {0, 1280, 2560, 3840, ...}
After:  memoryId ∈ {0, 1, 2, 3, ...}
```

Benefits:
- Dense: 32-bit slot index can address 4B records (vs 32-bit byte offset = 3.3M fingerprints).
- Freelist works naturally: push/pop slot indices.
- HNSW graph stores smaller IDs.

### 3. Self-Describing File Header

Every store file starts with a 256-byte header:

| Offset | Size | Field |
|--------|------|-------|
| 0 | 4 | Magic: `"ADLN"` |
| 4 | 2 | Version (major << 8 \| minor) |
| 6 | 2 | Record size (bytes) |
| 8 | 8 | Record count (used slots) |
| 16 | 8 | Capacity (allocated slots) |
| 24 | 8 | Freelist head slot index (`uint64.high` = empty) |
| 32 | 8 | Freelist count |
| 40 | 216 | Reserved / padding |

Total: 256 bytes. Record 0 starts at byte 256.

### 4. Freelist for Deletes

On delete, push the slot index onto a **stack** stored in the file header:

```
freelist_head → slot_5 → slot_2 → slot_9 → 0 (null)
```

Each free slot stores the *next* free slot index as an in-band linked list. No extra memory needed.

On insert:
1. Pop from freelist if non-empty (reuse slot).
2. Otherwise append at `record_count` (grow file if needed).

### 5. mmap for Reads, Explicit Writes for Mutations

**Query path** (read-only, hot):
- mmap the store file with `PROT_READ`.
- Cast pointers directly from mmap. Zero copy, kernel-page-cache friendly.

**Insert path** (write, cold):
- Build fingerprint/graph in a private heap buffer.
- Use `pwrite()` or `msync` to push to disk.
- Or write to mmap with `MAP_SHARED` but only after ensuring disk space (pre-allocated regions help here).

This matches SQLite: reads via `xFetch()` (mmap), writes via `xWrite()` (heap copy + explicit push).

### 6. Persisted Indexes (Avoid Rebuild on Startup)

Current: On startup, replay WAL → rebuild LSH → rebuild lexical → rebuild HNSW graph.

SQLite-inspired: The main store files are the source of truth. WAL is only for crash recovery.

**New files:**
- `lsh.bin` — serialized LSH buckets (Table → flat array).
- `lexical.bin` — serialized postings lists.
- `corpus.bin` — serialized CorpusIndex (token frequencies, IDF table).

**Startup path:**
1. Read headers from all store files (O(1)).
2. If `lsh.bin` exists and checksum matches: mmap and use directly.
3. If `lsh.bin` is missing or stale: rebuild from fingerprints and write it back.
4. Same for `lexical.bin` and `corpus.bin`.
5. WAL replay only for memories inserted *after* the last checkpoint.

**Checkpoint:** After bulk insert, call `checkpoint()` to serialize LSH, lexical, and corpus to disk. Truncate WAL.

## New File Layout

```
{dataDir}/
  store.adaline          # Unified or header + data files
  wal.bin                # Append-only text records (crash recovery)
  lsh.bin                # Serialized LSH index
  lexical.bin            # Serialized lexical postings
  corpus.bin             # Serialized corpus statistics
```

Option A: **Keep separate files** (fingerprint.bin, graph.bin) with headers.
Option B: **Unified file** with page types (like SQLite). More complex; start with A.

## Implementation Phases

### Phase 1: Headers + Slot Addressing
- Add 256-byte header to fingerprint.bin and graph.bin.
- Switch from byte-offset IDs to slot-index IDs.
- Update all pointer math in memory_service, hnsw_graph, etc.

### Phase 2: Pre-Allocated Growth + Freelist
- Pre-allocate 64MB chunks instead of doubling.
- Add freelist push/pop to headers (in-band linked list).
- Add `deleteMemory()` to memory_service.
- HNSW graph nodes for deleted slots marked with `layerCount = 0`.

### Phase 3: Persisted Indexes
- Serialize LSH buckets to `lsh.bin`.
- Serialize lexical index to `lexical.bin`.
- Serialize corpus index to `corpus.bin`.
- Add `checkpoint()` API.
- On startup: load persisted indexes, replay only WAL entries after checkpoint offset.

### Phase 4: mmap Reads + Explicit Writes (future)
- Open stores as `fmRead` during queries.
- Open stores as `fmReadWrite` only during ingestion.
- Use heap buffers for mutations, `pwrite` for commits.

# Adaline — Operation Flows

This doc walks through the exact data flow for each use-case so agents don't have to reverse-engineer it from the code.

---

## Dependency Overview

```mermaid
flowchart TD
    subgraph UseCases["use_cases/"]
        UC_INSERT[insert_memory]
        UC_SEARCH[search_memories]
        UC_UPDATE[update_memory]
        UC_DELETE[delete_memory]
    end

    subgraph Domain["domain/"]
        subgraph Services["services/memory/"]
            SVC_INIT[init]
            SVC_INSERT[insert]
            SVC_SEARCH[search]
            SVC_UPDATE[update]
            SVC_DELETE[delete]
            SVC_CHK[checkpoint]
        end
        subgraph Algorithms["algorithms/"]
            ENC[sdr_encoder]
            LSH[fingerprint_lsh]
            HNSW[hnsw_graph]
            LEX[lexical_index]
            CORP[corpus_index]
            RRF[rrf_merger]
            RER[reranker]
            CHK[chunker]
        end
        subgraph Entities["entities/"]
            TYPES[MemoryService, Memory, Config]
        end
    end

    subgraph Infra["infrastructure/"]
        STORE[mmapped_storage]
    end

    UC_INSERT --> SVC_INSERT
    UC_SEARCH --> SVC_SEARCH
    UC_UPDATE --> SVC_UPDATE
    UC_DELETE --> SVC_DELETE

    SVC_INIT --> STORE
    SVC_INSERT --> ENC & LSH & HNSW & LEX & CORP & CHK & STORE
    SVC_SEARCH --> ENC & LSH & HNSW & LEX & CORP & RRF & RER
    SVC_UPDATE --> SVC_DELETE & SVC_INSERT
    SVC_DELETE --> LSH & LEX & HNSW & STORE
    SVC_CHK --> LSH & LEX & CORP
```

---

## 1. Insert

```mermaid
sequenceDiagram
    actor User
    participant UC as use_cases/insert
    participant SVC as memory/insert
    participant WAL as mmapped_storage
    participant CORP as corpus_index
    participant CHK as chunker
    participant ENC as sdr_encoder
    participant LSH as fingerprint_lsh
    participant LEX as lexical_index
    participant HNSW as hnsw_graph

    User->>UC: insertMemory(content)
    UC->>SVC: service.insert(content)
    SVC->>WAL: allocSlot() → parentId
    SVC->>WAL: appendWal(parentId, timestamp, content)
    SVC->>SVC: textCache[parentId] = content
    SVC->>CORP: addMemory(content)
    SVC->>CHK: splitIntoChunks(content, cfg) → chunks

    alt chunks.len == 1 (unchunked)
        SVC->>SVC: chunkId = parentId
        SVC->>WAL: appendChunkMapping(parentId, chunkId)
        SVC->>ENC: encodeSdr(content, cfg, corpus) → fp
        SVC->>WAL: writeFingerprintUnsafe(chunkId, fp)
        SVC->>LSH: insertLsh(lsh, fp, chunkId)
        SVC->>LEX: addMemory(lexical, chunkId, content)
        SVC->>HNSW: insertHnsw(..., chunkId, fp, ...)
    else chunks.len > 1
        loop for each chunkText
            SVC->>WAL: chunkId = allocSlot()
            SVC->>SVC: chunkToParent[chunkId] = parentId
            SVC->>WAL: appendChunkMapping(parentId, chunkId)
            SVC->>ENC: encodeSdr(chunkText, cfg, corpus) → fp
            SVC->>WAL: writeFingerprintUnsafe(chunkId, fp)
            SVC->>LSH: insertLsh(lsh, fp, chunkId)
            SVC->>LEX: addMemory(lexical, chunkId, chunkText)
            SVC->>HNSW: insertHnsw(..., chunkId, fp, ...)
        end
    end
    SVC-->>UC: parentId
    UC-->>User: InsertMemoryOutput(memoryId: parentId)
```

### Step-by-step

1. **Allocate parent slot** `storage.allocSlot()` → `parentId` (dense uint64 slot).
2. **WAL append** — write `(parentId, timestamp, content)` to `wal.bin` for durability.
3. **Cache** — store `textCache[parentId]` and `timestampCache[parentId]`.
4. **Corpus** — `corpus.addMemory(content)` updates document-frequency table and IDF scores.
5. **Chunk** — `splitIntoChunks(content, cfg)` checks block saturation; if any block exceeds `chunkSaturationThreshold`, splits on sentence boundaries with overlap.
6. **For each chunk**:
   - Allocate chunk slot (or reuse parent slot if unchunked).
   - Write chunk→parent mapping to `chunks.bin`.
   - `encodeSdr(text, cfg, corpus)` produces a 10240-bit fingerprint (Tokens / Bigrams / XOR Context blocks).
   - Write fingerprint to `fingerprints.bin` at slot offset.
   - `insertLsh(lsh, fp, chunkId)` hashes fingerprint bands into LSH buckets.
   - `addMemory(lexical, chunkId, text)` tokenizes and builds postings list.
   - `insertHnsw(...)` greedily searches each layer, adds bidirectional edges, updates entry point if needed, and writes to `graph.bin`.
7. Return `parentId`.

---

## 2. Search

```mermaid
sequenceDiagram
    actor User
    participant UC as use_cases/search
    participant SVC as memory/search
    participant ENC as sdr_encoder
    participant LSH as fingerprint_lsh
    participant HNSW as hnsw_graph
    participant LEX as lexical_index
    participant RRF as rrf_merger
    participant RER as reranker

    User->>UC: searchMemories(query, topK)
    UC->>SVC: service.search(query, k)
    SVC->>ENC: encodeSdr(query, cfg, corpus, isQuery=true) → qfp
    SVC->>LSH: queryLsh(lsh, qfp) → seedIds
    SVC->>HNSW: searchHnsw(graphMem, fpMem, seeds, entryPt, qfp, k, cfg) → semanticResults
    SVC->>LEX: searchLexical(lexical, query, k) → lexicalResults
    SVC->>RRF: mergeRrf(semantic, lexical, k, rrfK, semW, lexW) → mergedChunks
    SVC->>SVC: map chunks → parents, dedupe, keep best score
    SVC->>SVC: sort by score desc, trim to k
    SVC->>RER: rerank(query, candidates, textCache, cfg)
    SVC-->>UC: seq[Memory]
    UC-->>User: SearchMemoriesOutput(memories)
```

### Step-by-step

1. **Query fingerprint** — `encodeSdr(query, cfg, corpus, isQuery=true)` with denser probes (`queryProbeMultiplier`).
2. **LSH seeds** — `queryLsh(lsh, qfp)` band-hashes the query fingerprint and returns all IDs colliding in any bucket.
3. **Semantic lane** — `searchHnsw(graphMem, fpMem, seeds, entryPoint, qfp, k, cfg)`:
   - Uses seeds + entry point as starting nodes.
   - Greedy best-first search per layer, computing weighted Jaccard against `qfp`.
   - Returns top-k chunk IDs with similarity scores.
4. **Lexical lane** — `searchLexical(lexical, query, k)` scores chunks via Query Likelihood Model with Dirichlet smoothing.
5. **RRF merge** — `mergeRrf(semantic, lexical, k, rrfK, semWeight, lexWeight)` combines both lanes using Reciprocal Rank Fusion. Memories present in both lanes get boosted.
6. **Chunk→parent mapping** — for each merged chunk, look up `chunkToParent.getOrDefault(chunkId, chunkId)`. Deduplicate by parent, keeping the highest score.
7. **Build candidates** — construct `Memory` objects from `textCache`, `timestampCache`, and the deduplicated scores.
8. **Sort & trim** — sort descending by score, keep top `k`.
9. **Rerank** — `rerank(query, candidates, textCache, cfg)` applies a term-coverage boost. Exact query matches get pushed to the top.
10. Return candidates.

---

## 3. Update

```mermaid
sequenceDiagram
    actor User
    participant UC as use_cases/update
    participant SVC_U as memory/update
    participant SVC_D as memory/delete
    participant SVC_I as memory/insert

    User->>UC: updateMemory(memoryId, content)
    UC->>SVC_U: service.updateMemory(id, content)
    alt id not in textCache
        SVC_U-->>UC: (no-op)
    else
        SVC_U->>SVC_D: deleteMemory(service, id)
        Note over SVC_D: heals HNSW, frees slots, clears indexes
        SVC_U->>SVC_I: (re-insert new content reusing same parentId)
        Note over SVC_I: WAL append, chunk, encode, index
    end
    UC-->>User: UpdateMemoryOutput(memoryId: id)
```

### Step-by-step

1. **Guard** — if `parentId` not in `textCache`, return immediately (idempotent no-op).
2. **Delete old** — call `deleteMemory(service, parentId)` which heals the HNSW graph, removes from LSH/lexical, frees slots, and clears caches.
3. **Insert new** — reuse the same `parentId`:
   - WAL append `(parentId, newTimestamp, newContent)`.
   - Update `textCache` and `timestampCache`.
   - `corpus.addMemory(newContent)`.
   - Chunk, encode, index exactly like **Insert**.
4. The parent ID is preserved; only the underlying chunks change.

---

## 4. Delete

```mermaid
sequenceDiagram
    actor User
    participant UC as use_cases/delete
    participant SVC as memory/delete
    participant HNSW as hnsw_graph
    participant LSH as fingerprint_lsh
    participant LEX as lexical_index
    participant WAL as mmapped_storage

    User->>UC: deleteMemory(memoryId)
    UC->>SVC: service.deleteMemory(id)
    alt id not in textCache
        SVC-->>UC: (no-op)
    else
        SVC->>SVC: collect chunkIds for parent
        loop for each chunkId
            SVC->>HNSW: heal forward edges
            Note over HNSW: for each neighbor N of chunkId,<br/>remove chunkId from reverseIndex[N]
            SVC->>HNSW: heal backward edges
            Note over HNSW: for each M pointing to chunkId,<br/>remove chunkId from M's neighbor list
            SVC->>LSH: removeLsh(lsh, fpPtr, chunkId)
            SVC->>LEX: removeMemory(lexical, chunkId, text)
            SVC->>WAL: freeSlot(chunkId)
            SVC->>SVC: del chunkToParent[chunkId]
        end
        SVC->>SVC: del textCache[id], timestampCache[id]
        SVC->>WAL: syncHeader()
    end
    UC-->>User: DeleteMemoryOutput(memoryId: id)
```

### Step-by-step

1. **Guard** — if `parentId` not in `textCache`, return immediately.
2. **Collect chunks** — iterate `chunkToParent` to find all `chunkId`s where `chunkToParent[chunkId] == parentId`. If none, treat the parent itself as the sole chunk.
3. **For each chunk**:
   - **Forward edge healing** — for each layer, iterate the chunk's neighbors. For each neighbor `N`, remove `chunkId` from `hnswReverseIndex[N]`.
   - **Backward edge healing** — look up `hnswReverseIndex[chunkId]`. For each node `M` that pointed to the chunk, call `removeNeighbor(mnode, layer, chunkId)` on every layer. Then delete `hnswReverseIndex[chunkId]`.
   - **Remove from LSH** — get fingerprint pointer, call `removeLsh(lsh, fpPtr, chunkId)`.
   - **Remove from lexical** — call `removeMemory(lexical, chunkId, chunkText)`.
   - **Free slot** — `storage.freeSlot(chunkId)` pushes the slot onto the freelist and zeroes the graph node.
   - **Delete mapping** — `chunkToParent.del(chunkId)`.
4. **Clean parent caches** — `textCache.del(parentId)`, `timestampCache.del(parentId)`.
5. **Sync header** — persist updated freelist and record count to `fingerprints.bin` header.

---

## 5. Checkpoint

```mermaid
sequenceDiagram
    actor User
    participant SVC as memory/checkpoint
    participant LSH as fingerprint_lsh
    participant LEX as lexical_index
    participant CORP as corpus_index

    User->>SVC: checkpoint(service)
    SVC->>SVC: walOffset = storage.walSize
    SVC->>LSH: saveLsh(lsh, lshPath, walOffset)
    SVC->>LEX: saveLexical(lexical, lexicalPath, walOffset)
    SVC->>CORP: saveCorpus(corpus, corpusPath, walOffset)
```

### Step-by-step

1. Capture current `walSize` as the offset.
2. Serialize in-memory LSH buckets to `lsh.bin`.
3. Serialize lexical postings to `lexical.bin`.
4. Serialize corpus frequencies to `corpus.bin`.
5. On next startup, `initMemoryService` loads these three indexes and only replays WAL entries written **after** `walOffset`.

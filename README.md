# Adaline

A Nim vector search engine that turns text into **10240-bit sparse fingerprints** and searches them with HNSW + LSH, plus a lexical index for token-based matching. Results are fused with Reciprocal Rank Fusion and reranked by term coverage.

- **Conditional chunking** — long text splits into sentence-aware chunks when fingerprints approach saturation
- **Delete & update** — `deleteMemory()` physically heals HNSW edges via a reverse edge index, removes from all indexes, and returns slots to a freelist. `updateMemory()` preserves the parent ID while replacing content.
- **Checkpoint / fast restart** — `checkpoint()` persists indexes to disk; restarts only replay WAL after the checkpoint
- **Single binary, no dependencies** — memory-mapped files, no external services

---

## How it works

Adaline uses a **dual-lane retrieval architecture**: every document is indexed simultaneously into a **semantic lane** (sparse fingerprint + HNSW graph + LSH seeds) and a **lexical lane** (inverted index with QLM scoring). At query time both lanes run in parallel; their results are fused with Reciprocal Rank Fusion and reranked by term coverage.

### Flow overview

**Insert** is the only write path. Content enters through the WAL for durability, is conditionally split into sentence-aware chunks, and each chunk is fingerprinted and added to all three indexes (LSH, HNSW, lexical). Short content skips chunking and the parent ID serves as its own chunk ID.

**Search** runs two independent retrievals: LSH seeds jump-start an HNSW graph descent for semantic neighbors, while the lexical lane scores documents with Query Likelihood + Dirichlet smoothing. RRF merges the two ranked lists, chunk IDs are resolved back to parent memories (deduplicated), and a term-coverage reranker boosts exact matches.

**Update** is an atomic delete-then-insert that preserves the parent ID. Old chunks are physically removed from all indexes (with HNSW edge healing via the reverse edge index), new chunks are inserted, and the WAL logs the replacement.

**Delete** physically removes every chunk of a parent from all indexes. The reverse edge index makes HNSW healing possible without rebuilding the graph. Freed slots go to a freelist for reuse.

---

### Insert

Takes a text string, writes it to the WAL, and indexes it across three lanes:
an LSH index (for seeding), an HNSW graph (for approximate search), and a
lexical inverted index (for token matching). Long text is automatically split
into sentence-aware chunks; each chunk gets its own fingerprint and graph node.

```
  INPUT: content (string)
         │
         ▼
  +-----------------------------------------+
  | 1. ALLOCATE PARENT ID                   |
  |    allocId() --> parentId (uint64)      |
  |                                         |
  | 2. WRITE-AHEAD LOG (durability)         |
  |    wal.bin += (parentId, timestamp,     |
  |                 content)                |
  |                                         |
  | 3. UPDATE IN-MEMORY TABLES              |
  |    textCache[parentId] = content        |
  |    timestampCache[parentId] = now       |
  |    corpus.addMemory(content)            |
  |      --> updates doc-freq / IDF tables  |
  +--------------------+--------------------+
                       |
                       ▼
  +-----------------------------------------+
  | 4. CHUNKING                             |
  |    splitIntoChunks(content, cfg)        |
  |                                         |
  |    +-- short text --+  +-- long text --+|
  |    | 1 chunk only   |  | N chunks      ||
  |    | (no split)     |  | (sentence-    ||
  |    +--------+-------+  |  aware)       ||
  |             |          +--------+------+|
  +-------------|------------------|--------+
                |                  |
                ▼                  ▼
  +--------------+    +---------------------------+
  | parentId IS  |    | for each chunkText:       |
  | the chunkId  |    |   allocId() --> chunkId   |
  |              |    |                           |
  | encodeSdr()  |    |   chunkToParent[chunkId]  |
  |   10240-bit  |    |     = parentId            |
  |   fingerprint|    |   chunks.bin += mapping   |
  |              |    |                           |
  | write to:    |    |   encodeSdr() --> fp      |
  | fingerprints |    |     10240-bit fingerprint |
  | .bin[id]     |    |   write to:               |
  |              |    |   fingerprints.bin[id]    |
  | lshInsert()  |    |                           |
  |   buckets[   |    |   lshInsert()             |
  |   band,hash] |    |     buckets[band,hash]    |
  |              |    |     += chunkId            |
  | lexicalAdd() |    |                           |
  |   postings[  |    |   lexicalAdd()            |
  |   term] +=   |    |     postings[term]        |
  |   (id,freq)  |    |     += (chunkId, freq)    |
  |              |    |                           |
  | hnswInsert() |    |   hnswInsert()            |
  |   graph.bin  |    |     graph.bin[id]         |
  |   [id]       |    |     gets edges + layer    |
  +--------------+    +---------------------------+
                |                  |
                +--------+---------+
                         |
                         ▼
                OUTPUT: parentId (uint64)
```

### Search

Takes a query string, runs it through both a **semantic lane** (fingerprint
similarity via HNSW) and a **lexical lane** (inverted index with QLM scoring),
then fuses the two lists with Reciprocal Rank Fusion. Results are mapped from
chunks back to parent memories, deduplicated, and reranked by term coverage.

```
  INPUT: query (string), k (int)
         |
         ▼
  +------------------------------------------+
  | 1. QUERY FINGERPRINT                     |
  |    encodeSdr(query, isQuery=true)        |
  |      --> denser probes than insert       |
  |      --> qfp (10240-bit fingerprint)     |
  +--------------------+---------------------+
                       |
         +-------------+-------------+
         |                           |
         ▼                           ▼
  +-----------------+      +-------------------+
  | 2a. LSH SEEDS   |      | 2b. HNSW SEARCH   |
  |                 |      |                   |
  | queryLsh(qfp)   |      | searchHnsw(       |
  |   band-hash each|      |   seeds,          |
  |   band of qfp   |      |   entryPoint,     |
  |   collect all   |      |   qfp, k, cfg)    |
  |   colliding IDs |      |                   |
  +--------+--------+      | greedy best-      |
           |               | first descent     |
           |               | per layer         |
           |               |                   |
           |               | weighted Jaccard  |
           |               | vs. qfp           |
           |               +---------+---------+
           |                         |
           |                         |
  +--------+--------+                |
  | 3. LEXICAL LANE |<---------------+
  |                 |
  | searchLexical(  |
  |   query, k)     |
  |                 |
  | tokenize query  |
  | score each doc  |
  | with QLM +      |
  | Dirichlet(mu)   |
  +--------+--------+
           |
           ▼
  +------------------------------------------+
  | 4. FUSION                                |
  |    mergeRrf(semanticResults,             |
  |              lexicalResults,             |
  |              k, rrfK,                    |
  |              semWeight, lexWeight)       |
  |                                          |
  |    RRF formula:                          |
  |    score = w_sem/(rrfK+rank_sem)         |
  |          + w_lex/(rrfK+rank_lex)         |
  |                                          |
  |    Items in BOTH lanes get boosted       |
  +--------------------+---------------------+
                       |
                       ▼
  +------------------------------------------+
  | 5. CHUNK --> PARENT RESOLUTION           |
  |    for each merged chunkId:              |
  |      parentId = chunkToParent.getOrDef(  |
  |                 chunkId, chunkId)        |
  |    deduplicate by parent, keep max score |
  +--------------------+---------------------+
                       |
                       ▼
  +------------------------------------------+
  | 6. RERANK                                |
  |    sort by score desc                    |
  |    trim to k                             |
  |    rerank(query, candidates,             |
  |           textCache, cfg)                |
  |      --> term-coverage boost             |
  |      --> exact match --> top             |
  +--------------------+---------------------+
                       |
                       ▼
              OUTPUT: seq[Memory]
                      {id, content, score, createdAt}
```

### Update

Preserves the parent ID while atomically replacing the content. The old chunks
are deleted (healing the HNSW graph and freeing slots) and new chunks are
inserted with the **same** parent ID.

```
  INPUT: id (uint64), content (string)
         |
         ▼
      +--+--+
      |known?|    -->  textCache.hasKey(id)
      +--+--+
       no/ \yes
         /   \
        ▼     ▼
     (no-op)  |
              ▼
  +------------------------------------------+
  | 1. DELETE OLD                            |
  |    deleteMemory(service, id)             |
  |                                          |
  |    |-- find all chunkIds for this parent |
  |    |-- heal HNSW forward edges           |
  |    |    (remove from reverseIndex[N])    |
  |    |-- heal HNSW backward edges          |
  |    |    (remove from predecessors' lists)|
  |    |-- remove from LSH buckets           |
  |    |-- remove from lexical postings      |
  |    |-- freeId(chunkId) --> freelist      |
  |    |-- del chunkToParent[chunkId]        |
  |                                          |
  |    |-- del textCache[id]                 |
  |       del timestampCache[id]             |
  +--------------------+---------------------+
                       |
                       ▼
  +------------------------------------------+
  | 2. INSERT NEW (same parentId)            |
  |                                          |
  |    wal.bin += (id, newTimestamp, content)|
  |    textCache[id] = content               |
  |    corpus.addMemory(content)             |
  |    chunk, encode, index exactly like     |
  |    Insert step 4-5 above                 |
  |    ...but reuse parentId as root         |
  +--------------------+---------------------+
                       |
                       ▼
              OUTPUT: id (unchanged)
```

### Delete

Physically removes a memory and all its chunks from every index. Uses an
in-memory **reverse edge index** to heal HNSW neighbor lists so no orphaned
references remain. Freed IDs go to a freelist for reuse.

```
  INPUT: id (uint64)
         |
         ▼
      +--+--+
      |known?|    -->  textCache.hasKey(id)
      +--+--+
       no/ \yes
         /   \
        ▼     ▼
     (no-op)  |
              ▼
  +------------------------------------------+
  | 1. COLLECT CHUNKS                        |
  |    chunkIds = all entries in             |
  |    chunkToParent where value == id       |
  |                                          |
  |    (if none, treat parent as unchunked   |
  |     and use id itself)                   |
  +--------------------+---------------------+
                       |
                       ▼
  +------------------------------------------+
  | 2. FOR EACH CHUNK                        |
  |                                          |
  |    |-- HEAL FORWARD EDGES                |
  |    |   for each layer of chunk's node:   |
  |    |     for each neighbor N:            |
  |    |       reverseIndex[N].del(chunkId)  |
  |    |                                     |
  |    |-- HEAL BACKWARD EDGES               |
  |    |   for each M in reverseIndex[chunkId|
  |    |     for each layer of M's node:     |
  |    |       removeNeighbor(M, layer,      |
  |    |                      chunkId)       |
  |    |   del reverseIndex[chunkId]         |
  |    |                                     |
  |    |-- REMOVE FROM LSH                   |
  |    |   removeLsh(lsh, fpPtr, chunkId)    |
  |    |     re-hashes bands, removes id     |
  |    |     from each bucket                |
  |    |                                     |
  |    |-- REMOVE FROM LEXICAL               |
  |    |   removeMemory(lexical, chunkId,    |
  |    |               chunkText)            |
  |    |     decrements postings, corpus TF  |
  |    |                                     |
  |    |-- FREE ID                           |
  |    |   freeId(chunkId)                   |
  |    |     push onto freelist (linked list |
  |    |     stored in fingerprint bytes 0-7)|
  |    |     zero graph node                 |
  |    |                                     |
  |    +-- DELETE MAPPING                    |
  |       del chunkToParent[chunkId]         |
  +--------------------+---------------------+
                       |
                       ▼
  +------------------------------------------+
  | 3. CLEAN PARENT                          |
  |    del textCache[id]                     |
  |    del timestampCache[id]                |
  |    syncHeader() --> write recordCount +  |
  |                     freelistHead to disk |
  +--------------------+---------------------+
                       |
                       ▼
              OUTPUT: id (now reusable)
```

**Storage:** Flat memory-mapped files with 256-byte headers (`ADLN` magic). Slots are dense indices (0, 1, 2…) rather than byte offsets. Pre-allocated 64 MiB chunks minimize remap frequency.

| File | Purpose |
|------|---------|
| `wal.bin` | Append-only text + metadata |
| `fingerprints.bin` | Fingerprint store (header + 1280-byte slots) |
| `graph.bin` | HNSW node store (header + 2056-byte slots) |
| `chunks.bin` | Parent→chunk mappings |
| `lsh.bin` | Persisted LSH index (checkpoint) |
| `lexical.bin` | Persisted lexical postings (checkpoint) |
| `corpus.bin` | Persisted corpus stats (checkpoint) |

---

## Quick start

```bash
# Build
nimble build_release

# Insert
./adaline insert "The quick brown fox"
./adaline insert "Nim is a systems programming language"

# Update / Delete
./adaline update 0 "Updated text"
./adaline delete 1

# Search
./adaline search "quick fox" 5

# Stats
./adaline stats

# Tests
nimble test

# Benchmarks
nimble benchmark
./benchmarks/beir scifact
./benchmarks/beir nfcorpus
./benchmarks/longmemeval
./benchmarks/crud scifact 1000 1000
```

---

## Benchmarks

Apple MacBook Air M2 (16 GB). Full tables and methodology in [`BENCHMARK.md`](BENCHMARK.md).

| Dataset | Corpus | Indexing | Query (top-100) | nDCG@10 | R@5 |
|---------|--------|----------|-----------------|---------|-----|
| SciFact | 5,183 docs | 2,350 docs/s | 223 q/s | 0.557 | — |
| NFCorpus | 3,633 docs | 2,175 docs/s | 361 q/s | 0.277 | — |
| LongMemEval-S | 500 questions | — | — | — | 94.6% |

SciFact: Recall@1 = 43.3%, MRR = 0.54. P50 latency ~4.4 ms for top-100.
NFCorpus: Precision@1 = 38.7%, MRR = 0.47. Hard medical retrieval task with sparse labels.
LongMemEval-S: R@1 = 78.2%, R@5 = 94.6%. Conversational memory retrieval.

---

## Configuration

`domain/entities/config.nim`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fingerprintBytes` | 1280 | Fingerprint size (10240 bits) |
| `tokenWeight` | 0.50 | Jaccard weight for token block |
| `bigramWeight` | 0.25 | Jaccard weight for char-bigram block |
| `contextWeight` | 0.25 | Jaccard weight for XOR-context block |
| `lshBands` / `lshRows` | 50 / 2 | Fingerprint LSH banding |
| `hnswMaxLayers` | 8 | HNSW layer count |
| `hnswMaxNeighbors` | 32 | Max edges per layer |
| `hnswEfConstruction` | 200 | HNSW build beam width |
| `hnswEfSearch` | 64 | HNSW query beam width |
| `dirichletMu` | 2000.0 | QLM smoothing parameter |
| `rrfK` | 60 | RRF constant |
| `rerankCoverageWeight` | 0.5 | Term-coverage boost weight |
| `chunkSaturationThreshold` | 0.6 | Chunk when any block exceeds this saturation |

---

## Design

Dependency flow: `Use Cases ← Domain ← Infrastructure`. Domain services call into `infrastructure/` directly. Not Clean Architecture — simplicity and performance over strict layers.

**Delete / Update:** `deleteMemory()` removes from LSH, lexical, and HNSW. An in-memory reverse edge index heals HNSW neighbor lists so no orphaned edges remain. The slot goes to a freelist for reuse. `updateMemory()` deletes old chunks and inserts new ones, preserving the parent ID. Both are WAL-logged; restarts replay the log and rebuild all in-memory indexes. `checkpoint()` persists indexes to skip WAL replay on startup.

**Chunking:** Long documents are split into sentence-aware chunks with one-sentence overlap. Each chunk gets its own fingerprint. Prevents saturation and keeps fingerprints sparse. Short documents stay single-chunk.

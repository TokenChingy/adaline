# Adaline

A Nim vector search engine that turns text into **10240-bit sparse fingerprints** and searches them with HNSW + LSH, plus a lexical index for token-based matching. Results are fused with Reciprocal Rank Fusion and reranked by term coverage.

---

## How it works

Adaline uses a **dual-lane retrieval architecture**: every document is indexed simultaneously into a **semantic lane** (sparse fingerprint + HNSW graph + LSH seeds) and a **lexical lane** (inverted index with QLM scoring). At query time both lanes run in parallel; their results are fused with Reciprocal Rank Fusion and reranked by term coverage.

### Flow overview

All storage is slot-addressed memory-mapped flat files with 256-byte self-describing headers (`ADLN` magic). Fingerprint slots are 1280 bytes; graph nodes are 2056 bytes. Files grow in 64 MiB pre-allocated chunks to minimize `mmap` remaps. IDs are dense integers starting at 0; deleted slots are pushed onto a freelist implemented as a linked list in the first 8 bytes of each freed fingerprint slot.

**Insert.** `allocId()` hands out a slot from the freelist or appends a new one. The WAL record is `(memoryId: uint64, timestamp: uint64, textLen: uint32, text)` — append-only, flushed immediately. The text is then conditionally chunked: the chunker estimates saturation for each of the three fingerprint blocks (token, character-bigram, XOR-context) against `chunkSaturationThreshold` (default 0.6). If any block would exceed 60 % saturation, the text is split on sentence boundaries (`., !, ?`) with one-sentence overlap between chunks.

Each chunk is encoded into a 10240-bit sparse fingerprint. The encoder partitions the bitmap into three regions: 4096 bits for tokens, 3072 bits for character bigrams, and 3072 bits for XOR-bounded context. Tokens and bigrams are hashed into bit positions with a custom 64-bit hash; multiple probes per feature use different seeds so a single token can set more than one bit. Prefix/suffix 4-grams are also encoded for morphological robustness. Token-bigram co-occurrences are encoded in the token block. The context block binds each token to its left and right neighbor via `hash(token) xor hash(neighbor)`, producing order-aware bits without fixed windows. For queries, `queryProbeMultiplier = 2.0` makes the query fingerprint roughly twice as dense as insert fingerprints, compensating for the sparsity of short queries.

Probe counts are scaled by IDF squared via the corpus index: rare tokens get more probes, common tokens get fewer. This keeps frequent stopwords from saturating the bitmap while concentrating representational budget on discriminative terms.

Each chunk fingerprint is written to the fingerprint store, then indexed three ways:
- **LSH** — GoldFinger-style direct banding: the fingerprint's 160 uint64 segments are partitioned into 50 bands of 2 rows each. A band hash XORs its segments with two mixing constants. Two similar fingerprints share many identical segments, so they collide in the same LSH buckets with high probability.
- **HNSW** — Layer assignment uses an exponential distribution (`rand < 0.5` per layer, max 8 layers). Insertion descends from the entry point: greedy best-first search (`ef=1`) from the top layer down to the target layer, then a wider beam search (`efConstruction=200`) at the insertion layer to select up to 32 neighbors. Edges are bidirectional: when node A adds B, B also adds A (subject to B's 32-neighbor limit, pruned by weighted Jaccard distance). A reverse edge index (`node → predecessors`) is maintained in memory so deletions can heal the graph without a full rebuild.
- **Lexical** — Text is tokenized on non-alphanumeric boundaries. The index stores postings `(memoryId, freq)` per term, plus per-document lengths and corpus term frequencies. Scoring uses Query Likelihood with Dirichlet smoothing (`mu = 2000`): each matching term contributes `ln(1 + freq / (μ * pqc))`, where `pqc` is the term's corpus frequency ratio. A length-normalization term is added per document.

Chunk-to-parent mappings are appended to `chunks.bin` as `(parentId, chunkId)` pairs and kept in an in-memory table.

**Search.** The query string is encoded with the denser query probes. The semantic lane has two sub-lanes: (1) LSH seeds — all 50 bands of the query fingerprint are hashed and colliding IDs are collected and deduplicated; (2) HNSW descent — starting from the global entry point, greedy best-first search drops one layer at a time with `ef=1` until layer 0, where a beam search with `efSearch=64` explores the neighborhood. Both sub-lanes score candidates with weighted Jaccard: 50 % token block, 25 % bigram block, 25 % context block. Distance is `1.0 - jaccard`.

The lexical lane runs independently: query tokens are looked up in the inverted index and scored with the same QLM formula. Results from both lanes are merged with Reciprocal Rank Fusion (`rrfK = 10`). Memories present in both lanes receive a score boost. After fusion, chunk IDs are resolved to parent IDs via `chunkToParent`; duplicates are collapsed, keeping the highest score per parent. The top-k parents are then reranked by term-coverage boost: `coverage = |query_tokens ∩ doc_tokens| / |query_tokens|`, and `score += 0.5 * coverage`. This pushes exact-match and high-coverage memories to the top without discarding the semantic signal.

**Update.** `updateMemory(id, content)` performs a logical atomic delete-then-insert while preserving the parent ID. It calls `deleteMemory(id)` to remove all old chunks from every index (healing HNSW edges, reclaiming slots), then appends a new WAL record and re-inserts the content as if it were new, mapping fresh chunks back to the same parent ID.

**Delete.** `deleteMemory(id)` collects every chunk belonging to the parent. For each chunk, it heals the HNSW graph in both directions: forward edges (removing the chunk from its neighbors' reverse-index entries) and backward edges (removing the chunk from each predecessor's neighbor list using the reverse index). The chunk is then removed from LSH buckets (re-hashing its fingerprint bands), from lexical postings (decrementing term frequencies and corpus counts), and from the chunk mapping table. Finally `freeId(chunkId)` pushes the slot onto the freelist by writing the current freelist head into the first 8 bytes of the fingerprint slot and zeroing the graph node. The header is synced to disk so the freelist survives restarts.

**Checkpoint & restart.** `checkpoint()` serializes the in-memory LSH buckets, lexical postings, and corpus IDF tables to `lsh.bin`, `lexical.bin`, and `corpus.bin`, embedding the current WAL offset. On restart, the engine loads the persisted indexes, replays only the WAL entries after the checkpoint offset, and rebuilds the HNSW reverse edge index by scanning all graph nodes. This turns a full WAL replay into a partial one.

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
  |    score = 1/(rrfK+rank_sem)             |
  |          + 1/(rrfK+rank_lex)             |
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
./benchmarks/beir arguana
./benchmarks/longmemeval
./benchmarks/crud scifact 1000 1000
```

---

## Benchmarks

Apple MacBook Air M2 (16 GB). Full tables and methodology in [`BENCHMARK.md`](BENCHMARK.md).

| Dataset | Corpus | Indexing | Query (top-100) | nDCG@10 | R@5 |
|---------|--------|----------|-----------------|---------|-----|
| SciFact | 5,183 docs | 2,465 docs/s | 222 q/s | 0.579 | — |
| NFCorpus | 3,633 docs | 2,445 docs/s | 372 q/s | 0.279 | — |
| ArguAna | 8,674 docs | 3,724 docs/s | 43 q/s | 0.226 | — |
| LongMemEval-S | 500 questions | — | — | — | 93.6% |

SciFact: Recall@1 = 43.5%, MRR = 0.55. P50 latency ~4.5 ms for top-100.
NFCorpus: Precision@1 = 38.7%, MRR = 0.47. Hard medical retrieval task with sparse labels.
ArguAna: Recall@1 = 0%, MRR = 0.16. Adversarial counter-argument retrieval; semantic similarity cannot distinguish opposing stances.
LongMemEval-S: R@1 = 76.8%, R@5 = 93.6%. Conversational memory retrieval.

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
| `rrfK` | 10 | RRF constant |
| `rerankCoverageWeight` | 0.5 | Term-coverage boost weight |
| `chunkSaturationThreshold` | 0.6 | Chunk when any block exceeds this saturation |

---



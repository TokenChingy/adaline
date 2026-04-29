# Adaline

A Nim-based vector search engine that converts text into **10,240-bit sparse fingerprints**. It utilizes a dual-lane retrieval architecture, running semantic search (LSH) and lexical matching (Inverted Index + QLM) in parallel. Results are fused via score-aware Reciprocal Rank Fusion (RRF-S) and reranked by a phrase-aware lexical model.

---

## Core Architecture 

Adaline bypasses traditional database overhead by operating entirely on memory-mapped files. 

* **Zero-Copy Storage:** Data lives in memory-mapped flat files with 256-byte self-describing headers. `fingerprints.bin` holds append-only variable-length compressed records; `fingerprints.idx` maps dense slot IDs to byte offsets. Files grow in pre-allocated 64 MiB chunks to minimize `mmap` remaps.
* **Efficient Memory Management:** IDs are dense integers starting at 0. Deleted slots are pushed to a freelist stored in the `fingerprints.idx` offset field.
* **The Dual-Lane Highway:** Every document is indexed into both a Semantic Lane (for fuzzy, conceptual matching) and a Lexical Lane (for precise, token-based matching).

---

## System Workflows

### 1. Insert & Indexing (Text)

```
  Text Input
      |
      v
  +--------+     +------------------+
  | allocId|---->| wal.bin (append) |
  +--------+     +------------------+
      |
      v
  +------------------+
  | Update caches    |
  | (text, tokens,   |
  |  timestamp)      |
  +------------------+
      |
      v
  +------------------+
  | Corpus Index     |
  | (IDF tracking)   |
  +------------------+
      |
      v
  +------------------+
  | Dynamic Chunking |
  +------------------+
      |
      v
  +------------------+
  | SDR Encoder      |----> 10,240-bit fingerprint
  | (tokens/bigrams/ |      per chunk
  |  XOR-context)    |
  +------------------+
      |
      v
  +------------------+     +------------------+
  | Per chunk:       |---->| chunks.bin       |
  | writeFingerprint |     | (parent->chunk)  |
  +------------------+     +------------------+
      |
      +------------------+
      |                  |
      v                  v
  +--------+      +-------------+
  | LSH    |      | Lexical     |
  | Insert |      | Index       |
  +--------+      +-------------+
```

1. **Allocate & Log:** `allocId()` fetches a slot from the freelist. The record `(memoryId, timestamp, textLen, text)` is appended to `wal.bin` and flushed instantly.
2. **Update Caches:** The raw text, token set, and timestamp are stored in in-memory tables (`textCache`, `tokenCache`, `timestampCache`).
3. **Corpus Index:** The text is tokenized and its document frequencies are added to the global corpus index, which drives IDF-squared probe scaling.
4. **Dynamic Chunking:** The engine estimates saturation for the token, character-bigram, and XOR-context blocks. If any block hits the `chunkSaturationThreshold` (60%), the text is split on sentence boundaries with a one-sentence overlap.
5. **Encoding:** Each chunk is encoded into a 10,240-bit fingerprint. Probe counts are scaled by IDF-squared (rare tokens get more probes; stopwords get fewer).
6. **Store & Map:** Each chunk is compressed and written to `fingerprints.bin` at a new byte offset (recorded in `fingerprints.idx`), and its parent→chunk linkage is appended to `chunks.bin`.
7. **LSH Routing (Semantic):** The fingerprint is partitioned into 80 bands of 2 rows for GoldFinger-style direct banding, inserting the chunk into the in-memory LSH buckets.
8. **Lexical Routing:** Text is tokenized on non-alphanumeric boundaries and added to the inverted index using Query Likelihood with Dirichlet smoothing ($\mu = 2000$).

### 2. Search & Retrieval (Text)

```
  Query Text
      |
      v
  +------------------+
  | SDR Encoder      |
  | (probe mul 2.0)  |
  +------------------+
      |
      +----------+----------+
      |                     |
      v                     v
  +--------+           +--------+
  | LSH    |           | Lexical|
  | Seeds  |           | Index  |
  +--------+           +--------+
      |                     |
      v                     v
  +----------------+   +--------+
  | readFingerprint|   | QLM    |
  | + Weighted     |   | Score  |
  |   Jaccard      |   +--------+
  +----------------+        |
      |                     |
      +----------+----------+
                 |
                 v
          +------------+
          | RRF-S Merger |
          | (k=10)       |
          +------------+
                 |
                 v
          +------------+
          | Chunk ->   |
          | Parent     |
          | Resolution |
          +------------+
                 |
                 v
          +------------+
          | Phrase-Aw. |
          | Rerank     |
          +------------+
                 |
                 v
           Results[]
```

1. **Query Encoding:** The query is encoded with a `queryProbeMultiplier` of 2.0 to compensate for the sparsity of short queries.
2. **Semantic Lane:** The 80 LSH bands are hashed to collect candidate seeds. For each candidate, the stored fingerprint is read from `fingerprints.bin` and scored using brute-force Weighted Jaccard.
3. **Lexical Lane:** Query tokens are routed through the inverted index and scored via QLM.
4. **Fusion & Resolution:** Both lanes are merged using score-aware RRF-S (`rrfK = 10`). Chunk IDs are resolved back to their parent IDs, keeping only the highest-scoring chunk per parent.
5. **Rerank:** Top-30 candidates receive a final phrase-aware boost blending IDF-weighted coverage, exact substring phrase matching, and soft length normalization to push precise lexical matches to the top.

### 3. Update (Text)

```
  Update Request (id, newText)
      |
      v
  +--------+     +------------------+
  | Delete |---->| Heal old chunks  |
  | old    |     | (LSH + lexical   |
  | chunks |     |  purge)          |
  +--------+     +------------------+
      |
      v
  +--------+     +------------------+
  | Insert |---->| WAL append +     |
  | new    |     | full re-index    |
  | chunks |     | (same as Insert) |
  +--------+     +------------------+
      |
      v
  Same parent ID, fresh chunks
```

Performs a delete-then-insert. It calls the Delete workflow to wipe the old chunks from LSH and lexical indexes, then appends a new WAL record and re-inserts the updated content, seamlessly mapping the fresh chunks back to the original parent ID.

### 4. Delete (Text)

```
  Delete Request (parentId)
      |
      v
  +------------------+
  | Collect chunkIds |
  | from chunkToParent|
  +------------------+
      |
      v
  +------------------+
  | Purge Indexes    |----> Remove from LSH buckets
  +------------------+      Decrement lexical postings
      |
      v
  +------------------+
  | freeId()         |----> Mark idx entry free
  +------------------+      Push to freelist
      |
      v
  +------------------+
  | Delete caches    |
  | (text, tokens,   |
  |  timestamp)      |
  +------------------+
      |
      v
   syncHeader()
```

1. **Collect:** Identify all chunk IDs associated with the target parent ID from the in-memory `chunkToParent` table.
2. **Purge Indexes:** For each chunk, remove it from the LSH buckets and decrement its term frequencies from the lexical postings.
3. **Free ID:** Mark the slot as free in `fingerprints.idx` and push it to the freelist.
4. **Delete Caches:** Remove the parent entry from `textCache`, `tokenCache`, and `timestampCache`.
5. **Sync Header:** Persist the updated record count and freelist state to `fingerprints.idx`.

### 5. Vision / Dense-Vector Insert

```
  Float32 Vector (e.g. 1280-dim CNN features)
      |
      v
  +------------------+
  | allocId()        |
  +------------------+
      |
      v
  +------------------+
  | Dense Encoder    |----> Top-k dims by |value|
  | (k-WTA, 128      |      probed via probeBlock
  |  winners)        |
  +------------------+
      |
      v
  +------------------+
  | 10,240-bit       |      Same fingerprint format
  | Fingerprint      |      as text SDR
  +------------------+
      |
      +----------+----------+
      |                     |
      v                     v
  +--------+           +-------------+  (no lexical
  | LSH    |           | Fingerprint |   index for
  | Insert |           | Store       |   dense vectors)
  +--------+           +-------------+
      |
      v
  lsh.bin
```

Bypasses text chunking, SDR encoding, and lexical indexing. The `insertDense` use-case allocates an ID, encodes a float32 vector directly to a fingerprint, writes it to the fingerprint store, and inserts into LSH.

### 6. Vision / Dense-Vector Search

```
  Query Vector (float32)
      |
      v
  +------------------+
  | Dense Encoder    |
  | (same k-WTA)     |
  +------------------+
      |
      v
  +------------------+
  | 10,240-bit       |
  | Query Fingerprint|
  +------------------+
      |
      v
  +--------+
  | LSH    |
  | Seeds  |
  +--------+
      |
      v
  +----------------+
  | readFingerprint|
  | + Weighted     |
  |   Jaccard      |
  +----------------+
      |
      v
  +------------------------+
  | Top-k Results          |
  +------------------------+
```

The `searchDense` use-case encodes the query vector to a fingerprint, queries LSH for candidates, reads each candidate's stored fingerprint, and scores via brute-force Weighted Jaccard. No lexical lane, no RRF, no reranker — pure semantic similarity.

### 7. Checkpoint & Restart

```
  checkpoint()
      |
      +----------+----------+----------+
      |          |          |          |
      v          v          v          v
  +--------+ +--------+ +--------+ +--------+
  | LSH    | | Lexical | | Corpus| | WAL   |
  | Buckets| | Postings| | IDF   | | Offset |
  +--------+ +--------+ +--------+ +--------+
      |          |          |          |
      v          v          v          v
  lsh.bin   lexical.bin  corpus.bin  (embedded
                                         in
                                       each)
```

`checkpoint()` serializes the in-memory LSH buckets, lexical postings, and corpus IDF tables to disk, embedding the current WAL offset in the header of each persisted file. On restart, Adaline loads the persisted indexes and replays only the un-checkpointed WAL tail.

---

## Storage Map

| File | Purpose |
|------|---------|
| `wal.bin` | Append-only text + metadata |
| `fingerprints.bin` | Fingerprint store (header + compressed bitmaps) |
| `fingerprints.idx` | Slot index (offset + compressed size per ID) |
| `chunks.bin` | Parent-to-chunk mappings |
| `lsh.bin` | Persisted LSH index (checkpoint) |
| `lexical.bin` | Persisted lexical postings (checkpoint) |
| `corpus.bin` | Persisted corpus stats (checkpoint) |

---

## Quick Start

```bash
# Build CLI
nimble release

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

# Compile benchmarks
nimble benchmark

# BEIR
./benchmarks/beir scifact
./benchmarks/beir nfcorpus
./benchmarks/beir arguana

# Vision / Dense-Vector (requires PyTorch)
python3 benchmarks/benchmark.py --dataset cifar10 --benchmark all

# With compiled Adaline Engine (nimble python required)
python3 benchmarks/benchmark.py --dataset cifar10 --benchmark classify --engine

# Nim benchmark using insertDense / searchDense use-cases
python3 benchmarks/dump_features.py
nim c -d:release -o:benchmarks/vision_bench benchmarks/vision_bench.nim
./benchmarks/vision_bench benchmarks/data/cifar10_features.bin

# CRUD throughput
nim c -d:release -o:benchmarks/crud benchmarks/crud.nim
./benchmarks/crud scifact 1000 1000

# LongMemEval (requires dataset download — see BENCHMARK.md)
nim c -d:release -o:benchmarks/longmemeval benchmarks/longmemeval.nim
./benchmarks/longmemeval
```

---

## Benchmarks

*Tested on Apple MacBook Air M2 (16 GB). Full methodology in `BENCHMARK.md`.*

| Dataset | Corpus | Indexing | Query (top-100) | nDCG@10 | R@5 |
|---------|--------|----------|-----------------|---------|-----|
| SciFact | 5,183 docs | 4,669 docs/s | 273 q/s | 0.654 | 72.1% |
| NFCorpus | 3,633 docs | 4,646 docs/s | 460 q/s | 0.286 | 11.5% |
| ArguAna | 8,674 docs | 5,216 docs/s | 95 q/s | 0.337 | 50.0% |
| FIQA | 57K docs | 5,677 docs/s | 14.6 q/s | 0.168 | — |
| LongMemEval-S | 500 questions | — | — | — | 94.8% |
| Vision (CIFAR-10 / MobileNetV2) | 200 vectors | — | 0.22 ms/q | — | — |

**Performance Notes:**
* **SciFact:** Recall@1 = 52.2%, R@100 = 87.7%, MRR = 0.62. P50 latency ~3.7 ms for top-100.
* **NFCorpus:** Recall@1 = 5.2%, Precision@1 = 38.4%, MRR = 0.49. P50 latency ~2.0 ms.
* **ArguAna:** Recall@1 = 0%, R@100 = 97.3%, MRR = 0.23. P50 latency ~9.3 ms. (Adversarial counter-argument retrieval; lexical lane carries most signal).
* **FIQA:** 57K financial QA pairs. Query latency ~66 ms for top-100.
* **LongMemEval-S:** R@1 = 83.6%, R@5 = 94.8%. (Conversational memory retrieval).
* **Vision (CIFAR-10 / MobileNetV2):** 1-shot classification 44.4% dense vs 42.2% sparse. 20-shot converges to 62.6% dense vs 61.8% sparse. Open-set AUROC 0.578.

---

## Configuration (`domain/entities/config.nim`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fingerprintBytes` | 1280 | Fingerprint size (10,240 bits) |
| `tokenWeight` | 0.50 | Jaccard weight for token block |
| `bigramWeight` | 0.25 | Jaccard weight for char-bigram block |
| `contextWeight` | 0.25 | Jaccard weight for XOR-context block |
| `lshBands` / `lshRows` | 80 / 2 | Fingerprint LSH banding (full 160-segment coverage) |
| `dirichletMu` | 2000.0 | QLM smoothing parameter |
| `rrfK` | 10 | Score-aware RRF-S fusion constant |
| `rerankIdfWeight` | 0.30 | IDF-weighted coverage boost weight |
| `rerankPhraseWeight` | 0.20 | Exact phrase match boost weight |
| `rerankLenWeight` | 0.05 | Length normalization boost weight |
| `chunkSaturationThreshold` | 0.6 | Chunk trigger limit for block saturation |
| `tokenProbes` | 3 | Base probes per token (reduced for sparsity) |
| `tokenBigramProbes` | 2 | Base probes per adjacent token bigram |
| `bigramProbes` | 1 | Base probes per char-bigram |
| `contextProbes` | 1 | Base probes per XOR-context feature |
| `semanticSearchEnabled` | true | Enable LSH semantic lane |
| `lexicalSearchEnabled` | true | Enable lexical QLM lane |
| `maxTokenFeatures` | 12 | Top-K IDF tokens kept per document (0 = disable) |
| `queryProbeMultiplier` | 2.0 | Extra probe multiplier for short queries |

---

## Dense-Vector Pipeline

Adaline can index float32 vectors (e.g. CNN embeddings, tabular features) through a parallel API that bypasses text chunking and lexical indexing. The pipeline uses the same 10,240-bit fingerprint format and LSH bands as text — only the encoder changes.

### k-WTA Binarization

`domain/algorithms/dense_encoder.nim` implements **k-Winners Take All** encoding:

1. Input: an L2-normalised float32 vector.
2. Select the top-k dimensions by absolute value (default k = 128).
3. Hash each winning dimension via `hashFeature` to generate probes.
4. Flip the corresponding bits in a 10,240-bit fingerprint using the same `probeBlock` primitive as the text SDR encoder.

The resulting fingerprint is structurally identical to a text fingerprint and is stored in the same `fingerprints.bin` format.

### API

```nim
# Nim
let id = insertDense(service, InsertDenseInput(vec: features))
let results = searchDense(service, SearchDenseInput(vec: queryFeatures, topK: 10))
deleteDense(service, DeleteDenseInput(memoryId: id))
```

```python
# Python (nimble python)
id = eng.insertDense(features)
results = eng.searchDense(query_features, topK=10)
eng.deleteDense(id)
```

* Dense-vector insert populates **only** LSH — no WAL entry, no lexical posting, no chunk mapping.
* Dense-vector search queries **only** LSH — no lexical lane, no RRF, no reranker.
* Dense-vector delete removes the slot from LSH, same as text delete.

There is no schema flag or runtime dispatch. Text and dense vectors are managed through separate, explicit APIs (`insert` vs `insertDense`). They share the underlying storage but are not interchangeable at the entity level.

### Vision Benchmark

`benchmarks/vision_bench.nim` runs CIFAR-10 classification using MobileNetV2 features extracted via `benchmarks/dump_features.py`:

| Task | Dense Cosine | Sparse LSH |
|------|-------------|-------------|
| 1-shot classify | 44.4% | 42.2% |
| 20-shot classify | 62.6% | 61.8% |
| Open-set AUROC | — | 0.578 |
| Query latency | — | ~0.22 ms |

The sparse pipeline is competitive with dense cosine similarity while operating on 1,280-byte fingerprints instead of full float32 vectors.

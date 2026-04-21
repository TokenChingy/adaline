# Adaline

A Nim-based vector search engine that converts text into **10,240-bit sparse fingerprints**. It utilizes a dual-lane retrieval architecture, running semantic search (HNSW + LSH) and lexical matching (Inverted Index + QLM) in parallel. Results are fused via Reciprocal Rank Fusion (RRF) and reranked by term coverage.

---

## Core Architecture 

Adaline bypasses traditional database overhead by operating entirely on memory-mapped files. 

* **Zero-Copy Storage:** All data lives in slot-addressed, memory-mapped flat files (`.bin`) with 256-byte self-describing headers. Files grow in pre-allocated 64 MiB chunks to minimize `mmap` remaps.
* **Efficient Memory Management:** IDs are dense integers starting at 0. Deleted slots are pushed to a freelist stored directly in the first 8 bytes of freed fingerprint slots.
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
  | Dynamic Chunking |----> chunks.bin (parent->chunk mapping)
  +------------------+
      |
      v
  +------------------+
  | SDR Encoder      |----> 10,240-bit fingerprint
  | (tokens/bigrams/ |      per chunk
  |  XOR-context)    |
  +------------------+
      |
      +------------------+------------------+
      |                  |                  |
      v                  v                  v
  +--------+      +------------+      +-------------+
  | LSH    |      | HNSW Graph |      | Lexical     |
  | (80    |      | (layered   |      | Index (QLM  |
  |  bands)|      |  greedy)   |      | + Dirichlet)|
  +--------+      +------------+      +-------------+
      |                  |                  |
      v                  v                  v
  lsh.bin          graph.bin          lexical.bin
```

1. **Allocate & Log:** `allocId()` fetches a slot from the freelist. The record `(memoryId, timestamp, textLen, text)` is appended to `wal.bin` and flushed instantly.
2. **Dynamic Chunking:** The engine estimates saturation for the token, character-bigram, and XOR-context blocks. If any block hits the `chunkSaturationThreshold` (60%), the text is split on sentence boundaries with a one-sentence overlap.
3. **Encoding:** Each chunk is encoded into a 10,240-bit fingerprint. Probe counts are scaled by IDF-squared (rare tokens get more probes; stopwords get fewer).
4. **LSH Routing (Semantic):** The fingerprint is partitioned into 80 bands of 2 rows for GoldFinger-style direct banding, inserting the chunk into the `lsh.bin` buckets.
5. **HNSW Routing (Semantic):** The chunk descends the `graph.bin` layers via greedy best-first search, wiring bidirectional edges pruned by weighted Jaccard distance.
6. **Lexical Routing:** Text is tokenized on non-alphanumeric boundaries and added to the inverted index using Query Likelihood with Dirichlet smoothing ($\mu = 2000$).

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
      |          |          |
      v          v          v
  +--------+ +--------+ +--------+
  | LSH    | | HNSW   | | Lexical|
  | Seeds  | | Layer0 | | Index  |
  +--------+ +--------+ +--------+
      |          |          |
      v          v          v
  +--------+ +--------+ +--------+
  | Weighted| | Beam   | | QLM    |
  | Jaccard | | Search | | Score  |
  +--------+ +--------+ +--------+
      |          |          |
      +----------+----------+
                 |
                 v
          +------------+
          | RRF Merger |
          | (k=10)     |
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
          | Term-Cov   |
          | Rerank     |
          +------------+
                 |
                 v
           Results[]
```

1. **Query Encoding:** The query is encoded with a `queryProbeMultiplier` of 2.0 to compensate for the sparsity of short queries.
2. **Semantic Lane:** The 80 LSH bands are hashed to collect candidate seeds. These seeds are dropped into Layer 0 of the HNSW graph, triggering a beam search (`efSearch=64`). Candidates are scored using Weighted Jaccard.
3. **Lexical Lane:** Query tokens are concurrently routed through the inverted index and scored via QLM.
4. **Fusion & Resolution:** Both lanes are merged using Reciprocal Rank Fusion (`rrfK = 10`). Chunk IDs are resolved back to their parent IDs, keeping only the highest-scoring chunk per parent.
5. **Rerank:** Top candidates receive a final term-coverage boost (`+ 0.5 * coverage`) to push exact matches to the top of the result list.

### 3. Update (Text)

```
  Update Request (id, newText)
      |
      v
  +--------+     +------------------+
  | Delete |---->| Heal old chunks  |
  | old    |     | (HNSW + LSH +    |
  | chunks |     |  lexical purge)  |
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

Performs a logical atomic delete-then-insert. It calls the Delete workflow to wipe the old chunks and heal the graph, then appends a new WAL record and re-inserts the updated content, seamlessly mapping the fresh chunks back to the original parent ID.

### 4. Delete (Text)

```
  Delete Request (parentId)
      |
      v
  +------------------+
  | Collect chunkIds |
  | from chunks.bin  |
  +------------------+
      |
      v
  +------------------+
  | Heal Graph       |----> Reverse edge index
  | (sever fwd/bwd)  |      heals neighbor lists
  +------------------+
      |
      v
  +------------------+
  | Purge Indexes    |----> Remove from LSH buckets
  +------------------+      Decrement lexical postings
      |
      v
  +------------------+
  | freeId()         |----> Zero fingerprint
  +------------------+      Clear HNSW node
      |                     Push to freelist
      v
   syncHeader()
```

1. **Collect:** Identify all chunk IDs associated with the target parent ID.
2. **Heal Graph:** For each chunk, sever forward edges and backward edges. An in-memory reverse edge index is used to remove the chunk from all predecessors' neighbor lists without a full graph rebuild.
3. **Purge Indexes:** Remove the chunk from the LSH buckets and decrement its term frequencies from the lexical postings.
4. **Free ID:** Zero out the graph node and push the slot back to the `mmap` freelist.

### 5. Vision / Dense-Vector Insert

```
  Float32 Vector (e.g. 1280-dim CNN features)
      |
      v
  +------------------+
  | Dense Encoder    |
  | (k-WTA, 128      |----> Top-k dims by |value|
  |  winners)        |      probed via hashFeature
  +------------------+
      |
      v
  +------------------+
  | 10,240-bit       |      Same fingerprint format
  | Fingerprint      |      as text SDR
  +------------------+
      |
      +----------+----------+
      |          |          |
      v          v          v
  +--------+ +--------+  (no lexical
  | LSH    | | HNSW   |   index for
  | Insert | | Insert |   dense vectors)
  +--------+ +--------+
      |          |
      v          v
  lsh.bin   graph.bin
```

Bypasses text chunking, SDR encoding, and lexical indexing. The `insertDense` use-case encodes a float32 vector directly to a fingerprint and inserts into LSH and HNSW.

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
  +--------+     +--------+
  | LSH    |---->| HNSW   |
  | Seeds  |     | Layer0 |
  +--------+     +--------+
      |              |
      v              v
  +------------------------+
  | Weighted Jaccard Score |
  +------------------------+
      |
      v
   Results[]
```

The `searchDense` use-case encodes the query vector to a fingerprint and searches LSH + HNSW. No lexical lane, no RRF, no reranker — pure semantic similarity.

### 7. Checkpoint & Restart

```
  checkpoint()
      |
      +----------+----------+----------+
      |          |          |          |
      v          v          v          v
  +--------+ +--------+ +--------+ +--------+
  | LSH    | | Lexical| | Corpus | | WAL    |
  | Buckets| | Postings| | IDF   | | Offset |
  +--------+ +--------+ +--------+ +--------+
      |          |          |          |
      v          v          v          v
  lsh.bin   lexical.bin  corpus.bin  (header)
```

`checkpoint()` serializes the in-memory LSH buckets, lexical postings, and corpus IDF tables to disk, embedding the current WAL offset. On restart, Adaline loads the persisted indexes, replays only the un-checkpointed WAL tail, and rebuilds the HNSW reverse edge index in memory.

---

## Storage Map

| File | Purpose |
|------|---------|
| `wal.bin` | Append-only text + metadata |
| `fingerprints.bin` | Fingerprint store (header + 1280-byte slots) |
| `graph.bin` | HNSW node store (header + 1032-byte slots) |
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
| SciFact | 5,183 docs | 1,143 docs/s | 260 q/s | 0.567 | 65.7% |
| NFCorpus | 3,633 docs | 1,175 docs/s | 393 q/s | 0.273 | 10.8% |
| ArguAna | 8,674 docs | 1,127 docs/s | 47 q/s | 0.165 | 22.1% |
| LongMemEval-S | 500 questions | — | — | — | 93.6% |
| Vision (CIFAR-10 / MobileNetV2) | 200 vectors | — | 0.22 ms/q | — | — |

**Performance Notes:**
* **SciFact:** Recall@1 = 40.8%, MRR = 0.53. P50 latency ~3.9 ms for top-100.
* **NFCorpus:** Precision@1 = 37.1%, MRR = 0.46. (Hard medical retrieval task with sparse labels).
* **ArguAna:** Recall@1 = 0%, MRR = 0.12. (Adversarial counter-argument retrieval; semantic similarity struggles to distinguish opposing stances).
* **LongMemEval-S:** R@1 = 76.8%, R@5 = 93.6%. (Conversational memory retrieval).
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
| `hnswMaxLayers` | 8 | Maximum HNSW layers |
| `hnswMaxNeighbors` | 16 | Max edges per layer |
| `hnswEfConstruction` | 64 | HNSW build beam width |
| `hnswEfSearch` | 64 | HNSW query beam width |
| `dirichletMu` | 2000.0 | QLM smoothing parameter |
| `rrfK` | 10 | Reciprocal Rank Fusion constant |
| `rerankCoverageWeight` | 0.5 | Term-coverage boost weight |
| `chunkSaturationThreshold` | 0.6 | Chunk trigger limit for block saturation |
| `tokenProbes` | 3 | Base probes per token (reduced for sparsity) |
| `bigramProbes` | 1 | Base probes per char-bigram |
| `contextProbes` | 1 | Base probes per XOR-context feature |

---

## Dense-Vector Pipeline

Adaline can index float32 vectors (e.g. CNN embeddings, tabular features) through a parallel API that bypasses text chunking and lexical indexing. The pipeline uses the same 10,240-bit fingerprint format, LSH bands, and HNSW graph as text — only the encoder changes.

### k-WTA Binarization

`domain/algorithms/dense_encoder.nim` implements **k-Winners Take All** encoding:

1. Input: an L2-normalised float32 vector.
2. Select the top-k dimensions by absolute value (default k = 128).
3. Hash each winning dimension via `hashFeature` to generate probes.
4. Flip the corresponding bits in a 10,240-bit fingerprint using the same `probeBlock` primitive as the text SDR encoder.

The resulting fingerprint is structurally identical to a text fingerprint and is stored in the same `fingerprints.bin` slot format.

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

* Dense-vector insert populates **only** LSH and HNSW — no WAL entry, no lexical posting, no chunk mapping.
* Dense-vector search queries **only** LSH + HNSW — no lexical lane, no RRF, no reranker.
* Dense-vector delete removes the slot from LSH and heals HNSW edges via the reverse index, same as text delete.

There is no schema flag or runtime dispatch. Text and dense vectors are managed through separate, explicit APIs (`insert` vs `insertDense`). They share the underlying storage and graph but are not interchangeable at the entity level.

### Vision Benchmark

`benchmarks/vision_bench.nim` runs CIFAR-10 classification using MobileNetV2 features extracted via `benchmarks/dump_features.py`:

| Task | Dense Cosine | Sparse HNSW |
|------|-------------|-------------|
| 1-shot classify | 44.4% | 42.2% |
| 20-shot classify | 62.6% | 61.8% |
| Open-set AUROC | — | 0.578 |
| Query latency | — | ~0.22 ms |

The sparse pipeline is competitive with dense cosine similarity while operating on 1,280-byte fingerprints instead of full float32 vectors.

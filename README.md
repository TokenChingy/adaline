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

### 1. Insert & Indexing
Takes a text string, writes it to the Write-Ahead Log (WAL), and indexes it across all routing lanes. 

1. **Allocate & Log:** `allocId()` fetches a slot from the freelist. The record `(memoryId, timestamp, textLen, text)` is appended to `wal.bin` and flushed instantly.
2. **Dynamic Chunking:** The engine estimates saturation for the token, character-bigram, and XOR-context blocks. If any block hits the `chunkSaturationThreshold` (60%), the text is split on sentence boundaries with a one-sentence overlap.
3. **Encoding:** Each chunk is encoded into a 10,240-bit fingerprint. Probe counts are scaled by IDF-squared (rare tokens get more probes; stopwords get fewer). 
4. **LSH Routing (Semantic):** The fingerprint is partitioned into 80 bands of 2 rows for GoldFinger-style direct banding, inserting the chunk into the `lsh.bin` buckets.
5. **HNSW Routing (Semantic):** The chunk descends the `graph.bin` layers via greedy best-first search, wiring bidirectional edges pruned by weighted Jaccard distance.
6. **Lexical Routing:** Text is tokenized on non-alphanumeric boundaries and added to the inverted index using Query Likelihood with Dirichlet smoothing ($\mu = 2000$).

### 2. Search & Retrieval
Executes a parallel query across both lanes, merges the candidates, and isolates the highest-relevance parents.

1. **Query Encoding:** The query is encoded with a `queryProbeMultiplier` of 2.0 to compensate for the sparsity of short queries.
2. **Semantic Lane:** The 80 LSH bands are hashed to collect candidate seeds. These seeds are dropped into Layer 0 of the HNSW graph, triggering a beam search (`efSearch=64`). Candidates are scored using Weighted Jaccard.
3. **Lexical Lane:** Query tokens are concurrently routed through the inverted index and scored via QLM.
4. **Fusion & Resolution:** Both lanes are merged using Reciprocal Rank Fusion (`rrfK = 10`). Chunk IDs are resolved back to their parent IDs, keeping only the highest-scoring chunk per parent.
5. **Rerank:** Top candidates receive a final term-coverage boost (`+ 0.5 * coverage`) to push exact matches to the top of the result list.

### 3. Update
Performs a logical atomic delete-then-insert. It calls the Delete workflow to wipe the old chunks and heal the graph, then appends a new WAL record and re-inserts the updated content, seamlessly mapping the fresh chunks back to the original parent ID.

### 4. Delete
Physically removes a memory from every index and heals the surrounding data structures.

1. **Collect:** Identify all chunk IDs associated with the target parent ID.
2. **Heal Graph:** For each chunk, sever forward edges and backward edges. An in-memory reverse edge index is used to remove the chunk from all predecessors' neighbor lists without a full graph rebuild.
3. **Purge Indexes:** Remove the chunk from the LSH buckets and decrement its term frequencies from the lexical postings.
4. **Free ID:** Zero out the graph node and push the slot back to the `mmap` freelist.

### 5. Checkpoint & Restart
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

## Extended Multimodal Architecture

While Adaline is optimized for text retrieval, its underlying storage and indexing are agnostic to data type. At its core, the engine is a memory-mapped database for 10,240-bit sparse arrays. With minimal modifications, it can index and query multimodal data such as images and signals using the same storage layout.

### 1. Universal Payload Header

The 256-byte self-describing header currently tracks text metadata. A single-byte **Schema Flag** (`TEXT`, `VISION`, or `SIGNAL`) is added to dictate which encoding pipeline and retrieval lanes are activated for a given slot.

### 2. Pluggable Binarization Encoders

The text encoder (token, character-bigram, and XOR-context) is replaced by domain-specific binarizers that compress continuous data into Adaline's native 10,240-bit format.

* **Vision:** A dense floating-point vector from an edge model (e.g., MobileNet) is mapped into a sparse fingerprint via **k-Winners Take All (k-WTA)** encoding (`domain/algorithms/dense_encoder.nim`). The top-k dimensions by absolute value are treated as active features and probed into the fingerprint space using the same `hashFeature`/`probeBlock` primitives as the text SDR encoder.
* **Signal:** Frequency data (FFT or spectrogram) is reduced via **Winner-Take-All (KWTA)** thresholding. Only dominant peaks are flipped to active bits, suppressing background noise.

Because fingerprints are flat 1,280-byte slots, image and signal fingerprints occupy the exact same memory-mapped blocks as text. No schema migration is required.

### 3. Lexical Bypass

When a query is flagged as `VISION` or `SIGNAL`, the retrieval highway is altered dynamically:

* The **Lexical Lane** (inverted index and Dirichlet smoothing) is bypassed, as images and signals contain no indexable vocabulary.
* The **Semantic Lane** assumes full control. The fingerprint is sliced into the standard 80 LSH bands. LSH buckets yield candidate seeds, which are dropped into the HNSW graph. The engine executes its standard greedy beam search, scoring similarity with the same hardware-level Jaccard math used for text.

---

### Multimodal Classification and Detection

Storing images and signals in the HNSW graph enables machine learning tasks directly on the index without separate training pipelines.

**Few-Shot Classification**
Adaline uses K-Nearest Neighbor logic on the graph. If a user inserts five encoded images of a new object class, subsequent queries classify live inputs by neighbor majority vote in milliseconds.

**Anomaly Detection**
Because similarity is computed via strict Jaccard (intersection over union) rather than probabilistic scores, the system has a rigid mathematical threshold for familiarity. If the highest Jaccard score of the nearest neighbors falls below a configurable threshold (e.g., 35%), the input is flagged as a **novel anomaly**.

**Sensor Fusion (Cross-Modal Memory)**
Text, images, and signals all resolve to the same 1,280-byte slot format and can coexist in the same memory-mapped files. A single parent ID can map to text (a manual), an image (a machine photo), and a signal (a vibration sample). This allows edge devices to cross-reference multiple data types simultaneously without leaving local storage.

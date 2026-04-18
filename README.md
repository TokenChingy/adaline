# Adaline

A Nim engine for Sparse Distributed Representations (SDR) using memory-mapped flat files, MinHash Local Sensitivity Hash, Hierarchal Navigable Small World Graph Search, and a Lexical Sidecar with Reciprocal Rank Fusion, and Term Coverage Rerank.

## What Adaline Does

Adaline turns text into **Sparse Distributed Representations** — fixed-size 10240-bit bitmaps called **fingerprints**. Two pieces of text with similar meaning will have fingerprints that overlap significantly. This lets you search by semantic similarity, not just exact token matching.

Think of it like this:
- **Traditional search** asks "which memories contain these exact tokens?"
- **Adaline search** asks "which memories have fingerprints that look like this query's fingerprint?"

It combines both approaches to give you the best of each world.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  INSERT: How a memory gets stored                                               │
│                                                                                 │
│  Text ──► Tokenize ──► SDR Encoder ──► 1280-byte Fingerprint                    │
│                            │                                                    │
│                            ├──► MinHash LSH Index  ("Which bucket?")            │
│                            ├──► HNSW Graph         ("Nearest neighbors")        │
│                            ├──► Lexical Index      ("Token → memory list")      │
│                            └──► Corpus Index       ("How rare is each token?")  │
│                                                                                 │
│  Text + ID ──► WAL (Append-only log for crash recovery)                         │
│  Fingerprint ──► fingerprints.bin (Memory-mapped)                               │
│  Graph nodes ──► graph.bin (Memory-mapped)                                      │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│  SEARCH: How a query finds relevant memories                                    │
│                                                                                 │
│  Query ──► Same SDR Encoder ──► Fingerprint                                     │
│       │                                                                         │
│       ├──► SEMANTIC LANE ──────────────────────────────────────────────┐        │
│       │    MinHash LSH  ──►  Seed Candidates                           │        │
│       │         │          │                                           │        │
│       │         └─────────►└──► HNSW Graph Search (Layer-0 descent)    │        │
│       │                            │                                   │        │
│       │                            └──► Top-K by Weighted Jaccard      │        │
│       │                                                                │        │
│       ├──► LEXICAL LANE ───────────────────────────────────────────────┘        │
│       │    Inverted Index Lookup + Query Likelihood Scoring                     │
│       │                                                                         │
│       └──► MERGE: Reciprocal Rank Fusion (RRF) Combines Both Lanes              │
│            └──► RERANK: Boost memories containing all query terms               │
│                 └──► Final ranked results                                       │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Insert Flow (Step-by-Step)

### Step 0: What is a fingerprint?

Before anything else, understand the destination. Every memory becomes a **fingerprint**: a 10240-bit bitmap (1280 bytes) split into three blocks:

| Block | Size | Stores | Why |
|-------|------|--------|-----|
| A | 4096 bits | Tokens + token bigrams | "What tokens are in here?" |
| B | 3072 bits | Character bigrams | "What does it *spell* like?" |
| C | 3072 bits | XOR neighbor context | "What tokens are next to each other?" |

Each feature (token, bigram, etc.) is hashed to one or more bit positions. Two memories sharing many features will have overlapping bits — we measure this overlap with **Jaccard similarity**.

---

### Step 1: Write-Ahead Log (WAL)

The raw text and a `uint64` MemoryID are appended to `wal.bin` as binary:
```
[MemoryID: 8 bytes][textLength: 4 bytes][text: N bytes]
```

**Why:** If the process crashes after writing the WAL but before updating indexes, the WAL can be replayed on restart to rebuild everything. The MemoryID is also the byte offset in the fingerprint store (multiples of 1280), so lookup is O(1).

---

### Step 2: Update the Corpus Index

The memory's unique tokens are counted. From these counts we compute **IDF** (Inverse Memory Frequency):

```
IDF(token) = ln(totalMemories / memoriesContaining(token))
```

**What it means:** A token that appears in every memory has IDF ≈ 0 (useless for distinguishing memories). A token that appears in only one memory has a high IDF (very distinctive).

This IDF table is used in the next step to decide how strongly each token should be represented in the fingerprint.

---

### Step 3: SDR Encoding — Turning Text into a Fingerprint

The text is lowercased and broken into tokens (tokens). Then three parallel encoders run:

#### Block A: Tokens (4096 bits)
Each unique token is hashed to bit positions in the first 4096 bits. But here's the key trick: **rare tokens get more bits; common tokens get fewer**.

This is controlled by `scaledProbes`:
```
probes = baseProbes × (IDF(token) / maxIDF)²
```

For example, with `baseProbes = 4`:
- "the" (appears everywhere, IDF ≈ 0) → 1 probe (clamped minimum)
- "quick" (moderately common) → 1–2 probes
- "adaline" (rare, distinctive) → 4 probes

Token bigrams (adjacent pairs like "quick_brown") are also encoded here with their own `baseProbes = 2`.

**Why scale by IDF?** If "the" and "adaline" both set 4 bits, "the" would drown out meaningful matches because it appears in nearly every memory. Scaling gives rare, informative terms a stronger voice.

#### Block B: Character Bigrams (3072 bits)
A sliding 2-character window runs over the text. Each pair like `qu`, `ui`, `ic` is hashed to positions 4096–7167. Only letter/digit pairs are kept.

**Why:** This captures spelling, morphology, and typo resilience. "color" and "colour" share many character bigrams even though their tokens differ.

#### Block C: XOR Context (3072 bits)
For each token, its hash is XOR'd with its left neighbor's hash and its right neighbor's hash. These XOR values are hashed to positions 7168–10239.

**Why:** This captures local token order. "bank river" and "river bank" have identical token sets but different contexts. The XOR binds each token to its neighbors, distinguishing token order and collocations.

---

### Step 4: Write Fingerprint to Storage

The 1280-byte fingerprint is written to `fingerprints.bin` at offset `MemoryID` via memory-mapped I/O. No `seek` + `write` syscall — just a memory copy.

---

### Step 5: MinHash LSH Insert

Now we need a fast way to find "fingerprints that probably overlap with this one." We use **MinHash LSH** (Locality Sensitive Hashing).

**How MinHash works:**
1. Take the 10240-bit fingerprint and imagine 100 different hash functions looking at it.
2. For each hash function, find the first `1` bit it encounters. This gives 100 numbers.
3. Group those 100 numbers into 25 **bands** of 4 **rows** each.
4. Hash each band to a bucket ID. If two fingerprints share even one band hash, they're probably similar.

The `MemoryID` is appended to each of the 25 bucket lists.

**Analogy:** Imagine 100 people each pick the first red door they see in a building. You group their answers into 25 teams of 4. If two buildings produce the same answer for even one team, the buildings are probably very similar. You only need to compare buildings that share a team answer — a massive shortcut.

---

### Step 6: Lexical Index Insert

While the fingerprint captures meaning, tokens themselves still matter. We build an inverted index:

```
"quick" → [(MemoryID=0, freq=1), (MemoryID=1280, freq=2), ...]
"fox"   → [(MemoryID=0, freq=1), ...]
```

We also track:
- How many tokens each memory has (`memLengths`)
- How often each token appears across the entire corpus (`corpusTermFreqs`)
- Total tokens in the corpus (`totalCorpusTokens`)

This supports **Query Likelihood Model (QLM)** scoring at search time.

---

### Step 7: HNSW Graph Insert

The HNSW (Hierarchical Navigable Small World) graph is a multi-layer structure for approximate nearest neighbor search.

**How HNSW works:**
- Each memory becomes a **node** in the graph.
- A random layer is assigned with exponential decay: most nodes live at layer 0, fewer at layer 1, very few at layer 7.
- Layer 0 is the densest — every node connects to its nearest neighbors.
- Higher layers are sparser — they act as "expressways" for fast traversal.

**Insertion process:**
1. Assign a random target layer (0–7).
2. Start at the global entry point (highest layer).
3. At each layer above the target: greedily walk to the closest neighbor.
4. At the target layer and below: find the `efConstruction=200` nearest neighbors.
5. Connect bidirectionally, but prune each node to `maxNeighbors=32` closest edges.

**Analogy:** HNSW is like a road network. Layer 7 is an interstate (few exits, fast travel). Layer 0 is local streets (many connections, precise arrival). You take the interstate as far as possible, then exit to progressively smaller roads until you reach your exact destination.

---

## Search Flow (Step-by-Step)

### Step 1: Encode the Query

The query text goes through the **exact same SDR encoder** as memories. It produces a 10240-bit query fingerprint using the same IDF values from the corpus.

**Why the same encoder?** So query fingerprints and memory fingerprints speak the same language. If a query and memory share meaning, their bit patterns will overlap.

---

### Step 2: Semantic Search — The "LSH Wormhole"

We need to find memories whose fingerprints are similar to the query fingerprint. We do this in two phases:

#### Phase A: MinHash LSH Query
1. Compute the query's 100 MinHash values (same as during insert).
2. Band and hash them into 25 buckets.
3. Collect all `MemoryID`s stored in any matching bucket.
4. Deduplicate.

These are **seed candidates** — memories that are *probably* similar. This is extremely fast because we're just doing 25 hash table lookups.

#### Phase B: HNSW Graph Descent
Here's the "Wormhole" part: instead of starting HNSW search from the global entry point, we **drop the LSH seeds directly into layer 0** and search outward from there.

Why? On small datasets, HNSW search from the top can miss local clusters. The LSH seeds act as teleportation points — they drop you right next to promising candidates.

**Search process:**
1. Start from each LSH seed at layer 0 (or the global entry point if no seeds exist).
2. Greedily walk to neighbors with highest Jaccard overlap until no improvement.
3. Use a beam width of `efSearch=64` — keep the 64 best candidates at each step.
4. Score candidates by **weighted Jaccard**: tokens, bigrams, and context blocks each contribute differently (50%, 25%, 25%).
5. Return the top-K by `1.0 - weightedJaccard` (treated as distance).

---

### Step 3: Lexical Search

Independently, we run a traditional token-based search:

1. Tokenize the query.
2. For each query token, look up its postings list in the inverted index.
3. Score each memory using **Query Likelihood with Dirichlet smoothing**:

```
Score = Σ_q ln(1 + TF(q,D) / (μ × P(q|Corpus))) + |Q| × ln(μ / (|D| + μ))
```

**What this means in plain English:**
- `TF(q,D)` = how many times token `q` appears in memory `D`
- `P(q|Corpus)` = how common token `q` is across all memories
- `μ = 2000` = smoothing parameter (prevents zero probabilities for unseen tokens)
- Memories that contain many query tokens, especially rare ones, score higher.
- Longer memories are slightly penalized.

The key optimization: we iterate postings lists directly and accumulate scores in a hash table. No nested per-memory lookups.

---

### Step 4: RRF Merge — Combining Semantic and Lexical

Now we have two ranked lists:
- **Semantic top-K**: memories that *mean* similar things (from HNSW + LSH)
- **Lexical top-K**: memories that *contain* similar tokens (from inverted index)

We merge them with **Reciprocal Rank Fusion (RRF)**:

```
RRF_score = 1/(60 + rank_semantic) + 1/(60 + rank_lexical)
```

**Why RRF?**
- It's simple and parameter-light (just one constant, `k=60`).
- A memory that ranks well in *both* lanes gets a big boost.
- A memory that only appears in one lane can still surface if it ranks very highly there.
- No training required — it works out of the box.

**Analogy:** Imagine two friends recommending restaurants. One knows your taste (semantic), the other knows what's on the menu (lexical). RRF says: if both friends recommend the same place, it's probably great. If only one recommends it, it might still be worth trying if they ranked it #1.

---

### Step 5: Term-Coverage Rerank

After RRF merging, we apply a final reranker that boosts memories containing **all query terms**.

For each candidate:
1. Tokenize the query and the memory.
2. Count how many query terms appear in the memory.
3. `coverageRatio = matchedTerms / totalQueryTerms`
4. `bonus = coverageRatio × 0.5`
5. If the exact query phrase appears: extra `+0.10` bonus.
6. If any adjacent query bigram appears: extra `+0.05` bonus.
7. `finalScore = RRF_score × (1.0 + bonus)`
8. Re-sort by final score descending.

**Why:** The semantic and lexical scores can both be fooled by partial matches. A memory about "fast dogs" might score well for "quick fox" semantically, and a memory with "quick" 50 times might score well lexically. The reranker pushes memories that actually contain all the query tokens to the top.

---

## Storage Layout

| File | Purpose | Format |
|------|---------|--------|
| `data/wal.bin` | Append-only text + metadata | `[MemoryID: u64][timestamp: u64][len: u32][text: bytes]` |
| `data/fingerprints.bin` | Flat fingerprint array | 1280 bytes per fingerprint |
| `data/graph.bin` | Flat HNSW node array | 2056 bytes per node |

All stores are memory-mapped via `mmap` for zero-copy reads.

---

## Building & Testing

```bash
# Build CLI
nim c -d:release adaline.nim

# Insert memories
./adaline insert "The quick brown fox"
./adaline insert "Nim is a systems programming language"

# Search
./adaline search "quick fox" 5

# Show index stats
./adaline stats

# Run tests
nim c -r tests/domain/test_fingerprint.nim
nim c -r tests/domain/test_sdr_encoder.nim
nim c -r tests/domain/test_corpus_index.nim
nim c -r tests/domain/test_hnsw_node.nim
nim c -r tests/domain/test_lexical_index.nim
nim c -r tests/domain/test_minhash_lsh.nim
nim c -r tests/domain/test_reranker.nim
nim c -r tests/domain/test_rrf_merger.nim
nim c -r tests/domain/test_weighted_jaccard.nim
nim c -r tests/domain/test_memory_service.nim
nim c -r tests/use_cases/test_insert_memory.nim
nim c -r tests/use_cases/test_search_memories.nim

# Run BEIR benchmarks
nim c -d:release benchmarks/benchmark_beir.nim
./benchmarks/benchmark_beir scifact
./benchmarks/benchmark_beir nfcorpus

# Run LongMemEval benchmark
python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='xiaowu0162/longmemeval-cleaned', filename='longmemeval_s_cleaned.json', repo_type='dataset', local_dir='benchmarks/data')"
nim c -d:release benchmarks/benchmark_longmemeval.nim
./benchmarks/benchmark_longmemeval
```

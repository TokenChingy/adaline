import ../entities/memory
import ../entities/config
import ../entities/hnsw_node
import ../algorithms/sdr_encoder
import ../algorithms/corpus_index
import ../algorithms/fingerprint_lsh
import ../algorithms/hnsw_graph
import ../algorithms/weighted_jaccard
import ../algorithms/lexical_index
import ../algorithms/rrf_merger
import ../algorithms/reranker
import ../algorithms/chunker
import ../../infrastructure/mmapped_storage
import std/[tables, random, times, algorithm]

type
  MemoryService* = object
    storage*: MmappedStorage
    cfg*: EngineConfig
    lsh*: FingerprintLshIndex
    lexical*: LexicalIndex
    corpus*: CorpusIndex
    maxHnswLayer*: int
    hnswEntryPoint*: uint64
    memoryIdCounter*: uint64
    textCache*: Table[uint64, string]
    timestampCache*: Table[uint64, uint64]
    chunkToParent*: Table[uint64, uint64]

proc initMemoryService*(dataDir: string; cfg: EngineConfig = defaultEngineConfig()): MemoryService =
  randomize()
  result.storage = initStorage(dataDir)
  result.cfg = cfg
  result.lsh = initLshIndex(cfg)
  result.lexical = LexicalIndex(mu: cfg.dirichletMu)
  result.corpus = CorpusIndex()
  result.maxHnswLayer = -1
  result.hnswEntryPoint = 0
  result.memoryIdCounter = 0
  result.textCache = initTable[uint64, string]()
  result.timestampCache = initTable[uint64, uint64]()
  result.chunkToParent = initTable[uint64, uint64]()

  # Replay chunk mappings first
  let chunkEntries = replayChunks(result.storage)
  var parentToChunks = initTable[uint64, seq[uint64]]()
  var maxId: uint64 = 0
  var hasData = false
  for (parentId, chunkId) in chunkEntries:
    result.chunkToParent[chunkId] = parentId
    parentToChunks.mgetOrPut(parentId, @[]).add(chunkId)
    if chunkId > maxId:
      maxId = chunkId
    hasData = true

  # Replay WAL to rebuild in-memory indexes
  let entries = replayWal(result.storage)
  for (parentId, timestamp, text) in entries:
    result.textCache[parentId] = text
    result.timestampCache[parentId] = timestamp
    result.corpus.addMemory(text)
    if parentId > maxId:
      maxId = parentId
    hasData = true

    # Determine chunk IDs for this parent
    let storedChunkIds = parentToChunks.getOrDefault(parentId, @[])
    let chunkTexts = splitIntoChunks(text, cfg)

    let effectiveChunkIds = if storedChunkIds.len > 0:
                              storedChunkIds
                            else:
                              # Legacy unchunked memory
                              @[parentId]

    let numChunks = min(effectiveChunkIds.len, chunkTexts.len)
    for i in 0 ..< numChunks:
      let chunkId = effectiveChunkIds[i]
      let chunkText = chunkTexts[i]

      # Rebuild lexical index from chunk text
      addMemory(result.lexical, chunkId, chunkText)

      # Rebuild LSH from stored fingerprint
      let fpPtr = result.storage.getFingerprintPtr(chunkId)
      insertLsh(result.lsh, fpPtr, chunkId)

      # Find HNSW entry point from stored graph
      let node = result.storage.getHnswNodePtr(chunkId)
      if node.layerCount > 0:
        let layer = int(node.entryLayer)
        if layer > result.maxHnswLayer:
          result.maxHnswLayer = layer
          result.hnswEntryPoint = chunkId

  if hasData:
    result.memoryIdCounter = maxId + uint64(cfg.fingerprintBytes)
  else:
    result.memoryIdCounter = 0

proc insert*(service: var MemoryService; content: string): uint64 =
  let parentId = service.memoryIdCounter
  service.memoryIdCounter += uint64(service.cfg.fingerprintBytes)
  let timestamp = uint64(getTime().toUnix())

  # 1. WAL stores the parent memory
  discard service.storage.appendWal(parentId, timestamp, content)
  service.textCache[parentId] = content
  service.timestampCache[parentId] = timestamp

  # 2. Update corpus index with full text
  service.corpus.addMemory(content)

  # 3. Chunk if needed
  let chunks = splitIntoChunks(content, service.cfg)

  if chunks.len == 1:
    # Unchunked: parent is its own chunk
    let chunkId = parentId
    service.chunkToParent[chunkId] = parentId
    discard service.storage.appendChunkMapping(parentId, chunkId)

    var fp = encodeSdr(content, service.cfg, service.corpus)
    service.storage.writeFingerprint(chunkId, fp)

    insertLsh(service.lsh, addr fp, chunkId)

    addMemory(service.lexical, chunkId, content)

    service.storage.ensureGraphCapacity(chunkId)
    insertHnsw(service.storage.graphMem, service.storage.fpMem, chunkId, addr fp,
               service.cfg, service.maxHnswLayer, service.hnswEntryPoint)
  else:
    # Chunked: create multiple indexable units
    for chunkText in chunks:
      let chunkId = service.memoryIdCounter
      service.memoryIdCounter += uint64(service.cfg.fingerprintBytes)
      service.chunkToParent[chunkId] = parentId
      discard service.storage.appendChunkMapping(parentId, chunkId)

      var fp = encodeSdr(chunkText, service.cfg, service.corpus)
      service.storage.writeFingerprint(chunkId, fp)

      insertLsh(service.lsh, addr fp, chunkId)

      addMemory(service.lexical, chunkId, chunkText)

      service.storage.ensureGraphCapacity(chunkId)
      insertHnsw(service.storage.graphMem, service.storage.fpMem, chunkId, addr fp,
                 service.cfg, service.maxHnswLayer, service.hnswEntryPoint)

  result = parentId

proc search*(service: var MemoryService; query: string; k: int): seq[Memory] =
  let qfp = encodeSdr(query, service.cfg, service.corpus, isQuery = true)
  # Semantic lane: LSH seeds + HNSW approximate search
  var semanticResults = newSeq[tuple[memoryId: uint64, score: float]]()
  let lshSeeds = queryLsh(service.lsh, addr qfp)
  if service.hnswEntryPoint != 0 or lshSeeds.len > 0:
    semanticResults = searchHnsw(service.storage.graphMem, service.storage.fpMem,
                                 lshSeeds, service.hnswEntryPoint, addr qfp, k, service.cfg)

  # Lexical lane at chunk level
  var lexicalResults = searchLexical(service.lexical, query, k)

  # RRF merge at chunk level
  let merged = mergeRrf(semanticResults, lexicalResults, k, service.cfg.rrfK,
                        service.cfg.semanticRrfWeight, service.cfg.lexicalRrfWeight)

  # Map chunks to parents and deduplicate, keeping best score per parent
  var parentScores = initTable[uint64, float]()
  for item in merged:
    let chunkId = item.memoryId
    let parentId = service.chunkToParent.getOrDefault(chunkId, chunkId)
    let existing = parentScores.getOrDefault(parentId, -1.0)
    if item.score > existing:
      parentScores[parentId] = item.score

  # Build candidates from parents
  var candidates = newSeq[Memory]()
  for parentId, score in parentScores:
    candidates.add(Memory(
      id: parentId,
      content: service.textCache.getOrDefault(parentId, ""),
      score: score,
      createdAt: service.timestampCache.getOrDefault(parentId, 0)
    ))

  # Sort by score descending before rerank
  candidates.sort(proc(a, b: Memory): int =
    if a.score > b.score: return -1
    if a.score < b.score: return 1
    return 0
  )

  # Limit to k before reranking
  if candidates.len > k:
    candidates.setLen(k)

  # Rerank top candidates with term-coverage boost on full parent text
  rerank(query, candidates, service.textCache, service.cfg)
  result = candidates

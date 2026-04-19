import ../entities/memory
import ../entities/config
import ../entities/hnsw_node
import ../algorithms/sdr_encoder
import ../algorithms/corpus_index
import ../algorithms/fingerprint_lsh
import ../algorithms/hnsw_graph
import ../algorithms/lexical_index
import ../algorithms/rrf_merger
import ../algorithms/reranker
import ../algorithms/chunker
import ../../infrastructure/mmapped_storage
import std/[tables, random, times, algorithm, os]

type
  MemoryService* = object
    storage*: MmappedStorage
    cfg*: EngineConfig
    lsh*: FingerprintLshIndex
    lexical*: LexicalIndex
    corpus*: CorpusIndex
    maxHnswLayer*: int
    hnswEntryPoint*: uint64
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
  result.textCache = initTable[uint64, string]()
  result.timestampCache = initTable[uint64, uint64]()
  result.chunkToParent = initTable[uint64, uint64]()

  # Try loading persisted indexes
  let lshPath = dataDir / "lsh.bin"
  let lexicalPath = dataDir / "lexical.bin"
  let corpusPath = dataDir / "corpus.bin"
  var lshOffset, lexicalOffset, corpusOffset: uint64
  var haveLsh = false
  var haveLexical = false
  var haveCorpus = false
  if fileExists(lshPath) and fileExists(lexicalPath) and fileExists(corpusPath):
    result.lsh = loadLsh(cfg, lshPath, lshOffset)
    result.lexical = loadLexical(lexicalPath, lexicalOffset)
    result.corpus = loadCorpus(corpusPath, corpusOffset)
    haveLsh = lshOffset > 0 or result.lsh.buckets.len > 0
    haveLexical = lexicalOffset > 0 or result.lexical.postings.len > 0
    haveCorpus = corpusOffset > 0 or result.corpus.memFreqs.len > 0

  let persistedOffset = if haveLsh and haveLexical and haveCorpus and
                          lshOffset == lexicalOffset and lexicalOffset == corpusOffset:
                          lshOffset
                        else:
                          0'u64

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

  # Replay WAL: always rebuild textCache/timestampCache; only rebuild
  # LSH/lexical/corpus for entries after persistedOffset
  let entries = replayWal(result.storage)
  var walPos: uint64 = 0
  for (parentId, timestamp, text) in entries:
    result.textCache[parentId] = text
    result.timestampCache[parentId] = timestamp
    if parentId > maxId:
      maxId = parentId
    hasData = true

    let entrySize = uint64(sizeof(uint64) + sizeof(uint64) + sizeof(uint32) + text.len)
    let entryStart = walPos
    walPos += entrySize

    # Determine chunk IDs for this parent
    let storedChunkIds = parentToChunks.getOrDefault(parentId, @[])
    let chunkTexts = splitIntoChunks(text, cfg)
    let effectiveChunkIds = if storedChunkIds.len > 0:
                              storedChunkIds
                            else:
                              @[parentId]
    let numChunks = min(effectiveChunkIds.len, chunkTexts.len)

    if entryStart < persistedOffset:
      # Already in persisted indexes; just rebuild corpus (needed for IDF)
      result.corpus.addMemory(text)
    else:
      # New entry since last checkpoint
      result.corpus.addMemory(text)
      for i in 0 ..< numChunks:
        let chunkId = effectiveChunkIds[i]
        let chunkText = chunkTexts[i]
        addMemory(result.lexical, chunkId, chunkText)
        let fpPtr = result.storage.getFingerprintPtr(chunkId)
        insertLsh(result.lsh, fpPtr, chunkId)

    # Find HNSW entry point from stored graph (always needed)
    for i in 0 ..< numChunks:
      let chunkId = effectiveChunkIds[i]
      let node = result.storage.getHnswNodePtr(chunkId)
      if node.layerCount > 0:
        let layer = int(node.entryLayer)
        if layer > result.maxHnswLayer:
          result.maxHnswLayer = layer
          result.hnswEntryPoint = chunkId

  if hasData:
    result.storage.syncRecordCount(maxId + 1)

proc insert*(service: var MemoryService; content: string): uint64 =
  let parentId = service.storage.allocSlot()
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
      let chunkId = service.storage.allocSlot()
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

proc deleteMemory*(service: var MemoryService; parentId: uint64) =
  ## Delete a memory and all its chunks. Frees slots for reuse.
  ## Note: WAL does not currently persist tombstones; deleted memories
  ## will reappear on restart until checkpointing is implemented.
  if not service.textCache.hasKey(parentId):
    return

  let content = service.textCache[parentId]

  # Find all chunk IDs for this parent
  var chunkIds = newSeq[uint64]()
  for chunkId, pid in service.chunkToParent:
    if pid == parentId:
      chunkIds.add(chunkId)

  # If unchunked, the parent is its own chunk
  if chunkIds.len == 0:
    chunkIds.add(parentId)

  for chunkId in chunkIds:
    # Remove from LSH (needs fingerprint before freeing slot)
    let fpPtr = service.storage.getFingerprintPtr(chunkId)
    removeLsh(service.lsh, fpPtr, chunkId)

    # Remove from lexical index
    let chunkText = if chunkId == parentId: content
                    else: service.textCache.getOrDefault(service.chunkToParent.getOrDefault(chunkId, chunkId), "")
    removeMemory(service.lexical, chunkId, chunkText)

    # Free the storage slot
    service.storage.freeSlot(chunkId)

    # Remove chunk→parent mapping
    service.chunkToParent.del(chunkId)

  # Remove parent from caches
  service.textCache.del(parentId)
  service.timestampCache.del(parentId)

  # Sync header to disk after delete
  service.storage.syncHeader()

proc checkpoint*(service: var MemoryService) =
  ## Persist in-memory indexes to disk so future restarts can skip WAL replay.
  let walOffset = service.storage.walSize
  let lshPath = service.storage.dataDir / "lsh.bin"
  let lexicalPath = service.storage.dataDir / "lexical.bin"
  let corpusPath = service.storage.dataDir / "corpus.bin"
  saveLsh(service.lsh, lshPath, walOffset)
  saveLexical(service.lexical, lexicalPath, walOffset)
  saveCorpus(service.corpus, corpusPath, walOffset)

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

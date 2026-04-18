import ../entities/memory
import ../entities/config
import ../entities/hnsw_node
import ../algorithms/sdr_encoder
import ../algorithms/corpus_index
import ../algorithms/minhash_lsh
import ../algorithms/hnsw_graph
import ../algorithms/lexical_index
import ../algorithms/rrf_merger
import ../algorithms/reranker
import ../../infrastructure/mmapped_storage
import std/[tables, random, times]

type
  MemoryService* = object
    storage*: MmappedStorage
    cfg*: EngineConfig
    lsh*: MinHashLshIndex
    lexical*: LexicalIndex
    corpus*: CorpusIndex
    maxHnswLayer*: int
    hnswEntryPoint*: uint64
    memoryIdCounter*: uint64
    textCache*: Table[uint64, string]
    timestampCache*: Table[uint64, uint64]

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

  # Replay WAL to rebuild in-memory indexes
  let entries = replayWal(result.storage)
  for (memoryId, timestamp, text) in entries:
    if memoryId >= result.memoryIdCounter:
      result.memoryIdCounter = memoryId + uint64(cfg.fingerprintBytes)
    result.textCache[memoryId] = text
    result.timestampCache[memoryId] = timestamp
    result.corpus.addMemory(text)

    let fpPtr = result.storage.getFingerprintPtr(memoryId)
    let sig = computeSignature(fpPtr, cfg)
    insertLsh(result.lsh, sig, memoryId)
    addMemory(result.lexical, memoryId, text)

    let node = result.storage.getHnswNodePtr(memoryId)
    if node.layerCount > 0:
      let layer = int(node.entryLayer)
      if layer > result.maxHnswLayer:
        result.maxHnswLayer = layer
        result.hnswEntryPoint = memoryId

proc insert*(service: var MemoryService; content: string): uint64 =
  let memoryId = service.memoryIdCounter
  service.memoryIdCounter += uint64(service.cfg.fingerprintBytes)
  let timestamp = uint64(getTime().toUnix())

  # 1. WAL
  discard service.storage.appendWal(memoryId, timestamp, content)
  service.textCache[memoryId] = content
  service.timestampCache[memoryId] = timestamp

  # 2. Update corpus index
  service.corpus.addMemory(content)

  # 3. Fingerprint (with IDF scaling)
  var fp = encodeSdr(content, service.cfg, service.corpus)
  service.storage.writeFingerprint(memoryId, fp)

  # 4. LSH
  let sig = computeSignature(addr fp, service.cfg)
  insertLsh(service.lsh, sig, memoryId)

  # 5. Lexical
  addMemory(service.lexical, memoryId, content)

  # 6. HNSW
  service.storage.ensureGraphCapacity(memoryId)
  insertHnsw(service.storage.graphMem, service.storage.fpMem, memoryId, addr fp,
             service.cfg, service.maxHnswLayer, service.hnswEntryPoint)

  result = memoryId

proc search*(service: var MemoryService; query: string; k: int): seq[Memory] =
  let qfp = encodeSdr(query, service.cfg, service.corpus)
  let qsig = computeSignature(addr qfp, service.cfg)

  # LSH Wormhole: hash the query through MinHash LSH to retrieve seed MemoryIDs,
  # then drop those seeds into the HNSW graph's lower layers and greedily
  # search outward to find the true local optimums.
  let lshSeeds = queryLsh(service.lsh, qsig)
  var semanticResults = newSeq[tuple[memoryId: uint64, score: float]]()
  if service.hnswEntryPoint != 0 or lshSeeds.len > 0:
    semanticResults = searchHnsw(service.storage.graphMem, service.storage.fpMem,
                                 lshSeeds, service.hnswEntryPoint, addr qfp, k, service.cfg)

  # Lexical lane
  var lexicalResults = searchLexical(service.lexical, query, k)

  # RRF merge
  let merged = mergeRrf(semanticResults, lexicalResults, k, service.cfg.rrfK)

  var candidates = newSeq[Memory](merged.len)
  for i in 0 ..< merged.len:
    let mid = merged[i].memoryId
    candidates[i] = Memory(
      id: mid,
      content: service.textCache.getOrDefault(mid, ""),
      score: merged[i].score,
      createdAt: service.timestampCache.getOrDefault(mid, 0)
    )

  # Rerank top candidates with term-coverage boost
  rerank(query, candidates, service.textCache, service.cfg)
  result = candidates

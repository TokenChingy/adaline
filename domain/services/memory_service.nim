import ../entities/memory
import ../entities/config
import ../entities/hnsw_node
import ../algorithms/sdr_encoder
import ../algorithms/corpus_index
import ../algorithms/weighted_jaccard
import ../algorithms/minhash_lsh
import ../algorithms/hnsw_graph
import ../algorithms/lexical_index
import ../algorithms/rrf_merger
import ../../infrastructure/mmapped_storage
import std/[tables, random, algorithm]

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

  # Replay WAL to rebuild in-memory indexes
  let entries = replayWal(result.storage)
  for (memoryId, text) in entries:
    if memoryId >= result.memoryIdCounter:
      result.memoryIdCounter = memoryId + uint64(cfg.fingerprintBytes)
    result.textCache[memoryId] = text
    result.corpus.addDocument(text)

    let fpPtr = result.storage.getFingerprintPtr(memoryId)
    let sig = computeSignature(fpPtr, cfg)
    insertLsh(result.lsh, sig, memoryId)
    addDocument(result.lexical, memoryId, text)

    let node = result.storage.getHnswNodePtr(memoryId)
    if node.layerCount > 0:
      let layer = int(node.entryLayer)
      if layer > result.maxHnswLayer:
        result.maxHnswLayer = layer
        result.hnswEntryPoint = memoryId

proc insert*(service: var MemoryService; content: string): uint64 =
  let memoryId = service.memoryIdCounter
  service.memoryIdCounter += uint64(service.cfg.fingerprintBytes)

  # 1. WAL
  discard service.storage.appendWal(memoryId, content)
  service.textCache[memoryId] = content

  # 2. Update corpus index
  service.corpus.addDocument(content)

  # 3. Fingerprint (with IDF scaling)
  var fp = encodeSdr(content, service.cfg, service.corpus)
  service.storage.writeFingerprint(memoryId, fp)

  # 4. LSH
  let sig = computeSignature(addr fp, service.cfg)
  insertLsh(service.lsh, sig, memoryId)

  # 5. Lexical
  addDocument(service.lexical, memoryId, content)

  # 6. HNSW
  service.storage.ensureGraphCapacity(memoryId)
  insertHnsw(service.storage.graphMem, service.storage.fpMem, memoryId, addr fp,
             service.cfg, service.maxHnswLayer, service.hnswEntryPoint)

  result = memoryId

proc search*(service: var MemoryService; query: string; k: int): seq[Memory] =
  let qfp = encodeSdr(query, service.cfg, service.corpus)
  let qsig = computeSignature(addr qfp, service.cfg)

  # Semantic lane: LSH candidates -> brute-force scoring (fast for small datasets)
  let lshSeeds = queryLsh(service.lsh, qsig)
  var semanticResults = newSeq[tuple[memoryId: uint64, score: float]]()

  # For small datasets, brute-force score all documents or LSH seeds
  if service.textCache.len < 100:
    var scored = newSeq[tuple[score: float, mid: uint64]]()
    for mid, _ in service.textCache:
      let sptr = service.storage.getFingerprintPtr(mid)
      let score = weightedJaccard(addr qfp, sptr, service.cfg)
      scored.add((score, mid))
    scored.sort(proc(a, b: auto): int =
      if a.score > b.score: return -1
      if a.score < b.score: return 1
      return 0
    )
    for i in 0 ..< min(k, scored.len):
      semanticResults.add((scored[i].mid, scored[i].score))
  elif lshSeeds.len > 0 and service.textCache.len < 10000:
    var scored = newSeq[tuple[score: float, mid: uint64]]()
    for seed in lshSeeds:
      let sptr = service.storage.getFingerprintPtr(seed)
      let score = weightedJaccard(addr qfp, sptr, service.cfg)
      scored.add((score, seed))
    scored.sort(proc(a, b: auto): int =
      if a.score > b.score: return -1
      if a.score < b.score: return 1
      return 0
    )
    for i in 0 ..< min(k, scored.len):
      semanticResults.add((scored[i].mid, scored[i].score))
  elif service.hnswEntryPoint != 0:
    semanticResults = searchHnsw(service.storage.graphMem, service.storage.fpMem,
                                 lshSeeds, service.hnswEntryPoint, addr qfp, k, service.cfg)

  # Lexical lane
  var lexicalResults = searchLexical(service.lexical, query, k)

  # RRF merge
  let merged = mergeRrf(semanticResults, lexicalResults, k, service.cfg.rrfK)

  result = newSeq[Memory](merged.len)
  for i in 0 ..< merged.len:
    let mid = merged[i].memoryId
    result[i] = Memory(
      id: mid,
      content: service.textCache.getOrDefault(mid, ""),
      score: merged[i].score
    )

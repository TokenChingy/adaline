import std/sequtils
import std/algorithm
import std/[os, times, tables, strformat]
import ../domain/services/memory_service
import ../domain/entities/config
import ../domain/entities/fingerprint
import ../domain/entities/hnsw_node
import ../domain/algorithms/sdr_encoder
import ../domain/algorithms/fingerprint_lsh
import ../domain/algorithms/hnsw_graph
import ../domain/algorithms/weighted_jaccard
import ../domain/algorithms/lexical_index
import ../domain/algorithms/chunker
import ../domain/algorithms/corpus_index
import ../infrastructure/mmapped_storage

proc loadCorpus(dataDir: string): seq[string] =
  let f = open(dataDir / "corpus.jsonl")
  defer: f.close()
  for line in f.lines:
    result.add(line)
    if result.len >= 1000: break

# Replicate insertHnsw with instrumentation
proc insertHnswProfiled(graphMem, fpMem: pointer; memoryId: uint64; fp: ptr Fingerprint;
                        cfg: EngineConfig; maxLayer: var int; entryPoint: var uint64;
                        reverseIndex: var Table[uint64, seq[uint64]];
                        tSearchLayer: var float; tNeighborRescore: var float;
                        nJaccardSearch: var int; nJaccardRescore: var int) =
  let node = getHnswNodePtr(graphMem, memoryId)
  clearNode(node)

  let layer = randomLevel(cfg.hnswMaxLayers - 1)
  node.layerCount = uint8(layer + 1)
  node.entryLayer = uint8(layer)

  if entryPoint == 0 or maxLayer < 0:
    entryPoint = memoryId
    maxLayer = layer
    return

  var currEp = entryPoint
  for lc in countdown(maxLayer, layer + 1):
    let t0 = cpuTime()
    let neighbors = searchLayer(graphMem, fpMem, fp, currEp, lc, 1, cfg)
    tSearchLayer += cpuTime() - t0
    if neighbors.len > 0:
      currEp = neighbors[0].id

  for lc in countdown(layer, 0):
    let t0 = cpuTime()
    let neighbors = searchLayer(graphMem, fpMem, fp, currEp, lc, cfg.hnswEfConstruction, cfg)
    tSearchLayer += cpuTime() - t0
    var selected = newSeq[uint64]()
    for i in 0 ..< min(neighbors.len, cfg.hnswMaxNeighbors):
      selected.add(neighbors[i].id)
    setNeighbors(node, lc, selected)

    for nid in selected:
      reverseIndex.mgetOrPut(nid, @[]).add(memoryId)

    for nid in selected:
      let t1 = cpuTime()
      let nnode = getHnswNodePtr(graphMem, nid)
      var oldNeighbors = newSeq[uint64]()
      for nnid in nnode.neighbors(lc):
        oldNeighbors.add(nnid)
      oldNeighbors.add(memoryId)

      var scored = newSeq[tuple[dist: float, id: uint64]]()
      for nnid in oldNeighbors:
        let nnptr = getFingerprintPtr(fpMem, nnid)
        let d = 1.0 - weightedJaccard(getFingerprintPtr(fpMem, nid), nnptr, cfg)
        scored.add((d, nnid))
        nJaccardRescore.inc

      scored.sort(proc(a, b: auto): int =
        if a.dist < b.dist: return -1
        if a.dist > b.dist: return 1
        return 0
      )

      var kept = newSeq[uint64]()
      for i in 0 ..< min(scored.len, cfg.hnswMaxNeighbors):
        kept.add(scored[i].id)
      setNeighbors(nnode, lc, kept)

      for o in oldNeighbors:
        if o notin kept:
          if reverseIndex.hasKey(o):
            reverseIndex[o] = reverseIndex[o].filterIt(it != nid)
      for k in kept:
        if k == memoryId or k notin oldNeighbors[0 ..< oldNeighbors.len - 1]:
          var list = reverseIndex.mgetOrPut(k, @[])
          if nid notin list:
            list.add(nid)
            reverseIndex[k] = list
      tNeighborRescore += cpuTime() - t1

    if neighbors.len > 0:
      currEp = neighbors[0].id

  if layer > maxLayer:
    maxLayer = layer
    entryPoint = memoryId

proc main() =
  let dataDir = "benchmarks/scifact"
  let corpus = loadCorpus(dataDir)
  echo "Profiling HNSW detail on ", corpus.len, " documents"

  var cfg = defaultEngineConfig()
  cfg.hnswEnabled = true
  cfg.hnswEfConstruction = 200
  cfg.hnswMaxNeighbors = 32

  let benchDir = getCurrentDir() / "benchmarks" / "data" / "profile_hnsw_detail"
  removeDir(benchDir)
  var svc = initMemoryService(benchDir, cfg)

  var tSearchLayer = 0.0
  var tNeighborRescore = 0.0
  var nJaccardSearch = 0
  var nJaccardRescore = 0
  var tTotal = 0.0

  let tStart = cpuTime()
  for i, text in corpus:
    let parentId = svc.storage.allocId()
    discard svc.storage.appendWal(parentId, uint64(0), text)
    addMemory(svc.corpus, text)
    let chunks = splitIntoChunks(text, svc.cfg)
    let chunkText = if chunks.len == 1: text else: chunks[0]
    var fp = encodeSdr(chunkText, svc.cfg, svc.corpus)
    svc.storage.writeFingerprintUnsafe(parentId, fp)
    insertLsh(svc.lsh, addr fp, parentId)
    addMemory(svc.lexical, parentId, chunkText)

    insertHnswProfiled(svc.storage.graphMem, svc.storage.fpMem, parentId, addr fp,
                       svc.cfg, svc.maxHnswLayer, svc.hnswEntryPoint,
                       svc.hnswReverseIndex,
                       tSearchLayer, tNeighborRescore,
                       nJaccardSearch, nJaccardRescore)
  tTotal = cpuTime() - tStart

  echo &"\n=== HNSW Detail Profile ({corpus.len} docs) ==="
  echo &"Total insert time: {tTotal:.4f}s ({corpus.len.float / tTotal:.1f} docs/s)"
  echo &"searchLayer time:    {tSearchLayer:.4f}s  ({100.0 * tSearchLayer / tTotal:.1f}%)"
  echo &"neighborRescore time:{tNeighborRescore:.4f}s  ({100.0 * tNeighborRescore / tTotal:.1f}%)"
  echo &"searchLayer Jaccard calls: {nJaccardSearch}"
  echo &"rescore Jaccard calls:     {nJaccardRescore}"
  echo &"Total Jaccard calls:       {nJaccardSearch + nJaccardRescore}"
  echo &"Avg Jaccard per doc:       {(nJaccardSearch + nJaccardRescore).float / corpus.len.float:.0f}"

when isMainModule:
  main()

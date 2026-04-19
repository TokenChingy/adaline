import ../entities/fingerprint
import ../entities/hnsw_node
import ../entities/config
import weighted_jaccard
import std/[heapqueue, sets, random, algorithm, tables]

proc getFingerprintPtr*(fpMem: pointer; memoryId: uint64): ptr Fingerprint =
  let offset = memoryId
  result = cast[ptr Fingerprint](cast[pointer](cast[uint](fpMem) + uint(offset)))

proc getHnswNodePtr*(graphMem: pointer; memoryId: uint64): ptr HnswNode =
  let idx = memoryId div uint64(FingerprintBytes)
  let offset = idx * uint64(sizeof(HnswNode))
  result = cast[ptr HnswNode](cast[pointer](cast[uint](graphMem) + uint(offset)))

proc randomLevel*(maxLayer: int): int =
  var lvl = 0
  while rand(1.0) < 0.5 and lvl < maxLayer:
    lvl.inc
  result = lvl

type
  MinItem = tuple[dist: float, id: uint64]
  MaxItem = tuple[negDist: float, id: uint64]

proc searchLayer*(graphMem, fpMem: pointer; queryFp: ptr Fingerprint; entryId: uint64;
                  layer, ef: int; cfg: EngineConfig): seq[tuple[id: uint64, dist: float]] =
  var visited = initHashSet[uint64]()
  var candidates = initHeapQueue[MinItem]()
  var results = initHeapQueue[MaxItem]()

  let entryPtr = getFingerprintPtr(fpMem, entryId)
  let entryDist = 1.0 - weightedJaccard(queryFp, entryPtr, cfg)
  visited.incl(entryId)
  candidates.push((entryDist, entryId))

  while candidates.len > 0:
    let curr = candidates.pop()
    if results.len > 0 and curr.dist > -results[0].negDist:
      break

    results.push((-curr.dist, curr.id))
    if results.len > ef:
      discard results.pop()

    let node = getHnswNodePtr(graphMem, curr.id)
    for nid in node.neighbors(layer):
      if nid notin visited:
        visited.incl(nid)
        let nptr = getFingerprintPtr(fpMem, nid)
        let ndist = 1.0 - weightedJaccard(queryFp, nptr, cfg)
        if ndist < -results[0].negDist or results.len < ef:
          candidates.push((ndist, nid))

  let n = results.len
  result = newSeq[tuple[id: uint64, dist: float]](n)
  for i in 0 ..< n:
    let (negDist, id) = results.pop()
    result[n - 1 - i] = (id, -negDist)

proc insertHnsw*(graphMem, fpMem: pointer; memoryId: uint64; fp: ptr Fingerprint;
                 cfg: EngineConfig; maxLayer: var int; entryPoint: var uint64) =
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
    let neighbors = searchLayer(graphMem, fpMem, fp, currEp, lc, 1, cfg)
    if neighbors.len > 0:
      currEp = neighbors[0].id

  for lc in countdown(layer, 0):
    let neighbors = searchLayer(graphMem, fpMem, fp, currEp, lc, cfg.hnswEfConstruction, cfg)
    var selected = newSeq[uint64]()
    for i in 0 ..< min(neighbors.len, cfg.hnswMaxNeighbors):
      selected.add(neighbors[i].id)
    setNeighbors(node, lc, selected)

    for nid in selected:
      let nnode = getHnswNodePtr(graphMem, nid)
      var nneighbors = newSeq[uint64]()
      for nnid in nnode.neighbors(lc):
        nneighbors.add(nnid)
      nneighbors.add(memoryId)

      var scored = newSeq[tuple[dist: float, id: uint64]]()
      for nnid in nneighbors:
        let nnptr = getFingerprintPtr(fpMem, nnid)
        let d = 1.0 - weightedJaccard(getFingerprintPtr(fpMem, nid), nnptr, cfg)
        scored.add((d, nnid))

      scored.sort(proc(a, b: auto): int =
        if a.dist < b.dist: return -1
        if a.dist > b.dist: return 1
        return 0
      )

      var kept = newSeq[uint64]()
      for i in 0 ..< min(scored.len, cfg.hnswMaxNeighbors):
        kept.add(scored[i].id)
      setNeighbors(nnode, lc, kept)

    if neighbors.len > 0:
      currEp = neighbors[0].id

  if layer > maxLayer:
    maxLayer = layer
    entryPoint = memoryId

proc searchHnsw*(graphMem, fpMem: pointer; seeds: seq[uint64]; entryPoint: uint64;
                 queryFp: ptr Fingerprint; k: int; cfg: EngineConfig): seq[tuple[memoryId: uint64, score: float]] =
  var allResults = initTable[uint64, float]()

  # --- Lane 1: Score LSH seeds directly ---
  for seed in seeds:
    if not allResults.hasKey(seed):
      let sptr = getFingerprintPtr(fpMem, seed)
      let score = weightedJaccard(queryFp, sptr, cfg)
      allResults[seed] = score

  # --- Lane 2: Standard HNSW descent from entry point ---
  if entryPoint != 0:
    let entryNode = getHnswNodePtr(graphMem, entryPoint)
    let maxLayer = int(entryNode.entryLayer)
    var currEp = entryPoint

    # Greedy descent from top layer to layer 1
    for lc in countdown(maxLayer, 1):
      let neighbors = searchLayer(graphMem, fpMem, queryFp, currEp, lc, 1, cfg)
      if neighbors.len > 0:
        currEp = neighbors[0].id

    # Beam search at layer 0 with efSearch
    let layer0Neighbors = searchLayer(graphMem, fpMem, queryFp, currEp, 0, cfg.hnswEfSearch, cfg)
    for (nid, dist) in layer0Neighbors:
      let score = 1.0 - dist
      if not allResults.hasKey(nid) or score > allResults[nid]:
        allResults[nid] = score

  # Sort and return top-k
  var sorted = newSeq[tuple[score: float, memoryId: uint64]]()
  for mid, score in allResults:
    sorted.add((score, mid))
  sorted.sort(proc(a, b: auto): int =
    if a.score > b.score: return -1
    if a.score < b.score: return 1
    return 0
  )

  let topK = min(k, sorted.len)
  result = newSeq[tuple[memoryId: uint64, score: float]](topK)
  for i in 0 ..< topK:
    result[i] = (sorted[i].memoryId, sorted[i].score)

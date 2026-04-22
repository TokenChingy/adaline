## Hierarchical Navigable Small World (HNSW) graph.
## Construction, greedy best-first search, and neighbor selection
## for sparse fingerprint vectors using weighted Jaccard distance.
## Includes layer assignment, edge wiring, and graph healing on delete.
##
## This module now uses ``HnswNodeView`` so that the on-disk record size
## can vary with the runtime ``hnswMaxLayers`` / ``hnswMaxNeighbors`` config.


import ../entities/fingerprint
import ../entities/hnsw_node
import ../entities/config
import weighted_jaccard
import std/[heapqueue, sets, random, algorithm, tables, sequtils]

proc getFingerprintPtr*(fpMem: pointer; memoryId: uint64): ptr Fingerprint {.inline.} =
  let offset = memoryId * uint64(FingerprintBytes)
  cast[ptr Fingerprint](cast[pointer](cast[uint](fpMem) + uint(offset)))

proc getHnswNodeView*(graphMem: pointer; memoryId: uint64;
                      graphRecordSize: uint64; maxNeighbors: int): HnswNodeView {.inline.} =
  let offset = memoryId * graphRecordSize
  let p = cast[pointer](cast[uint](graphMem) + uint(offset))
  makeView(p, maxNeighbors, HnswMaxLayers)

# Backward-compatible wrapper for tests that still use compile-time max size.
proc getHnswNodePtr*(graphMem: pointer; memoryId: uint64): ptr HnswNode {.inline.} =
  let offset = memoryId * uint64(sizeof(HnswNode))
  cast[ptr HnswNode](cast[pointer](cast[uint](graphMem) + uint(offset)))

proc randomLevel*(maxLayer: int): int =
  var lvl = 0
  while rand(1.0) < 0.5 and lvl < maxLayer:
    lvl.inc
  result = lvl

type
  MinItem = tuple[dist: float, id: uint64]
  MaxItem = tuple[negDist: float, id: uint64]

proc searchLayer*(graphMem, fpMem: pointer; queryFp: ptr Fingerprint; entryId: uint64;
                  layer, ef: int; cfg: EngineConfig; graphRecordSize: uint64): seq[tuple[id: uint64, dist: float]] =
  var visited = initHashSet[uint64]()
  var candidates = initHeapQueue[MinItem]()
  var results = initHeapQueue[MaxItem]()

  let entryPtr = getFingerprintPtr(fpMem, entryId)
  let entryDist = 1.0 - weightedJaccard(queryFp, entryPtr, cfg)
  visited.incl(entryId)
  candidates.push((entryDist, entryId))

  while candidates.len > 0:
    let curr = candidates.pop()
    if results.len >= ef and curr.dist > -results[0].negDist:
      break

    results.push((-curr.dist, curr.id))
    if results.len > ef:
      discard results.pop()

    let node = getHnswNodeView(graphMem, curr.id, graphRecordSize, cfg.hnswMaxNeighbors)
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
                 cfg: EngineConfig; maxLayer: var int; entryPoint: var uint64;
                 reverseIndex: var Table[uint64, seq[uint64]]; graphRecordSize: uint64) =
  let node = getHnswNodeView(graphMem, memoryId, graphRecordSize, cfg.hnswMaxNeighbors)
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
    let neighbors = searchLayer(graphMem, fpMem, fp, currEp, lc, 1, cfg, graphRecordSize)
    if neighbors.len > 0:
      currEp = neighbors[0].id

  for lc in countdown(layer, 0):
    let neighbors = searchLayer(graphMem, fpMem, fp, currEp, lc, cfg.hnswEfConstruction, cfg, graphRecordSize)
    var selected = newSeq[uint64]()
    for i in 0 ..< min(neighbors.len, cfg.hnswMaxNeighbors):
      selected.add(neighbors[i].id)
    setNeighbors(node, lc, selected)

    for nid in selected:
      reverseIndex.mgetOrPut(nid, @[]).add(memoryId)

    for nid in selected:
      let nnode = getHnswNodeView(graphMem, nid, graphRecordSize, cfg.hnswMaxNeighbors)
      var oldNeighbors = newSeq[uint64]()
      for nnid in nnode.neighbors(lc):
        oldNeighbors.add(nnid)

      let newDist = 1.0 - weightedJaccard(
        getFingerprintPtr(fpMem, nid),
        getFingerprintPtr(fpMem, memoryId), cfg)

      if oldNeighbors.len < cfg.hnswMaxNeighbors:
        oldNeighbors.add(memoryId)
        setNeighbors(nnode, lc, oldNeighbors)
        var list = reverseIndex.mgetOrPut(memoryId, @[])
        if nid notin list:
          list.add(nid)
          reverseIndex[memoryId] = list
        continue

      var worstDist = newDist
      var worstIdx = -1
      for i, nnid in oldNeighbors:
        let d = 1.0 - weightedJaccard(
          getFingerprintPtr(fpMem, nid),
          getFingerprintPtr(fpMem, nnid), cfg)
        if d > worstDist:
          worstDist = d
          worstIdx = i

      if worstIdx >= 0:
        let dropped = oldNeighbors[worstIdx]
        oldNeighbors[worstIdx] = memoryId
        setNeighbors(nnode, lc, oldNeighbors)
        if reverseIndex.hasKey(dropped):
          reverseIndex[dropped] = reverseIndex[dropped].filterIt(it != nid)
        var list = reverseIndex.mgetOrPut(memoryId, @[])
        if nid notin list:
          list.add(nid)
          reverseIndex[memoryId] = list
      else:
        if reverseIndex.hasKey(memoryId):
          reverseIndex[memoryId] = reverseIndex[memoryId].filterIt(it != nid)

    if neighbors.len > 0:
      currEp = neighbors[0].id

  if layer > maxLayer:
    maxLayer = layer
    entryPoint = memoryId

proc searchHnsw*(graphMem, fpMem: pointer; seeds: seq[uint64]; entryPoint: uint64;
                 queryFp: ptr Fingerprint; k: int; cfg: EngineConfig; graphRecordSize: uint64): seq[tuple[memoryId: uint64, score: float]] =
  var allResults = initTable[uint64, float]()

  for seed in seeds:
    if not allResults.hasKey(seed):
      let sptr = getFingerprintPtr(fpMem, seed)
      let score = weightedJaccard(queryFp, sptr, cfg)
      allResults[seed] = score

  if entryPoint != 0:
    let entryNode = getHnswNodeView(graphMem, entryPoint, graphRecordSize, cfg.hnswMaxNeighbors)
    let maxLayer = int(entryNode.entryLayer)
    var currEp = entryPoint

    for lc in countdown(maxLayer, 1):
      let neighbors = searchLayer(graphMem, fpMem, queryFp, currEp, lc, 1, cfg, graphRecordSize)
      if neighbors.len > 0:
        currEp = neighbors[0].id

    let layer0Neighbors = searchLayer(graphMem, fpMem, queryFp, currEp, 0, cfg.hnswEfSearch, cfg, graphRecordSize)
    for (nid, dist) in layer0Neighbors:
      let score = 1.0 - dist
      if not allResults.hasKey(nid) or score > allResults[nid]:
        allResults[nid] = score

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

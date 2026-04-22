## Unit tests for HNSW graph algorithm.


import unittest
import ../../domain/algorithms/hnsw_graph
import ../../domain/entities/fingerprint
import ../../domain/entities/hnsw_node
import ../../domain/entities/config
import std/[tables, random]

suite "HNSW graph":
  setup:
    randomize(42)

  test "randomLevel returns values within [0, maxLayer]":
    for _ in 0 ..< 100:
      let lvl = randomLevel(7)
      check lvl >= 0
      check lvl <= 7

  test "randomLevel is non-deterministic":
    var seen = newSeq[int]()
    for _ in 0 ..< 20:
      seen.add(randomLevel(4))
    var allSame = true
    for i in 1 ..< seen.len:
      if seen[i] != seen[0]:
        allSame = false
        break
    check not allSame

  test "insertHnsw sets entry point for first node":
    let cfg = defaultEngineConfig()
    var graphMem = alloc0(sizeof(HnswNode) * 10)
    var fpMem = alloc0(FingerprintBytes * 10)
    defer:
      dealloc(graphMem)
      dealloc(fpMem)

    var maxLayer = -1
    var entryPoint: uint64 = 0
    var reverseIndex = initTable[uint64, seq[uint64]]()

    var fp1 = initFingerprint()
    setBit(fp1, 0)
    setBit(fp1, 100)
    setBit(fp1, 500)
    copyMem(cast[pointer](cast[uint](fpMem) + FingerprintBytes), addr fp1, FingerprintBytes)

    insertHnsw(graphMem, fpMem, 1'u64, cast[ptr Fingerprint](cast[pointer](cast[uint](fpMem) + FingerprintBytes)),
               cfg, maxLayer, entryPoint, reverseIndex, uint64(sizeof(HnswNode)))

    check entryPoint == 1'u64
    check maxLayer >= 0
    let node1 = getHnswNodePtr(graphMem, 1'u64)
    check node1.layerCount > 0

  test "insertHnsw builds neighbors for second node":
    let cfg = defaultEngineConfig()
    var graphMem = alloc0(sizeof(HnswNode) * 10)
    var fpMem = alloc0(FingerprintBytes * 10)
    defer:
      dealloc(graphMem)
      dealloc(fpMem)

    var maxLayer = -1
    var entryPoint: uint64 = 0
    var reverseIndex = initTable[uint64, seq[uint64]]()

    for id in 1'u64 .. 2'u64:
      var fp = initFingerprint()
      setBit(fp, 0)
      setBit(fp, 100)
      setBit(fp, 500)
      if id == 2:
        setBit(fp, 600)
      let offset = id * uint64(FingerprintBytes)
      copyMem(cast[pointer](cast[uint](fpMem) + uint(offset)), addr fp, FingerprintBytes)
      insertHnsw(graphMem, fpMem, id, cast[ptr Fingerprint](cast[pointer](cast[uint](fpMem) + uint(offset))),
                 cfg, maxLayer, entryPoint, reverseIndex, uint64(sizeof(HnswNode)))

    let node2 = getHnswNodePtr(graphMem, 2'u64)
    check node2.layerCount > 0

    var hasNeighbor = false
    for lc in 0 ..< int(node2.layerCount):
      for nid in node2.neighbors(lc):
        if nid == 1'u64:
          hasNeighbor = true
    check hasNeighbor

  test "searchHnsw returns results for query matching inserted nodes":
    let cfg = defaultEngineConfig()
    var graphMem = alloc0(sizeof(HnswNode) * 10)
    var fpMem = alloc0(FingerprintBytes * 10)
    defer:
      dealloc(graphMem)
      dealloc(fpMem)

    var maxLayer = -1
    var entryPoint: uint64 = 0
    var reverseIndex = initTable[uint64, seq[uint64]]()

    for id in 1'u64 .. 5'u64:
      var fp = initFingerprint()
      for b in 0 ..< 50:
        setBit(fp, b)
      setBit(fp, int(id) * 10)
      let offset = id * uint64(FingerprintBytes)
      copyMem(cast[pointer](cast[uint](fpMem) + uint(offset)), addr fp, FingerprintBytes)
      insertHnsw(graphMem, fpMem, id, cast[ptr Fingerprint](cast[pointer](cast[uint](fpMem) + uint(offset))),
                 cfg, maxLayer, entryPoint, reverseIndex, uint64(sizeof(HnswNode)))

    var qfp = initFingerprint()
    for b in 0 ..< 50:
      setBit(qfp, b)
    setBit(qfp, 30)

    let results = searchHnsw(graphMem, fpMem, @[], entryPoint, addr qfp, 3, cfg,
                             uint64(sizeof(HnswNode)))
    check results.len > 0
    check results[0].score > 0.0

  test "searchLayer returns sorted results by distance":
    let cfg = defaultEngineConfig()
    var graphMem = alloc0(sizeof(HnswNode) * 10)
    var fpMem = alloc0(FingerprintBytes * 10)
    defer:
      dealloc(graphMem)
      dealloc(fpMem)

    var fp1 = initFingerprint()
    setBit(fp1, 0)
    setBit(fp1, 1)
    copyMem(cast[pointer](cast[uint](fpMem) + FingerprintBytes), addr fp1, FingerprintBytes)

    var fp2 = initFingerprint()
    setBit(fp2, 0)
    setBit(fp2, 2)
    copyMem(cast[pointer](cast[uint](fpMem) + 2 * FingerprintBytes), addr fp2, FingerprintBytes)

    let n1 = getHnswNodePtr(graphMem, 1'u64)
    n1.layerCount = 1
    n1.entryLayer = 0
    setNeighbors(n1, 0, @[2'u64])

    let n2 = getHnswNodePtr(graphMem, 2'u64)
    n2.layerCount = 1
    n2.entryLayer = 0
    setNeighbors(n2, 0, @[1'u64])

    var qfp = initFingerprint()
    setBit(qfp, 0)
    setBit(qfp, 1)

    let results = searchLayer(graphMem, fpMem, addr qfp, 1'u64, 0, 10, cfg,
                              uint64(sizeof(HnswNode)))
    check results.len >= 1
    check results[0].id == 1'u64
    for i in 1 ..< results.len:
      check results[i].dist >= results[i-1].dist

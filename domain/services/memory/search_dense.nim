## Dense-vector search service.
## Encodes a query vector to a fingerprint and searches LSH brute-force.


import ./types
import ../../entities/fingerprint
import ../../algorithms/dense_encoder
import ../../algorithms/fingerprint_lsh
import ../../algorithms/weighted_jaccard
import ../../../infrastructure/mmapped_storage
import std/algorithm

export types

proc searchDense*(service: var MemoryService; vec: seq[float32]; topK: int): seq[tuple[memoryId: uint64, score: float]] =
  let qfp = encodeDense(vec, service.cfg)
  var seeds = queryLsh(service.lsh, addr qfp)
  seeds.sort()
  var scored = newSeq[tuple[score: float, memoryId: uint64]](seeds.len)
  for i in 0 ..< seeds.len:
    var sfp: Fingerprint
    service.storage.readFingerprint(seeds[i], addr sfp)
    scored[i] = (weightedJaccard(addr qfp, addr sfp, service.cfg), seeds[i])
  scored.sort(proc(a, b: auto): int =
    if a.score > b.score: return -1
    if a.score < b.score: return 1
    return 0
  )
  let k = min(topK, scored.len)
  result = newSeq[tuple[memoryId: uint64, score: float]](k)
  for i in 0 ..< k:
    result[i] = (scored[i].memoryId, scored[i].score)

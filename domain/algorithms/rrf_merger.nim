# Reciprocal Rank Fusion (RRF) merger.
# Combines semantic and lexical lane result lists by summing
# reciprocal ranks, then deduplicates by parent memory ID.


import std/[tables, algorithm]

proc mergeRrf*(semantic: seq[tuple[memoryId: uint64, score: float]];
               lexical: seq[tuple[memoryId: uint64, score: float]];
               k: int; rrfK: int): seq[tuple[memoryId: uint64, score: float]] =
  var scores = initTable[uint64, float]()

  for rank, item in semantic:
    let rrfScore = 1.0 / (float(rrfK) + float(rank + 1))
    scores[item.memoryId] = scores.getOrDefault(item.memoryId, 0.0) + rrfScore

  for rank, item in lexical:
    let rrfScore = 1.0 / (float(rrfK) + float(rank + 1))
    scores[item.memoryId] = scores.getOrDefault(item.memoryId, 0.0) + rrfScore

  var ranked = newSeq[tuple[score: float, memoryId: uint64]]()
  for mid, score in scores:
    ranked.add((score, mid))

  ranked.sort(proc(a, b: auto): int =
    if a.score > b.score: return -1
    if a.score < b.score: return 1
    return 0
  )

  let topK = min(k, ranked.len)
  result = newSeq[tuple[memoryId: uint64, score: float]](topK)
  for i in 0 ..< topK:
    result[i] = (ranked[i].memoryId, ranked[i].score)

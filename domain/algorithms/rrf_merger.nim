## Reciprocal Rank Fusion (RRF) merger with score awareness (RRF-S).
## Combines semantic and lexical lane result lists by summing
## reciprocal ranks weighted by normalised raw scores. This preserves
## RRF's robustness while using the magnitude of each lane's similarity
## scores as a tie-breaker.


import std/[tables, algorithm]

proc mergeRrf*(semantic: seq[tuple[memoryId: uint64, score: float]];
               lexical: seq[tuple[memoryId: uint64, score: float]];
               k: int; rrfK: int): seq[tuple[memoryId: uint64, score: float]] =
  var scores = initTable[uint64, float]()

  let semMax = if semantic.len > 0: semantic[0].score else: 1.0
  let semMin = if semantic.len > 0: semantic[^1].score else: 0.0
  let lexMax = if lexical.len > 0: lexical[0].score else: 1.0
  let lexMin = if lexical.len > 0: lexical[^1].score else: 0.0

  let semSpread = semMax - semMin
  let lexSpread = lexMax - lexMin

  for rank, item in semantic:
    let norm = if semSpread > 1e-9: (item.score - semMin) / semSpread else: 1.0
    let rrfScore = 1.0 / (float(rrfK) + float(rank + 1)) * (1.0 + norm)
    scores[item.memoryId] = scores.getOrDefault(item.memoryId, 0.0) + rrfScore

  for rank, item in lexical:
    let norm = if lexSpread > 1e-9: (item.score - lexMin) / lexSpread else: 1.0
    let rrfScore = 1.0 / (float(rrfK) + float(rank + 1)) * (1.0 + norm)
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

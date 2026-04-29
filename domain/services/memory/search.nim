## Search memory service.
## Runs parallel semantic (LSH brute-force) and lexical (inverted index)
## lanes, merges results via score-aware RRF, resolves chunks to parents,
## and applies phrase-aware lexical reranking on the top candidates.


import ./types
import ../../entities/memory
import ../../entities/fingerprint
import ../../algorithms/sdr_encoder
import ../../algorithms/fingerprint_lsh
import ../../algorithms/weighted_jaccard
import ../../algorithms/lexical_index
import ../../algorithms/rrf_merger
import ../../algorithms/reranker
import ../../../infrastructure/mmapped_storage
import std/[tables, algorithm]

export types
export memory

proc search*(service: var MemoryService; query: string; k: int): seq[Memory] =
  let qfp = encodeSdr(query, service.cfg, service.corpus, isQuery = true)
  var semanticResults = newSeq[tuple[memoryId: uint64, score: float]]()
  if service.cfg.semanticSearchEnabled:
    var lshSeeds = queryLsh(service.lsh, addr qfp)
    lshSeeds.sort()
    var scored = newSeq[tuple[score: float, memoryId: uint64]](lshSeeds.len)
    for i in 0 ..< lshSeeds.len:
      var sfp: Fingerprint
      service.storage.readFingerprint(lshSeeds[i], addr sfp)
      scored[i] = (weightedJaccard(addr qfp, addr sfp, service.cfg), lshSeeds[i])
    scored.sort(proc(a, b: auto): int =
      if a.score > b.score: return -1
      if a.score < b.score: return 1
      return 0
    )
    let topK = min(k, scored.len)
    semanticResults = newSeq[tuple[memoryId: uint64, score: float]](topK)
    for i in 0 ..< topK:
      semanticResults[i] = (scored[i].memoryId, scored[i].score)

  var lexicalResults = newSeq[tuple[memoryId: uint64, score: float]]()
  if service.cfg.lexicalSearchEnabled:
    lexicalResults = searchLexical(service.lexical, query, k)

  let merged = mergeRrf(semanticResults, lexicalResults, k, service.cfg.rrfK)

  var parentScores = initTable[uint64, float]()
  for item in merged:
    let chunkId = item.memoryId
    let parentId = service.chunkToParent.getOrDefault(chunkId, chunkId)
    let existing = parentScores.getOrDefault(parentId, -1.0)
    if item.score > existing:
      parentScores[parentId] = item.score

  var candidates = newSeq[Memory]()
  for parentId, score in parentScores:
    candidates.add(Memory(
      id: parentId,
      content: service.textCache.getOrDefault(parentId, ""),
      score: score,
      createdAt: service.timestampCache.getOrDefault(parentId, 0)
    ))

  candidates.sort(proc(a, b: Memory): int =
    if a.score > b.score: return -1
    if a.score < b.score: return 1
    return 0
  )

  if candidates.len > k:
    candidates.setLen(k)

  if candidates.len > 30:
    var topCandidates = candidates[0 ..< 30]
    rerank(query, topCandidates, service.tokenCache, service.lowerTextCache, service.lexical, service.corpus, service.cfg)
    for i in 0 ..< 30:
      candidates[i] = topCandidates[i]
  else:
    rerank(query, candidates, service.tokenCache, service.lowerTextCache, service.lexical, service.corpus, service.cfg)
  result = candidates

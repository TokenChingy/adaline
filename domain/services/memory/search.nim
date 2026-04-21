# Search memory service.
# Runs parallel semantic (LSH → HNSW) and lexical (inverted index)
# lanes, merges results via RRF, resolves chunks to parents,
# and applies term-coverage reranking.


import ./types
import ../../entities/memory
import ../../algorithms/sdr_encoder
import ../../algorithms/fingerprint_lsh
import ../../algorithms/hnsw_graph
import ../../algorithms/lexical_index
import ../../algorithms/rrf_merger
import ../../algorithms/reranker
import std/[tables, algorithm]

export types
export memory

proc search*(service: var MemoryService; query: string; k: int): seq[Memory] =
  let qfp = encodeSdr(query, service.cfg, service.corpus, isQuery = true)
  var semanticResults = newSeq[tuple[memoryId: uint64, score: float]]()
  if service.cfg.semanticSearchEnabled:
    let lshSeeds = queryLsh(service.lsh, addr qfp)
    if service.hnswEntryPoint != 0 or lshSeeds.len > 0:
      semanticResults = searchHnsw(service.storage.graphMem, service.storage.fpMem,
                                   lshSeeds, service.hnswEntryPoint, addr qfp, k, service.cfg)

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

  rerank(query, candidates, service.textCache, service.cfg)
  result = candidates

## Term-coverage reranker.
## Boosts merged candidates by the fraction of query tokens that appear
## in the candidate text, pushing exact lexical matches toward the top.


import ../entities/memory
import ../entities/config
import std/[tables, sets, strutils, algorithm]

proc tokenize*(text: string): seq[string] =
  let normalised = text.toLowerAscii()
  for token in normalised.split(AllChars - Letters - Digits):
    if token.len > 0:
      result.add(token)

proc coverageBoost*(qTokens: seq[string]; docTokens: HashSet[string]): float =
  if qTokens.len == 0:
    return 0.0
  var covered = 0
  for t in qTokens:
    if t in docTokens:
      covered.inc
  return float(covered) / float(qTokens.len)

proc rerank*(query: string; candidates: var seq[Memory]; tokenCache: Table[uint64, HashSet[string]]; cfg: EngineConfig) =
  if candidates.len == 0:
    return
  let qTokens = tokenize(query)
  for mem in candidates.mitems:
    let docTokens = tokenCache.getOrDefault(mem.id, initHashSet[string]())
    let cov = coverageBoost(qTokens, docTokens)
    mem.score = mem.score + cfg.rerankCoverageWeight * cov
  candidates.sort(proc(a, b: Memory): int =
    if a.score > b.score: return -1
    if a.score < b.score: return 1
    return 0
  )

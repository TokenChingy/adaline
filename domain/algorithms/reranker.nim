import ../entities/memory
import ../entities/config
import std/[tables, sets, strutils, algorithm]

proc tokenize*(text: string): seq[string] =
  let normalised = text.toLowerAscii()
  for token in normalised.split(AllChars - Letters - Digits):
    if token.len > 0:
      result.add(token)

proc coverageBoost*(query, doc: string): float =
  let qTokens = tokenize(query)
  let dTokens = tokenize(doc)
  if qTokens.len == 0:
    return 0.0
  var dSet = initHashSet[string]()
  for t in dTokens:
    dSet.incl(t)
  var covered = 0
  for t in qTokens:
    if t in dSet:
      covered.inc
  return float(covered) / float(qTokens.len)

proc rerank*(query: string; candidates: var seq[Memory]; textCache: Table[uint64, string]; cfg: EngineConfig) =
  if candidates.len == 0:
    return
  for mem in candidates.mitems:
    let docText = textCache.getOrDefault(mem.id, "")
    let cov = coverageBoost(query, docText)
    mem.score = mem.score + cfg.rerankCoverageWeight * cov
  candidates.sort(proc(a, b: Memory): int =
    if a.score > b.score: return -1
    if a.score < b.score: return 1
    return 0
  )

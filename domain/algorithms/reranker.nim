## Phrase-aware lexical reranker.
## Boosts merged candidates by a three-factor blend:
##   1. IDF-weighted unigram coverage
##   2. Exact substring phrase bonus
##   3. Soft document-length normalization


import ../entities/memory
import ../entities/config
import lexical_index
import corpus_index
import std/[tables, sets, strutils, algorithm, math]

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

proc buildBigrams*(tokens: seq[string]): seq[string] =
  if tokens.len < 2:
    return @[]
  result = newSeq[string](tokens.len - 1)
  for i in 0 ..< tokens.len - 1:
    result[i] = tokens[i] & " " & tokens[i + 1]

proc idfWeightedCoverage(
    qTokens: seq[string];
    docTokens: HashSet[string];
    corpus: CorpusIndex
): float =
  var sumIdf = 0.0
  var coveredIdf = 0.0
  for t in qTokens:
    let idf = corpus.idf.getOrDefault(t, 0.0)
    sumIdf += idf
    if t in docTokens:
      coveredIdf += idf
  if sumIdf <= 0.0:
    return coverageBoost(qTokens, docTokens)
  return coveredIdf / sumIdf

proc exactPhraseRatio*(qBigrams: seq[string]; lowerDocText: string): float =
  if qBigrams.len == 0:
    return 0.0
  var hits = 0
  for bg in qBigrams:
    if lowerDocText.find(bg) != -1:
      hits.inc
  return float(hits) / float(qBigrams.len)

proc lengthNormFactor(
    memId: uint64;
    lexical: LexicalIndex;
    numMemories: int
): float =
  if numMemories == 0 or lexical.totalCorpusTokens == 0:
    return 1.0
  let docLen = float(lexical.memLengths.getOrDefault(memId, 0))
  let avgLen = float(lexical.totalCorpusTokens) / float(numMemories)
  return 1.0 / (1.0 + ln(1.0 + docLen / max(1.0, avgLen)))

proc rerank*(
    query: string;
    candidates: var seq[Memory];
    tokenCache: Table[uint64, HashSet[string]];
    lowerTextCache: Table[uint64, string];
    lexical: LexicalIndex;
    corpus: CorpusIndex;
    cfg: EngineConfig
) =
  if candidates.len == 0:
    return

  let qTokens = tokenize(query)
  let qBigrams = buildBigrams(qTokens)

  for mem in candidates.mitems:
    let docTokens = tokenCache.getOrDefault(mem.id, initHashSet[string]())
    let lowerDoc  = lowerTextCache.getOrDefault(mem.id, "")

    let idfCov    = idfWeightedCoverage(qTokens, docTokens, corpus)
    let phraseRat = exactPhraseRatio(qBigrams, lowerDoc)
    let lenFac    = lengthNormFactor(mem.id, lexical, corpus.numMemories)

    let boost = cfg.rerankIdfWeight    * idfCov +
                cfg.rerankPhraseWeight * phraseRat +
                cfg.rerankLenWeight    * lenFac

    mem.score = mem.score + boost

  candidates.sort(proc(a, b: Memory): int =
    if a.score > b.score: return -1
    if a.score < b.score: return 1
    return 0
  )

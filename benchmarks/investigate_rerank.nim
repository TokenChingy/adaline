import std/[os, strutils, json, times, tables, sets, sequtils, algorithm, math]
import ../domain/entities/config
import ../domain/services/memory_service
import ../domain/entities/config

const dataDir = "benchmarks/scifact"

proc mean(values: seq[float]): float =
  if values.len == 0: return 0.0
  var s = 0.0
  for v in values: s += v
  return s / float(values.len)

proc loadCorpus(): seq[tuple[id, text: string]] =
  let f = open(dataDir / "corpus.jsonl")
  defer: f.close()
  for line in f.lines:
    let j = parseJson(line)
    let id = j["_id"].getStr()
    let title = j["title"].getStr()
    let text = j["text"].getStr()
    let fullText = (if title.len > 0: title & " " & text else: text)
    result.add((id, fullText))

proc loadQueries(): seq[tuple[id, text: string]] =
  let f = open(dataDir / "queries.jsonl")
  defer: f.close()
  for line in f.lines:
    let j = parseJson(line)
    result.add((j["_id"].getStr(), j["text"].getStr()))

proc loadQrels(): Table[string, HashSet[string]] =
  result = initTable[string, HashSet[string]]()
  let f = open(dataDir / "qrels" / "test.tsv")
  defer: f.close()
  var first = true
  for line in f.lines:
    if first:
      first = false
      continue
    let parts = line.split('\t')
    if parts.len < 3: continue
    let qid = parts[0].strip()
    let cid = parts[1].strip()
    if not result.hasKey(qid):
      result[qid] = initHashSet[string]()
    result[qid].incl(cid)

proc tokenize(text: string): seq[string] =
  let normalised = text.toLowerAscii()
  for token in normalised.split(AllChars - Letters - Digits):
    if token.len > 0:
      result.add(token)

proc computeRecall(results: Table[string, seq[string]], qrels: Table[string, HashSet[string]], k: int): float =
  var scores: seq[float]
  for qid, relevant in qrels.pairs:
    if not results.hasKey(qid): continue
    let topK = results[qid][0 ..< min(k, results[qid].len)]
    var found = 0
    for cid in topK:
      if relevant.contains(cid): inc found
    if relevant.len > 0:
      scores.add(float(found) / float(relevant.len))
  return mean(scores)

proc computePrecision(results: Table[string, seq[string]], qrels: Table[string, HashSet[string]], k: int): float =
  var scores: seq[float]
  for qid, relevant in qrels.pairs:
    if not results.hasKey(qid): continue
    let topK = results[qid][0 ..< min(k, results[qid].len)]
    var found = 0
    for cid in topK:
      if relevant.contains(cid): inc found
    scores.add(float(found) / float(topK.len))
  return mean(scores)

proc computeMrr(results: Table[string, seq[string]], qrels: Table[string, HashSet[string]]): float =
  var scores: seq[float]
  for qid, relevant in qrels.pairs:
    if not results.hasKey(qid): continue
    var rank = 0
    for i, cid in results[qid]:
      if relevant.contains(cid):
        rank = i + 1
        break
    if rank > 0:
      scores.add(1.0 / float(rank))
    else:
      scores.add(0.0)
  return mean(scores)

proc computeNdcg(results: Table[string, seq[string]], qrels: Table[string, HashSet[string]], k: int): float =
  var scores: seq[float]
  for qid, relevant in qrels.pairs:
    if not results.hasKey(qid): continue
    let topK = results[qid][0 ..< min(k, results[qid].len)]
    var dcg = 0.0
    for i, cid in topK:
      if relevant.contains(cid):
        dcg += 1.0 / log2(float(i + 2))
    var idcg = 0.0
    let relCount = min(relevant.len, k)
    for i in 0 ..< relCount:
      idcg += 1.0 / log2(float(i + 2))
    if idcg > 0:
      scores.add(dcg / idcg)
    else:
      scores.add(0.0)
  return mean(scores)

# ─── Reranking Scorers ───

proc termCoverageScore(query, doc: string): float =
  let qTokens = tokenize(query)
  let dTokens = tokenize(doc)
  var dSet = initHashSet[string]()
  for t in dTokens: dSet.incl(t)
  var covered = 0
  for t in qTokens:
    if t in dSet: covered.inc
  if qTokens.len > 0:
    return float(covered) / float(qTokens.len)
  return 0.0

proc phraseMatchScore(query, doc: string): float =
  let q = query.toLowerAscii()
  let d = doc.toLowerAscii()
  if q in d:
    return 1.0
  # Check for partial phrase matches (at least half the query tokens in order)
  let qTokens = tokenize(query)
  if qTokens.len >= 3:
    for i in 0 ..< qTokens.len - 1:
      let bigram = qTokens[i] & " " & qTokens[i + 1]
      if bigram in d:
        return 0.5
  return 0.0

proc termProximityScore(query, doc: string): float =
  let qTokens = tokenize(query)
  let dTokens = tokenize(doc)
  if qTokens.len < 2 or dTokens.len == 0:
    return 0.0
  var qIdx = initTable[string, seq[int]]()
  for i, t in dTokens:
    qIdx.mgetOrPut(t, @[]).add(i)
  var minSpan = high(int)
  for firstPos in qIdx.getOrDefault(qTokens[0], @[]):
    var maxPos = firstPos
    var missing = false
    for qi in 1 ..< qTokens.len:
      let positions = qIdx.getOrDefault(qTokens[qi], @[])
      var bestPos = high(int)
      for p in positions:
        if p > firstPos and p < bestPos:
          bestPos = p
      if bestPos == high(int):
        missing = true
        break
      if bestPos > maxPos:
        maxPos = bestPos
    if not missing:
      let span = maxPos - firstPos + 1
      if span < minSpan:
        minSpan = span
  if minSpan == high(int):
    return 0.0
  # Normalize: span of len(query) is perfect (score 1.0)
  return float(qTokens.len) / float(minSpan)

# ─── Main Investigation ───

proc main() =
  let corpus = loadCorpus()
  let queries = loadQueries()
  let qrels = loadQrels()

  echo "Dataset loaded: ", corpus.len, " docs, ", queries.len, " queries, ", qrels.len, " qrels"

  let cfg = defaultEngineConfig()
  let benchDir = getCurrentDir() / "benchmarks" / "data_rerank"
  removeDir(benchDir)
  var service = initMemoryService(benchDir, cfg)

  var idMap = initTable[uint64, string]()
  var contentMap = initTable[string, string]()
  for doc in corpus:
    let memId = service.insert(doc.text)
    idMap[memId] = doc.id
    contentMap[doc.id] = doc.text

  # Collect base top-10 results for every query
  var baseTop10 = initTable[string, seq[tuple[docId: string, baseScore: float]]]()
  for q in queries:
    let res = service.search(q.text, 10)
    var ranked: seq[tuple[docId: string, baseScore: float]] = @[]
    for mem in res:
      let docId = idMap.getOrDefault(mem.id, "")
      if docId.len > 0:
        ranked.add((docId, mem.score))
    baseTop10[q.id] = ranked

  # Helper to evaluate a reranking strategy
  proc evaluate(name: string; reranker: proc(qid, query: string; candidates: seq[tuple[docId: string, baseScore: float]]): seq[tuple[docId: string, score: float]]) =
    var results = initTable[string, seq[string]]()
    for q in queries:
      if not baseTop10.hasKey(q.id): continue
      let reranked = reranker(q.id, q.text, baseTop10[q.id])
      results[q.id] = reranked.mapIt(it.docId)
    echo "\n--- ", name, " ---"
    echo "  Recall@5:    ", formatFloat(computeRecall(results, qrels, 5), ffDecimal, 4)
    echo "  Precision@5: ", formatFloat(computePrecision(results, qrels, 5), ffDecimal, 4)
    echo "  Recall@10:   ", formatFloat(computeRecall(results, qrels, 10), ffDecimal, 4)
    echo "  Precision@10:", formatFloat(computePrecision(results, qrels, 10), ffDecimal, 4)
    echo "  MRR:         ", formatFloat(computeMrr(results, qrels), ffDecimal, 4)
    echo "  nDCG@10:     ", formatFloat(computeNdcg(results, qrels, 10), ffDecimal, 4)

  # 1. Baseline (no reranking)
  evaluate("Baseline (RRF only)") do (qid, query: string; candidates: seq[tuple[docId: string, baseScore: float]]) -> seq[tuple[docId: string, score: float]]:
    result = candidates.mapIt((it.docId, it.baseScore))

  # 2. Term coverage boost
  evaluate("+ Term Coverage Boost") do (qid, query: string; candidates: seq[tuple[docId: string, baseScore: float]]) -> seq[tuple[docId: string, score: float]]:
    var scored = newSeq[tuple[docId: string, score: float]]()
    for (docId, baseScore) in candidates:
      let cov = termCoverageScore(query, contentMap.getOrDefault(docId, ""))
      let newScore = baseScore + 0.5 * cov
      scored.add((docId, newScore))
    scored.sort(proc(a, b: auto): int =
      if a.score > b.score: return -1
      if a.score < b.score: return 1
      return 0
    )
    result = scored

  # 3. Phrase match boost
  evaluate("+ Phrase Match Boost") do (qid, query: string; candidates: seq[tuple[docId: string, baseScore: float]]) -> seq[tuple[docId: string, score: float]]:
    var scored = newSeq[tuple[docId: string, score: float]]()
    for (docId, baseScore) in candidates:
      let phrase = phraseMatchScore(query, contentMap.getOrDefault(docId, ""))
      let newScore = baseScore + 1.0 * phrase
      scored.add((docId, newScore))
    scored.sort(proc(a, b: auto): int =
      if a.score > b.score: return -1
      if a.score < b.score: return 1
      return 0
    )
    result = scored

  # 4. Term proximity boost
  evaluate("+ Term Proximity Boost") do (qid, query: string; candidates: seq[tuple[docId: string, baseScore: float]]) -> seq[tuple[docId: string, score: float]]:
    var scored = newSeq[tuple[docId: string, score: float]]()
    for (docId, baseScore) in candidates:
      let prox = termProximityScore(query, contentMap.getOrDefault(docId, ""))
      let newScore = baseScore + 0.3 * prox
      scored.add((docId, newScore))
    scored.sort(proc(a, b: auto): int =
      if a.score > b.score: return -1
      if a.score < b.score: return 1
      return 0
    )
    result = scored

  # 5. Combined reranker (coverage + phrase + proximity)
  evaluate("+ Combined (Coverage + Phrase + Proximity)") do (qid, query: string; candidates: seq[tuple[docId: string, baseScore: float]]) -> seq[tuple[docId: string, score: float]]:
    var scored = newSeq[tuple[docId: string, score: float]]()
    for (docId, baseScore) in candidates:
      let content = contentMap.getOrDefault(docId, "")
      let cov = termCoverageScore(query, content)
      let phrase = phraseMatchScore(query, content)
      let prox = termProximityScore(query, content)
      let newScore = baseScore + 0.5 * cov + 1.0 * phrase + 0.3 * prox
      scored.add((docId, newScore))
    scored.sort(proc(a, b: auto): int =
      if a.score > b.score: return -1
      if a.score < b.score: return 1
      return 0
    )
    result = scored

when isMainModule:
  main()

## BEIR ablation & lane-contribution runner.
## Runs SciFact with semantic-only, lexical-only, and dual-lane configs,
## and reports lane-origin breakdown for the dual-lane run.


import std/[os, strutils, json, times, tables, sets, sequtils, algorithm, math]
import ../domain/services/memory/init
import ../domain/services/memory/insert
import ../domain/services/memory/search
import ../domain/entities/config
import ../domain/entities/fingerprint
import ../domain/algorithms/sdr_encoder
import ../domain/algorithms/fingerprint_lsh
import ../domain/algorithms/weighted_jaccard
import ../domain/algorithms/lexical_index
import ../domain/algorithms/rrf_merger
import ../infrastructure/mmapped_storage

proc loadCorpus(dataDir: string): seq[tuple[id, text: string]] =
  let f = open(dataDir / "corpus.jsonl")
  defer: f.close()
  for line in f.lines:
    let j = parseJson(line)
    let id = j["_id"].getStr()
    let title = j["title"].getStr()
    let text = j["text"].getStr()
    let fullText = (if title.len > 0: title & " " & text else: text)
    result.add((id, fullText))

proc loadQueries(dataDir: string): seq[tuple[id, text: string]] =
  let f = open(dataDir / "queries.jsonl")
  defer: f.close()
  for line in f.lines:
    let j = parseJson(line)
    result.add((j["_id"].getStr(), j["text"].getStr()))

proc loadQrels(dataDir: string): Table[string, HashSet[string]] =
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

proc percentile(sortedValues: seq[float], p: float): float =
  if sortedValues.len == 0: return 0.0
  let idx = int(float(sortedValues.high) * p)
  return sortedValues[idx]

proc mean(values: seq[float]): float =
  if values.len == 0: return 0.0
  var s = 0.0
  for v in values: s += v
  return s / float(values.len)

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

proc computeMap(results: Table[string, seq[string]], qrels: Table[string, HashSet[string]]): float =
  var scores: seq[float]
  for qid, relevant in qrels.pairs:
    if not results.hasKey(qid): continue
    var found = 0
    var precSum = 0.0
    for i, cid in results[qid]:
      if relevant.contains(cid):
        inc found
        precSum += float(found) / float(i + 1)
    if relevant.len > 0:
      scores.add(precSum / float(relevant.len))
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

proc runAblatedBenchmark*(semanticEnabled, lexicalEnabled: bool): tuple[
    recall1, recall5, recall10, recall100, mrr, mapVal, ndcg10: float] =
  let dataDir = "benchmarks/scifact"
  let corpus = loadCorpus(dataDir)
  let queries = loadQueries(dataDir)
  let qrels = loadQrels(dataDir)

  var cfg = defaultEngineConfig()
  cfg.semanticSearchEnabled = semanticEnabled
  cfg.lexicalSearchEnabled = lexicalEnabled

  let benchDir = getCurrentDir() / "benchmarks" / "data" / "scifact_ablation"
  removeDir(benchDir)
  var service = initMemoryService(benchDir, cfg)

  var idMap = initTable[uint64, string]()
  var reverseIdMap = initTable[string, uint64]()

  for doc in corpus:
    let memId = service.insert(doc.text)
    idMap[memId] = doc.id
    reverseIdMap[doc.id] = memId

  var allResults = initTable[string, seq[string]]()
  for q in queries:
    let res = service.search(q.text, 100)
    allResults[q.id] = res.mapIt(idMap.getOrDefault(it.id, ""))

  result.recall1 = computeRecall(allResults, qrels, 1)
  result.recall5 = computeRecall(allResults, qrels, 5)
  result.recall10 = computeRecall(allResults, qrels, 10)
  result.recall100 = computeRecall(allResults, qrels, 100)
  result.mrr = computeMrr(allResults, qrels)
  result.mapVal = computeMap(allResults, qrels)
  result.ndcg10 = computeNdcg(allResults, qrels, 10)

proc runLaneContribution*(): tuple[semanticOnly, lexicalOnly, both: float] =
  let dataDir = "benchmarks/scifact"
  let corpus = loadCorpus(dataDir)
  let queries = loadQueries(dataDir)

  let cfg = defaultEngineConfig()
  let benchDir = getCurrentDir() / "benchmarks" / "data" / "scifact_contrib"
  removeDir(benchDir)
  var service = initMemoryService(benchDir, cfg)

  var reverseIdMap = initTable[string, uint64]()
  for doc in corpus:
    let memId = service.insert(doc.text)
    reverseIdMap[doc.id] = memId

  var semOnlyCount = 0
  var lexOnlyCount = 0
  var bothCount = 0
  var totalCount = 0

  for q in queries:
    let qfp = encodeSdr(q.text, service.cfg, service.corpus, isQuery = true)

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
      let topK = min(100, scored.len)
      for i in 0 ..< topK:
        semanticResults.add((scored[i].memoryId, scored[i].score))

    var lexicalResults = newSeq[tuple[memoryId: uint64, score: float]]()
    if service.cfg.lexicalSearchEnabled:
      lexicalResults = searchLexical(service.lexical, q.text, 100)

    let merged = mergeRrf(semanticResults, lexicalResults, 100, service.cfg.rrfK)

    let topK = min(100, merged.len)
    for i in 0 ..< topK:
      let mid = merged[i].memoryId
      var inSem = false
      var inLex = false
      for s in semanticResults:
        if s.memoryId == mid:
          inSem = true
          break
      for l in lexicalResults:
        if l.memoryId == mid:
          inLex = true
          break
      if inSem and inLex:
        bothCount.inc
      elif inSem:
        semOnlyCount.inc
      elif inLex:
        lexOnlyCount.inc
      totalCount.inc

  if totalCount > 0:
    result.semanticOnly = float(semOnlyCount) / float(totalCount) * 100.0
    result.lexicalOnly = float(lexOnlyCount) / float(totalCount) * 100.0
    result.both = float(bothCount) / float(totalCount) * 100.0

proc main() =
  echo "\n=== SciFact Lane Ablation ==="
  echo "\n--- Dual Lane (baseline) ---"
  let dual = runAblatedBenchmark(true, true)
  echo "Recall@1:  ", formatFloat(dual.recall1, ffDecimal, 4)
  echo "Recall@5:  ", formatFloat(dual.recall5, ffDecimal, 4)
  echo "Recall@10: ", formatFloat(dual.recall10, ffDecimal, 4)
  echo "Recall@100:", formatFloat(dual.recall100, ffDecimal, 4)
  echo "MRR:       ", formatFloat(dual.mrr, ffDecimal, 4)
  echo "MAP:       ", formatFloat(dual.mapVal, ffDecimal, 4)
  echo "nDCG@10:   ", formatFloat(dual.ndcg10, ffDecimal, 4)

  echo "\n--- Semantic Only ---"
  let sem = runAblatedBenchmark(true, false)
  echo "Recall@1:  ", formatFloat(sem.recall1, ffDecimal, 4)
  echo "Recall@5:  ", formatFloat(sem.recall5, ffDecimal, 4)
  echo "Recall@10: ", formatFloat(sem.recall10, ffDecimal, 4)
  echo "Recall@100:", formatFloat(sem.recall100, ffDecimal, 4)
  echo "MRR:       ", formatFloat(sem.mrr, ffDecimal, 4)
  echo "MAP:       ", formatFloat(sem.mapVal, ffDecimal, 4)
  echo "nDCG@10:   ", formatFloat(sem.ndcg10, ffDecimal, 4)

  echo "\n--- Lexical Only ---"
  let lex = runAblatedBenchmark(false, true)
  echo "Recall@1:  ", formatFloat(lex.recall1, ffDecimal, 4)
  echo "Recall@5:  ", formatFloat(lex.recall5, ffDecimal, 4)
  echo "Recall@10: ", formatFloat(lex.recall10, ffDecimal, 4)
  echo "Recall@100:", formatFloat(lex.recall100, ffDecimal, 4)
  echo "MRR:       ", formatFloat(lex.mrr, ffDecimal, 4)
  echo "MAP:       ", formatFloat(lex.mapVal, ffDecimal, 4)
  echo "nDCG@10:   ", formatFloat(lex.ndcg10, ffDecimal, 4)

  echo "\n=== Lane Contribution (top-100 merged results) ==="
  let contrib = runLaneContribution()
  echo "Semantic only: ", formatFloat(contrib.semanticOnly, ffDecimal, 1), "%"
  echo "Lexical only:  ", formatFloat(contrib.lexicalOnly, ffDecimal, 1), "%"
  echo "Both lanes:    ", formatFloat(contrib.both, ffDecimal, 1), "%"

when isMainModule:
  main()

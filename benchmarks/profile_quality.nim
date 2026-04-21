import std/[os, times, tables, strformat, json, strutils, sets, sequtils, algorithm, math]
import ../domain/services/memory_service
import ../domain/entities/config

proc myMean(values: seq[float]): float =
  if values.len == 0: return 0.0
  var s = 0.0
  for v in values: s += v
  return s / float(values.len)

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
  return myMean(scores)

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
  return myMean(scores)

proc runQuality(label: string; maxNeighbors, efConstruction, efSearch: int) =
  let dataDir = "benchmarks/scifact"
  let corpus = loadCorpus(dataDir)
  let queries = loadQueries(dataDir)
  let qrels = loadQrels(dataDir)

  var cfg = defaultEngineConfig()
  cfg.hnswEnabled = true
  cfg.hnswMaxNeighbors = maxNeighbors
  cfg.hnswEfConstruction = efConstruction
  cfg.hnswEfSearch = efSearch

  let benchDir = getCurrentDir() / "benchmarks" / "data" / ("quality_" & label)
  removeDir(benchDir)
  var svc = initMemoryService(benchDir, cfg)

  var idMap = initTable[uint64, string]()
  let tInsertStart = cpuTime()
  for doc in corpus:
    let memId = svc.insert(doc.text)
    idMap[memId] = doc.id
  let tInsert = cpuTime() - tInsertStart

  var allResults = initTable[string, seq[string]]()
  let tSearchStart = cpuTime()
  for q in queries:
    let res = svc.search(q.text, 100)
    allResults[q.id] = res.mapIt(idMap.getOrDefault(it.id, ""))
  let tSearch = cpuTime() - tSearchStart

  echo &"\n=== {label} (M={maxNeighbors}, efC={efConstruction}, efS={efSearch}) ==="
  echo &"Insert: {tInsert:.2f}s ({corpus.len.float / tInsert:.1f} docs/s)"
  echo &"Search: {tSearch:.2f}s ({queries.len.float / tSearch:.1f} q/s)"
  echo &"R@100:  {computeRecall(allResults, qrels, 100):.4f}"
  echo &"nDCG@10:{computeNdcg(allResults, qrels, 10):.4f}"

proc main() =
  runQuality("M32_ef200_es64", 32, 200, 64)
  runQuality("M16_ef200_es64", 16, 200, 64)
  runQuality("M16_ef100_es64", 16, 100, 64)
  runQuality("M16_ef50_es64", 16, 50, 64)
  runQuality("M8_ef100_es64", 8, 100, 64)
  runQuality("M8_ef50_es64", 8, 50, 64)

when isMainModule:
  main()

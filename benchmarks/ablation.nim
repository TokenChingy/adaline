import std/[os, strutils, json, times, tables, sets, sequtils, algorithm, math]
import ../domain/services/memory/types
import ../domain/services/memory/init
import ../domain/services/memory/insert
import ../domain/services/memory/search
import ../domain/entities/config

# Re-use BEIR data loading helpers
const
  BeirBaseUrl = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"
  DefaultDataset = "scifact"

proc datasetUrl(name: string): string =
  BeirBaseUrl & "/" & name & ".zip"

proc datasetDir(name: string): string =
  "benchmarks/" & name

proc zipPath(name: string): string =
  "benchmarks/" & name & ".zip"

proc ensureDataset(name: string) =
  let dir = datasetDir(name)
  if dirExists(dir): return
  echo "Downloading ", name, " dataset..."
  let url = datasetUrl(name)
  let zip = zipPath(name)
  let wgetCmd = "wget -q -O " & zip & " " & url
  let curlCmd = "curl -L -o " & zip & " " & url
  if execShellCmd(wgetCmd) != 0:
    if execShellCmd(curlCmd) != 0:
      raise newException(IOError, "Failed to download dataset: " & name)
  echo "Extracting dataset..."
  if execShellCmd("unzip -q -o " & zip & " -d benchmarks/") != 0:
    raise newException(IOError, "Failed to extract dataset: " & name)

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

proc runMode*(datasetName, mode: string): Table[string, seq[string]] =
  ensureDataset(datasetName)
  let dataDir = datasetDir(datasetName)
  let corpus = loadCorpus(dataDir)
  let queries = loadQueries(dataDir)

  var cfg = defaultEngineConfig()
  case mode
  of "semantic":
    cfg.semanticSearchEnabled = true
    cfg.lexicalSearchEnabled = false
  of "lexical":
    cfg.semanticSearchEnabled = false
    cfg.lexicalSearchEnabled = true
  of "combined":
    cfg.semanticSearchEnabled = true
    cfg.lexicalSearchEnabled = true
  else:
    raise newException(ValueError, "Unknown mode: " & mode)

  let benchDir = getCurrentDir() / "benchmarks" / "data" / (datasetName & "_" & mode)
  removeDir(benchDir)
  var service = initMemoryService(benchDir, cfg)

  var idMap = initTable[uint64, string]()
  var reverseIdMap = initTable[string, uint64]()

  for i, doc in corpus:
    let memId = service.insert(doc.text)
    idMap[memId] = doc.id
    reverseIdMap[doc.id] = memId

  var queryTimes = newSeq[float](queries.len)
  var allResults = initTable[string, seq[string]]()
  let tQueryStart = cpuTime()
  for i, q in queries:
    let t0 = cpuTime()
    let res = service.search(q.text, 100)
    queryTimes[i] = cpuTime() - t0
    allResults[q.id] = res.mapIt(idMap.getOrDefault(it.id, ""))
  let totalQuery = cpuTime() - tQueryStart

  sort(queryTimes)

  echo "\n=== ", datasetName, " | ", mode, " ==="
  echo "Query Speed (top-100):"
  echo "  Total:     ", formatFloat(totalQuery, ffDecimal, 4), " s"
  echo "  Throughput:", formatFloat(float(queries.len) / totalQuery, ffDecimal, 2), " queries/s"
  echo "  P50:       ", formatFloat(percentile(queryTimes, 0.5) * 1000, ffDecimal, 4), " ms"
  echo "  P95:       ", formatFloat(percentile(queryTimes, 0.95) * 1000, ffDecimal, 4), " ms"
  echo "  P99:       ", formatFloat(percentile(queryTimes, 0.99) * 1000, ffDecimal, 4), " ms"

  return allResults

proc reportQuality(allResults: Table[string, seq[string]], qrels: Table[string, HashSet[string]]) =
  echo "Recall Metrics:"
  echo "  Recall@1:   ", formatFloat(computeRecall(allResults, qrels, 1), ffDecimal, 4)
  echo "  Recall@5:   ", formatFloat(computeRecall(allResults, qrels, 5), ffDecimal, 4)
  echo "  Recall@10:  ", formatFloat(computeRecall(allResults, qrels, 10), ffDecimal, 4)
  echo "  Recall@100: ", formatFloat(computeRecall(allResults, qrels, 100), ffDecimal, 4)
  echo "  Precision@1:  ", formatFloat(computePrecision(allResults, qrels, 1), ffDecimal, 4)
  echo "  Precision@5:  ", formatFloat(computePrecision(allResults, qrels, 5), ffDecimal, 4)
  echo "  Precision@10: ", formatFloat(computePrecision(allResults, qrels, 10), ffDecimal, 4)
  echo "  Precision@100:", formatFloat(computePrecision(allResults, qrels, 100), ffDecimal, 4)
  echo "  MRR:        ", formatFloat(computeMrr(allResults, qrels), ffDecimal, 4)
  echo "  MAP:        ", formatFloat(computeMap(allResults, qrels), ffDecimal, 4)
  echo "  nDCG@10:    ", formatFloat(computeNdcg(allResults, qrels, 10), ffDecimal, 4)

proc runAblation*(datasetName: string) =
  ensureDataset(datasetName)
  let dataDir = datasetDir(datasetName)
  let qrels = loadQrels(dataDir)

  echo "Dataset: ", datasetName
  echo "Queries with judgments: ", qrels.len

  let semanticResults = runMode(datasetName, "semantic")
  reportQuality(semanticResults, qrels)

  let lexicalResults = runMode(datasetName, "lexical")
  reportQuality(lexicalResults, qrels)

  let combinedResults = runMode(datasetName, "combined")
  reportQuality(combinedResults, qrels)

proc printUsage() =
  echo """
Adaline Ablation Benchmark

Usage:
  ablation <dataset> [mode]

Modes:
  semantic   - semantic lane only (LSH + HNSW)
  lexical    - lexical lane only (QLM)
  combined   - both lanes with RRF (default for ablation)
  all        - run all three and print comparison

Examples:
  ablation scifact semantic
  ablation scifact all
"""

proc main() =
  let args = commandLineParams()
  if args.len == 0 or args[0] in ["help", "--help", "-h"]:
    printUsage()
    return

  let datasetName = args[0]
  let mode = if args.len > 1: args[1] else: "all"

  if mode == "all":
    runAblation(datasetName)
  else:
    ensureDataset(datasetName)
    let dataDir = datasetDir(datasetName)
    let qrels = loadQrels(dataDir)
    let results = runMode(datasetName, mode)
    reportQuality(results, qrels)

when isMainModule:
  main()

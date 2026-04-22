## BEIR benchmark runner.
## Downloads, indexes, and queries BEIR datasets (SciFact, NFCorpus,
## ArguAna, MS MARCO). Reports insert throughput, query latency,
## and retrieval quality metrics (Recall, Precision, MRR, MAP, nDCG).


import std/[os, strutils, json, times, tables, sets, sequtils, algorithm, math]
import ../domain/services/memory/init
import ../domain/services/memory/insert
import ../domain/services/memory/search
import ../domain/entities/config

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

proc runBenchmark*(datasetName: string) =
  ensureDataset(datasetName)
  let dataDir = datasetDir(datasetName)
  let corpus = loadCorpus(dataDir)
  let queries = loadQueries(dataDir)
  let qrels = loadQrels(dataDir)

  echo "Dataset: ", datasetName
  echo "  Corpus:   ", corpus.len
  echo "  Queries:  ", queries.len
  echo "  Qrels:    ", qrels.len

  let cfg = defaultEngineConfig()
  let benchDir = getCurrentDir() / "benchmarks" / "data" / datasetName
  removeDir(benchDir)
  var service = initMemoryService(benchDir, cfg)

  var idMap = initTable[uint64, string]()
  var reverseIdMap = initTable[string, uint64]()

  var insertTimes = newSeq[float](corpus.len)
  let tInsertStart = cpuTime()
  for i, doc in corpus:
    let t0 = cpuTime()
    let memId = service.insert(doc.text)
    insertTimes[i] = cpuTime() - t0
    idMap[memId] = doc.id
    reverseIdMap[doc.id] = memId
  let totalInsert = cpuTime() - tInsertStart

  var queryTimes = newSeq[float](queries.len)
  var allResults = initTable[string, seq[string]]()
  let tQueryStart = cpuTime()
  for i, q in queries:
    let t0 = cpuTime()
    let res = service.search(q.text, 100)
    queryTimes[i] = cpuTime() - t0
    allResults[q.id] = res.mapIt(idMap.getOrDefault(it.id, ""))
  let totalQuery = cpuTime() - tQueryStart

  sort(insertTimes)
  sort(queryTimes)

  echo "\n=== Benchmark Results: ", datasetName, " ==="
  echo "\nFull Insert (storage + SDR + LSH + Lexical):"
  echo "  Total:     ", formatFloat(totalInsert, ffDecimal, 4), " s"
  echo "  Throughput:", formatFloat(float(corpus.len) / totalInsert, ffDecimal, 2), " docs/s"
  echo "  P50:       ", formatFloat(percentile(insertTimes, 0.5) * 1000, ffDecimal, 4), " ms"
  echo "  P95:       ", formatFloat(percentile(insertTimes, 0.95) * 1000, ffDecimal, 4), " ms"
  echo "  P99:       ", formatFloat(percentile(insertTimes, 0.99) * 1000, ffDecimal, 4), " ms"

  echo "\nQuery Speed (top-100):"
  echo "  Total:     ", formatFloat(totalQuery, ffDecimal, 4), " s"
  echo "  Throughput:", formatFloat(float(queries.len) / totalQuery, ffDecimal, 2), " queries/s"
  echo "  P50:       ", formatFloat(percentile(queryTimes, 0.5) * 1000, ffDecimal, 4), " ms"
  echo "  P95:       ", formatFloat(percentile(queryTimes, 0.95) * 1000, ffDecimal, 4), " ms"
  echo "  P99:       ", formatFloat(percentile(queryTimes, 0.99) * 1000, ffDecimal, 4), " ms"

  echo "\nRecall Metrics:"
  echo "  Recall@1:  ", formatFloat(computeRecall(allResults, qrels, 1), ffDecimal, 4)
  echo "  Recall@5:  ", formatFloat(computeRecall(allResults, qrels, 5), ffDecimal, 4)
  echo "  Recall@10: ", formatFloat(computeRecall(allResults, qrels, 10), ffDecimal, 4)
  echo "  Recall@100:", formatFloat(computeRecall(allResults, qrels, 100), ffDecimal, 4)
  echo "  Precision@1:  ", formatFloat(computePrecision(allResults, qrels, 1), ffDecimal, 4)
  echo "  Precision@5:  ", formatFloat(computePrecision(allResults, qrels, 5), ffDecimal, 4)
  echo "  Precision@10: ", formatFloat(computePrecision(allResults, qrels, 10), ffDecimal, 4)
  echo "  Precision@100:", formatFloat(computePrecision(allResults, qrels, 100), ffDecimal, 4)
  echo "  MRR:       ", formatFloat(computeMrr(allResults, qrels), ffDecimal, 4)
  echo "  MAP:       ", formatFloat(computeMap(allResults, qrels), ffDecimal, 4)
  echo "  nDCG@10:   ", formatFloat(computeNdcg(allResults, qrels, 10), ffDecimal, 4)

proc printUsage() =
  echo """
Adaline BEIR Benchmark

Usage:
  beir <dataset>

Supported datasets:
  scifact   - Scientific fact verification (default, ~5K docs)
  nfcorpus  - NFCorpus medical literature (~3.6K docs)
  arguana   - Argument retrieval (~8.7K docs)
  fiqa      - Financial QA (~57K docs)
  msmarco   - MS MARCO passage retrieval (~8.8M docs, 1GB+ download)

Examples:
  beir scifact
  beir nfcorpus
  beir msmarco
"""

proc main() =
  let args = commandLineParams()
  let datasetName = if args.len > 0: args[0] else: DefaultDataset

  if datasetName in ["help", "--help", "-h"]:
    printUsage()
    return

  if datasetName == "arguana":
    if not dirExists("benchmarks/arguana"):
      echo "Error: benchmarks/arguana/ not found"
      return

  runBenchmark(datasetName)

when isMainModule:
  main()

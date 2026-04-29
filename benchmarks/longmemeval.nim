## LongMemEval-S benchmark runner.
## Evaluates conversational memory retrieval over ~500 questions
## with multi-session context. Measures R@1, R@5, R@10 per category.


import std/[os, strutils, json, tables, algorithm, math, times]
import ../domain/services/memory/init
import ../domain/services/memory/insert
import ../domain/services/memory/search
import ../domain/entities/config

const
  DatasetPath = "benchmarks/data/longmemeval_s_cleaned.json"
  DefaultTopK = 10

type
  Question = object
    id: string
    qType: string
    question: string
    answer: string
    answerSessionIds: seq[string]
    haystackSessions: seq[tuple[sessionId: string, text: string]]

proc loadDataset(): seq[Question] =
  let j = parseJson(readFile(DatasetPath))
  for item in j:
    var q = Question()
    q.id = item["question_id"].getStr()
    q.qType = item["question_type"].getStr()
    q.question = item["question"].getStr()
    q.answer = item["answer"].getStr()
    for sid in item["answer_session_ids"]:
      q.answerSessionIds.add(sid.getStr())

    let sessionIds = item["haystack_session_ids"]
    let sessions = item["haystack_sessions"]
    for i in 0 ..< sessionIds.len:
      let sid = sessionIds[i].getStr()
      var textParts: seq[string]
      for turn in sessions[i]:
        let role = turn["role"].getStr()
        let content = turn["content"].getStr()
        textParts.add(role & ": " & content)
      let fullText = textParts.join("\n")
      q.haystackSessions.add((sid, fullText))

    result.add(q)

proc runBenchmark*(topK: int = DefaultTopK) =
  if not fileExists(DatasetPath):
    echo "Dataset not found: ", DatasetPath
    echo "Download it with:"
    echo "  python3 -c \"from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='xiaowu0162/longmemeval-cleaned', filename='longmemeval_s_cleaned.json', repo_type='dataset', local_dir='benchmarks/data')\""
    quit(1)

  let questions = loadDataset()
  echo "Loaded ", questions.len, " questions"

  var typeCounts = initTable[string, int]()
  for q in questions:
    typeCounts.mgetOrPut(q.qType, 0).inc
  echo "Question types:"
  for t, c in typeCounts:
    echo "  ", t, ": ", c

  var r1Hits = 0
  var r5Hits = 0
  var r10Hits = 0
  var typeR1 = initTable[string, tuple[hits, total: int]]()
  var typeR5 = initTable[string, tuple[hits, total: int]]()
  var typeR10 = initTable[string, tuple[hits, total: int]]()

  var totalSessions = 0
  var totalInsertTime = 0.0
  var totalQueryTime = 0.0
  var queryTimes = newSeq[float]()
  var sessionCounts = newSeq[int]()

  let cfg = defaultEngineConfig()

  for i, q in questions:
    if i mod 50 == 0:
      echo "Processing ", i, "/", questions.len, "..."

    let benchDir = getCurrentDir() / "benchmarks" / "data" / "longmemeval_run" / q.id
    removeDir(benchDir)
    var service = initMemoryService(benchDir, cfg)

    var sidToMid = initTable[string, uint64]()
    var midToSid = initTable[uint64, string]()

    let tInsertStart = cpuTime()
    for (sid, text) in q.haystackSessions:
      let mid = service.insert(text)
      sidToMid[sid] = mid
      midToSid[mid] = sid
    let insertElapsed = cpuTime() - tInsertStart

    let tQueryStart = cpuTime()
    let res = service.search(q.question, topK)
    let queryElapsed = cpuTime() - tQueryStart

    totalSessions += q.haystackSessions.len
    totalInsertTime += insertElapsed
    totalQueryTime += queryElapsed
    queryTimes.add(queryElapsed)
    sessionCounts.add(q.haystackSessions.len)

    var foundAt = -1
    for j, mem in res:
      let sid = midToSid.getOrDefault(mem.id, "")
      if sid.len == 0: continue
      if q.answerSessionIds.contains(sid):
        foundAt = j
        break

    if foundAt == 0: r1Hits.inc
    if foundAt >= 0 and foundAt < 5: r5Hits.inc
    if foundAt >= 0 and foundAt < 10: r10Hits.inc

    let t = q.qType
    typeR1[t] = (typeR1.getOrDefault(t).hits + (if foundAt == 0: 1 else: 0), typeR1.getOrDefault(t).total + 1)
    typeR5[t] = (typeR5.getOrDefault(t).hits + (if foundAt >= 0 and foundAt < 5: 1 else: 0), typeR5.getOrDefault(t).total + 1)
    typeR10[t] = (typeR10.getOrDefault(t).hits + (if foundAt >= 0 and foundAt < 10: 1 else: 0), typeR10.getOrDefault(t).total + 1)

  sort(queryTimes)
  let avgSessions = if questions.len > 0: float(totalSessions) / float(questions.len) else: 0.0

  echo "\n=== LongMemEval-S Retrieval Results ==="
  echo "Evaluated: ", questions.len, " questions"
  echo "Total sessions indexed: ", totalSessions
  echo "Avg sessions per question: ", formatFloat(avgSessions, ffDecimal, 1)

  echo "\nIndexing (per-question haystack):"
  echo "  Total insert time: ", formatFloat(totalInsertTime, ffDecimal, 2), " s"
  echo "  Insert throughput: ", formatFloat(float(totalSessions) / totalInsertTime, ffDecimal, 1), " sessions/s"
  echo "  Avg per question:  ", formatFloat(totalInsertTime / float(questions.len) * 1000, ffDecimal, 2), " ms"

  echo "\nQuery Speed:"
  echo "  Total query time:  ", formatFloat(totalQueryTime, ffDecimal, 2), " s"
  echo "  Query throughput:  ", formatFloat(float(questions.len) / totalQueryTime, ffDecimal, 2), " q/s"
  echo "  P50 latency:       ", formatFloat(queryTimes[int(float(queryTimes.high) * 0.5)] * 1000, ffDecimal, 2), " ms"
  echo "  P95 latency:       ", formatFloat(queryTimes[int(float(queryTimes.high) * 0.95)] * 1000, ffDecimal, 2), " ms"

  echo "\nRecall Metrics:"
  echo "  R@1:  ", formatFloat(float(r1Hits) / float(questions.len) * 100, ffDecimal, 2), "%"
  echo "  R@5:  ", formatFloat(float(r5Hits) / float(questions.len) * 100, ffDecimal, 2), "%"
  echo "  R@10: ", formatFloat(float(r10Hits) / float(questions.len) * 100, ffDecimal, 2), "%"

  echo "\nPer-category breakdown:"
  var typeList = newSeq[string]()
  for t, _ in typeR5:
    typeList.add(t)
  typeList.sort()
  echo "  Category                     R@1        R@5        R@10       Count"
  for t in typeList:
    let r1v = typeR1[t]
    let r5v = typeR5[t]
    let r10v = typeR10[t]
    let r1pct = formatFloat(float(r1v.hits) / float(r1v.total) * 100, ffDecimal, 2)
    let r5pct = formatFloat(float(r5v.hits) / float(r5v.total) * 100, ffDecimal, 2)
    let r10pct = formatFloat(float(r10v.hits) / float(r10v.total) * 100, ffDecimal, 2)
    echo "  ", align(t, 28), " ", align(r1pct, 6), "%  ", align(r5pct, 6), "%  ", align(r10pct, 6), "%  ", r1v.total

proc printUsage() =
  echo """
Adaline LongMemEval Benchmark

Usage:
  longmemeval [topK]

Options:
  topK    Number of results to retrieve (default: 10)

Examples:
  longmemeval
  longmemeval 5
"""

when isMainModule:
  let args = commandLineParams()
  if args.len > 0 and args[0] in ["help", "--help", "-h"]:
    printUsage()
    quit(0)
  let k = if args.len > 0: parseInt(args[0]) else: DefaultTopK
  runBenchmark(k)

import std/[os, strutils, json, times, tables, sequtils, algorithm]
import ../domain/services/memory_service
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

  let cfg = defaultEngineConfig()

  for i, q in questions:
    if i mod 50 == 0:
      echo "Processing ", i, "/", questions.len, "..."

    let benchDir = getCurrentDir() / "benchmarks" / "data" / "longmemeval_run" / q.id
    removeDir(benchDir)
    var service = initMemoryService(benchDir, cfg)

    var sidToMid = initTable[string, uint64]()
    var midToSid = initTable[uint64, string]()
    for (sid, text) in q.haystackSessions:
      let mid = service.insert(text)
      sidToMid[sid] = mid
      midToSid[mid] = sid

    let res = service.search(q.question, topK)

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

  echo "\n=== LongMemEval-S Retrieval Results ==="
  echo "Evaluated: ", questions.len, " questions"
  echo "R@1:  ", formatFloat(float(r1Hits) / float(questions.len) * 100, ffDecimal, 2), "%"
  echo "R@5:  ", formatFloat(float(r5Hits) / float(questions.len) * 100, ffDecimal, 2), "%"
  echo "R@10: ", formatFloat(float(r10Hits) / float(questions.len) * 100, ffDecimal, 2), "%"

  echo "\nPer-category R@5:"
  var typeList = newSeq[string]()
  for t, _ in typeR5:
    typeList.add(t)
  typeList.sort()
  for t in typeList:
    let v = typeR5[t]
    let pct = formatFloat(float(v.hits) / float(v.total) * 100, ffDecimal, 2)
    echo "  ", align(t, 30), " ", align(pct, 6), "% (", v.hits, "/", v.total, ")"

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

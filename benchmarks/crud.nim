import std/[os, strutils, json, times, tables, sequtils, algorithm]
import ../domain/services/memory/types
import ../domain/services/memory/init
import ../domain/services/memory/insert
import ../domain/services/memory/delete
import ../domain/services/memory/update
import ../domain/services/memory/search
import ../domain/entities/config

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

proc percentile(sortedValues: seq[float], p: float): float =
  if sortedValues.len == 0: return 0.0
  let idx = int(float(sortedValues.high) * p)
  return sortedValues[idx]

proc runCrudBenchmark*(datasetName: string; deleteCount, updateCount: int) =
  let dataDir = "benchmarks/" & datasetName
  if not dirExists(dataDir):
    echo "Dataset not found: ", dataDir
    echo "Run ./benchmarks/beir ", datasetName, " first to download."
    quit(1)

  let corpus = loadCorpus(dataDir)
  echo "Dataset: ", datasetName
  echo "  Corpus:   ", corpus.len
  echo "  Delete:   ", deleteCount
  echo "  Update:   ", updateCount

  let cfg = defaultEngineConfig()
  let benchDir = getCurrentDir() / "benchmarks" / "data" / (datasetName & "_crud")
  removeDir(benchDir)
  var service = initMemoryService(benchDir, cfg)

  var idMap = initTable[uint64, string]()

  # --- Insert phase ---
  var insertTimes = newSeq[float](corpus.len)
  let tInsertStart = cpuTime()
  for i, doc in corpus:
    let t0 = cpuTime()
    let memId = service.insert(doc.text)
    insertTimes[i] = cpuTime() - t0
    idMap[memId] = doc.id
  let totalInsert = cpuTime() - tInsertStart

  sort(insertTimes)
  echo "\n=== Insert Phase ==="
  echo "  Total:     ", formatFloat(totalInsert, ffDecimal, 4), " s"
  echo "  Throughput:", formatFloat(float(corpus.len) / totalInsert, ffDecimal, 2), " docs/s"
  echo "  P50:       ", formatFloat(percentile(insertTimes, 0.5) * 1000, ffDecimal, 4), " ms"
  echo "  P95:       ", formatFloat(percentile(insertTimes, 0.95) * 1000, ffDecimal, 4), " ms"

  # --- Delete phase ---
  let deleteIds = toSeq(idMap.keys)[0 ..< min(deleteCount, idMap.len)]
  var deleteTimes = newSeq[float](deleteIds.len)
  let tDeleteStart = cpuTime()
  for i, memId in deleteIds:
    let t0 = cpuTime()
    service.deleteMemory(memId)
    deleteTimes[i] = cpuTime() - t0
  let totalDelete = cpuTime() - tDeleteStart

  sort(deleteTimes)
  echo "\n=== Delete Phase ==="
  echo "  Total:     ", formatFloat(totalDelete, ffDecimal, 4), " s"
  echo "  Throughput:", formatFloat(float(deleteIds.len) / totalDelete, ffDecimal, 2), " docs/s"
  echo "  P50:       ", formatFloat(percentile(deleteTimes, 0.5) * 1000, ffDecimal, 4), " ms"
  echo "  P95:       ", formatFloat(percentile(deleteTimes, 0.95) * 1000, ffDecimal, 4), " ms"

  # --- Update phase (on remaining docs) ---
  let remainingIds = toSeq(idMap.keys)[deleteCount ..< min(deleteCount + updateCount, idMap.len)]
  var updateTimes = newSeq[float](remainingIds.len)
  let tUpdateStart = cpuTime()
  for i, memId in remainingIds:
    let t0 = cpuTime()
    service.updateMemory(memId, "Updated content for benchmarking purposes. " & $i)
    updateTimes[i] = cpuTime() - t0
  let totalUpdate = cpuTime() - tUpdateStart

  sort(updateTimes)
  echo "\n=== Update Phase ==="
  echo "  Total:     ", formatFloat(totalUpdate, ffDecimal, 4), " s"
  echo "  Throughput:", formatFloat(float(remainingIds.len) / totalUpdate, ffDecimal, 2), " docs/s"
  echo "  P50:       ", formatFloat(percentile(updateTimes, 0.5) * 1000, ffDecimal, 4), " ms"
  echo "  P95:       ", formatFloat(percentile(updateTimes, 0.95) * 1000, ffDecimal, 4), " ms"

  # --- Post-CRUD query sanity check ---
  let tQueryStart = cpuTime()
  for i in 0 ..< 100:
    discard service.search("machine learning", 10)
  let totalQuery = cpuTime() - tQueryStart
  echo "\n=== Post-CRUD Query Sanity (100 queries) ==="
  echo "  Total:     ", formatFloat(totalQuery, ffDecimal, 4), " s"
  echo "  Throughput:", formatFloat(100.0 / totalQuery, ffDecimal, 2), " queries/s"
  echo "  Indexed:   ", service.textCache.len, " memories"

proc printUsage() =
  echo """
Adaline CRUD Benchmark

Usage:
  crud <dataset> [delete_count] [update_count]

Examples:
  crud scifact 1000 1000
  crud nfcorpus 500 500
"""

proc main() =
  let args = commandLineParams()
  if args.len == 0 or args[0] in ["help", "--help", "-h"]:
    printUsage()
    return

  let datasetName = args[0]
  let deleteCount = if args.len > 1: parseInt(args[1]) else: 1000
  let updateCount = if args.len > 2: parseInt(args[2]) else: 1000
  runCrudBenchmark(datasetName, deleteCount, updateCount)

when isMainModule:
  main()

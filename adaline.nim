import std/[os, strutils, tables, times]
import domain/services/memory_service
import domain/entities/config
import domain/entities/memory
import use_cases/insert_memory
import use_cases/search_memories

const
  AppName = "Adaline"
  DefaultDataDir = "./data"

proc printUsage() =
  echo """
Adaline — Sparse Distributed Representation Memory Engine

Usage:
  adaline <command> [options]

Commands:
  insert <text>           Insert a memory (text from argument)
  search <query> [k]      Search memories (default k=10)
  stats                   Show index statistics
  help                    Show this help message

Examples:
  adaline insert "The quick brown fox"
  adaline search "quick fox" 5
"""

proc getDataDir(): string =
  result = getEnv("ADALINE_DATA_DIR", DefaultDataDir)

proc formatTimestamp(ts: uint64): string =
  if ts == 0: return "unknown"
  let dt = fromUnix(int64(ts))
  return dt.format("yyyy-MM-dd HH:mm:ss")

proc cmdInsert(args: seq[string]) =
  let dataDir = getDataDir()
  let cfg = defaultEngineConfig()
  var service = initMemoryService(dataDir, cfg)

  if args.len == 0:
    stderr.writeLine("Error: insert requires text argument")
    quit(1)

  let content = args.join(" ").strip()
  if content.len == 0 or content == "-":
    stderr.writeLine("Error: no content to insert")
    quit(1)

  let output = insertMemory(service, InsertMemoryInput(content: content))
  let ts = service.timestampCache.getOrDefault(output.memoryId, 0)
  echo "Inserted: id=", output.memoryId, " at ", formatTimestamp(ts)
  echo "Total indexed: ", service.textCache.len, " memories"

proc cmdSearch(args: seq[string]) =
  let dataDir = getDataDir()
  let cfg = defaultEngineConfig()
  var service = initMemoryService(dataDir, cfg)

  if args.len == 0:
    stderr.writeLine("Error: search requires a query argument")
    quit(1)

  var k = 10
  var queryTokens: seq[string]

  for i in 0 ..< args.len:
    if i == args.len - 1 and args.len > 1:
      try:
        k = parseInt(args[i])
        if k < 1: k = 1
        if k > 100: k = 100
        continue
      except ValueError:
        discard
    queryTokens.add(args[i])

  let query = queryTokens.join(" ")
  let output = searchMemories(service, SearchMemoriesInput(query: query, topK: k))

  echo "Query: \"", query, "\""
  echo "Results: ", output.memories.len, " / indexed ", service.textCache.len
  echo ""

  if output.memories.len == 0:
    echo "No results found."
    return

  for i, mem in output.memories:
    let scoreStr = formatFloat(mem.score, ffDecimal, 4)
    let tsStr = formatTimestamp(mem.createdAt)
    echo align($(i + 1), 3), ". [", align($mem.id, 6), "]  score=", scoreStr,
             "  ", tsStr, "  ", mem.content

proc cmdStats() =
  let dataDir = getDataDir()
  let cfg = defaultEngineConfig()
  var service = initMemoryService(dataDir, cfg)

  var oldestTs = high(uint64)
  var newestTs = low(uint64)
  for mid, ts in service.timestampCache:
    if ts < oldestTs: oldestTs = ts
    if ts > newestTs: newestTs = ts

  echo "=== Adaline Index Stats ==="
  echo "Data directory: ", dataDir
  echo "Memories:       ", service.textCache.len
  echo "Corpus memories:", service.corpus.numMemories
  echo "HNSW layers:    ", service.maxHnswLayer + 1
  echo "HNSW entry:     ", service.hnswEntryPoint
  if service.timestampCache.len > 0:
    echo "Oldest memory:  ", formatTimestamp(oldestTs)
    echo "Newest memory:  ", formatTimestamp(newestTs)
  echo "Fingerprint:    ", cfg.fingerprintBits, " bits (", cfg.fingerprintBytes, " bytes)"
  echo "  Tokens:       ", cfg.tokenBits, " bits"
  echo "  Bigrams:      ", cfg.bigramBits, " bits"
  echo "  Context:      ", cfg.contextBits, " bits"
  echo "LSH bands:      ", cfg.lshBands, " x ", cfg.lshRows, " rows"
  echo "HNSW maxLayers: ", cfg.hnswMaxLayers
  echo "HNSW efSearch:  ", cfg.hnswEfSearch
  echo "RRF k:          ", cfg.rrfK
  echo "Dirichlet mu:   ", cfg.dirichletMu

proc main() =
  let args = commandLineParams()

  if args.len == 0:
    printUsage()
    quit(0)

  let cmd = args[0].toLowerAscii()
  let rest = if args.len > 1: args[1..^1] else: @[]

  case cmd
  of "insert":
    cmdInsert(rest)
  of "search":
    cmdSearch(rest)
  of "stats":
    cmdStats()
  of "help", "--help", "-h":
    printUsage()
  else:
    stderr.writeLine("Unknown command: ", cmd)
    printUsage()
    quit(1)

when isMainModule:
  main()

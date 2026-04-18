import std/[os, strutils, tables]
import domain/services/memory_service
import domain/entities/config
import domain/entities/memory
import use_cases/insert_memory
import use_cases/search_memories

const
  AppName = "Adaline"
  AppVersion = "0.1.0"
  DefaultDataDir = "./data"

proc printUsage() =
  echo """
Adaline — Sparse Distributed Representation Memory Engine

Usage:
  adaline <command> [options]

Commands:
  insert <text>           Insert a memory (text from argument)
  insert -                Insert a memory (text from stdin, one per line)
  search <query> [k]      Search memories (default k=10)
  stats                   Show index statistics
  repl                    Interactive search REPL
  help                    Show this help message
  version                 Show version

Examples:
  adaline insert "The quick brown fox"
  echo "Hello world" | adaline insert -
  adaline search "quick fox" 5
  adaline repl
"""

proc printVersion() =
  echo AppName, " v", AppVersion
  echo "Fingerprint: 10240 bits | HNSW + MinHash LSH + QLM Lexical"

proc getDataDir(): string =
  result = getEnv("ADALINE_DATA_DIR", DefaultDataDir)

proc cmdInsert(args: seq[string]) =
  let dataDir = getDataDir()
  let cfg = defaultEngineConfig()
  var service = initMemoryService(dataDir, cfg)

  var lines: seq[string]
  if args.len == 0:
    printUsage()
    quit(1)

  if args[0] == "-":
    # Read from stdin
    for line in stdin.lines:
      let trimmed = line.strip()
      if trimmed.len > 0:
        lines.add(trimmed)
  else:
    lines.add(args.join(" ").strip())

  if lines.len == 0:
    stderr.writeLine("Error: no content to insert")
    quit(1)

  for content in lines:
    let output = insertMemory(service, InsertMemoryInput(content: content))
    echo "Inserted: id=", output.memoryId, " len=", content.len

  echo "Total indexed: ", service.textCache.len, " memories"

proc cmdSearch(args: seq[string]) =
  let dataDir = getDataDir()
  let cfg = defaultEngineConfig()
  var service = initMemoryService(dataDir, cfg)

  if args.len == 0:
    stderr.writeLine("Error: search requires a query argument")
    quit(1)

  var k = 10
  var queryWords: seq[string]

  # Parse: last arg might be a number (topK)
  for i in 0 ..< args.len:
    if i == args.len - 1 and args.len > 1:
      try:
        k = parseInt(args[i])
        if k < 1: k = 1
        if k > 100: k = 100
        continue
      except ValueError:
        discard
    queryWords.add(args[i])

  let query = queryWords.join(" ")
  let output = searchMemories(service, SearchMemoriesInput(query: query, topK: k))

  echo "Query: \"", query, "\""
  echo "Results: ", output.memories.len, " / indexed ", service.textCache.len
  echo ""

  if output.memories.len == 0:
    echo "No results found."
    return

  for i, mem in output.memories:
    let scoreStr = formatFloat(mem.score, ffDecimal, 4)
    echo align($(i + 1), 3), ". [", align($mem.id, 6), "]  score=", scoreStr, "  ", mem.content

proc cmdStats() =
  let dataDir = getDataDir()
  let cfg = defaultEngineConfig()
  var service = initMemoryService(dataDir, cfg)

  echo "=== Adaline Index Stats ==="
  echo "Data directory: ", dataDir
  echo "Memories:       ", service.textCache.len
  echo "Corpus docs:    ", service.corpus.numDocs
  echo "HNSW layers:    ", service.maxHnswLayer + 1
  echo "HNSW entry:     ", service.hnswEntryPoint
  echo "Fingerprint:    ", cfg.fingerprintBits, " bits (", cfg.fingerprintBytes, " bytes)"
  echo "  Tokens:       ", cfg.tokenBits, " bits"
  echo "  Bigrams:      ", cfg.bigramBits, " bits"
  echo "  Context:      ", cfg.contextBits, " bits"
  echo "LSH bands:      ", cfg.lshBands, " x ", cfg.lshRows, " rows"
  echo "HNSW maxLayers: ", cfg.hnswMaxLayers
  echo "HNSW efSearch:  ", cfg.hnswEfSearch
  echo "RRF k:          ", cfg.rrfK
  echo "Dirichlet mu:   ", cfg.dirichletMu

proc cmdRepl() =
  let dataDir = getDataDir()
  let cfg = defaultEngineConfig()
  var service = initMemoryService(dataDir, cfg)

  echo "=== Adaline Interactive REPL ==="
  echo "Type a query to search. Empty line quits."
  echo "Memories indexed: ", service.textCache.len
  echo ""

  while true:
    stdout.write("adaline> ")
    stdout.flushFile()
    var line: string
    if not stdin.readLine(line):
      break
    let query = line.strip()
    if query.len == 0:
      break

    let output = searchMemories(service, SearchMemoriesInput(query: query, topK: 10))
    echo "Results: ", output.memories.len
    for i, mem in output.memories:
      let scoreStr = formatFloat(mem.score, ffDecimal, 4)
      echo "  ", align($(i + 1), 2), ". [", $mem.id, "] ", scoreStr, "  ", mem.content
    echo ""

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
  of "repl":
    cmdRepl()
  of "help", "--help", "-h":
    printUsage()
  of "version", "--version", "-v":
    printVersion()
  else:
    stderr.writeLine("Unknown command: ", cmd)
    printUsage()
    quit(1)

when isMainModule:
  main()

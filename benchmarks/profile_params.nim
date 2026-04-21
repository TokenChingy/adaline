import std/[os, times, tables, strformat]
import ../domain/services/memory_service
import ../domain/entities/config
import ../domain/algorithms/sdr_encoder
import ../domain/algorithms/fingerprint_lsh
import ../domain/algorithms/hnsw_graph
import ../domain/algorithms/lexical_index
import ../domain/algorithms/chunker
import ../domain/algorithms/corpus_index
import ../infrastructure/mmapped_storage

proc loadCorpus(dataDir: string): seq[string] =
  let f = open(dataDir / "corpus.jsonl")
  defer: f.close()
  for line in f.lines:
    result.add(line)
    if result.len >= 1000: break

proc runWithParams(label: string; maxNeighbors, efConstruction: int) =
  let dataDir = "benchmarks/scifact"
  let corpus = loadCorpus(dataDir)

  var cfg = defaultEngineConfig()
  cfg.hnswEnabled = true
  cfg.hnswMaxNeighbors = maxNeighbors
  cfg.hnswEfConstruction = efConstruction

  let benchDir = getCurrentDir() / "benchmarks" / "data" / ("params_" & label)
  removeDir(benchDir)
  var svc = initMemoryService(benchDir, cfg)

  let tInsertStart = cpuTime()
  for text in corpus:
    discard svc.insert(text)
  let tInsert = cpuTime() - tInsertStart

  echo &"\n=== {label} (M={maxNeighbors}, efC={efConstruction}) ==="
  echo &"Insert: {tInsert:.3f}s ({corpus.len.float / tInsert:.1f} docs/s)"

  let tSearchStart = cpuTime()
  for i in 0 ..< 100:
    discard svc.search("machine learning artificial intelligence", 10)
  let tSearch = cpuTime() - tSearchStart
  echo &"Search: {tSearch:.3f}s ({100.0 / tSearch:.1f} q/s)"

proc main() =
  runWithParams("M32_ef200", 32, 200)
  runWithParams("M16_ef200", 16, 200)
  runWithParams("M32_ef100", 32, 100)
  runWithParams("M16_ef100", 16, 100)
  runWithParams("M16_ef50", 16, 50)
  runWithParams("M8_ef100", 8, 100)

when isMainModule:
  main()

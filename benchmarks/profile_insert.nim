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

proc main() =
  let dataDir = "benchmarks/scifact"
  let corpus = loadCorpus(dataDir)
  echo "Profiling insertion of ", corpus.len, " documents"

  var cfg = defaultEngineConfig()
  cfg.hnswEnabled = true
  cfg.hnswEfConstruction = 200
  cfg.hnswMaxNeighbors = 32

  let benchDir = getCurrentDir() / "benchmarks" / "data" / "profile_insert"
  removeDir(benchDir)
  var svc = initMemoryService(benchDir, cfg)

  var tEncode = 0.0
  var tLsh = 0.0
  var tLexical = 0.0
  var tHnsw = 0.0
  var tWal = 0.0
  var tChunk = 0.0
  var tTotal = 0.0

  let tStart = cpuTime()
  for i, text in corpus:
    let t0 = cpuTime()
    let parentId = svc.storage.allocId()
    let timestamp = uint64(0)
    discard svc.storage.appendWal(parentId, timestamp, text)
    tWal += cpuTime() - t0

    let t1 = cpuTime()
    addMemory(svc.corpus, text)
    let chunks = splitIntoChunks(text, svc.cfg)
    tChunk += cpuTime() - t1

    let chunkText = if chunks.len == 1: text else: chunks[0]
    let chunkId = parentId

    let t2 = cpuTime()
    var fp = encodeSdr(chunkText, svc.cfg, svc.corpus)
    tEncode += cpuTime() - t2

    svc.storage.writeFingerprintUnsafe(chunkId, fp)

    let t3 = cpuTime()
    insertLsh(svc.lsh, addr fp, chunkId)
    tLsh += cpuTime() - t3

    let t4 = cpuTime()
    addMemory(svc.lexical, chunkId, chunkText)
    tLexical += cpuTime() - t4

    let t5 = cpuTime()
    insertHnsw(svc.storage.graphMem, svc.storage.fpMem, chunkId, addr fp,
               svc.cfg, svc.maxHnswLayer, svc.hnswEntryPoint,
               svc.hnswReverseIndex)
    tHnsw += cpuTime() - t5

  tTotal = cpuTime() - tStart

  echo &"\n=== Insertion Profile ({corpus.len} docs) ==="
  echo &"Total time:     {tTotal:.4f}s ({corpus.len.float / tTotal:.1f} docs/s)"
  echo &"WAL:            {tWal:.4f}s  ({100.0 * tWal / tTotal:.1f}%)"
  echo &"Chunk+Corpus:   {tChunk:.4f}s  ({100.0 * tChunk / tTotal:.1f}%)"
  echo &"Encode SDR:     {tEncode:.4f}s  ({100.0 * tEncode / tTotal:.1f}%)"
  echo &"LSH insert:     {tLsh:.4f}s  ({100.0 * tLsh / tTotal:.1f}%)"
  echo &"Lexical add:    {tLexical:.4f}s  ({100.0 * tLexical / tTotal:.1f}%)"
  echo &"HNSW insert:    {tHnsw:.4f}s  ({100.0 * tHnsw / tTotal:.1f}%)"

when isMainModule:
  main()

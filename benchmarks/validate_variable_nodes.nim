## Validation script for variable-size HNSW graph nodes.
## Creates a small index, verifies the graph file size matches the
## runtime-configurable record size, and runs basic search queries
## to confirm no functional regression.

import std/[os, strformat]
import ../domain/services/memory/init
import ../domain/services/memory/insert
import ../domain/services/memory/search
import ../domain/services/memory/delete

proc main() =
  let dataDir = "benchmarks/validate_variable_nodes_data"
  removeDir(dataDir)
  createDir(dataDir)

  var cfg = defaultEngineConfig()
  var service = initMemoryService(dataDir, cfg)

  let texts = @[
    "The quick brown fox jumps over the lazy dog",
    "A fast brown fox leaps over a sleepy dog",
    "The lazy dog sleeps in the sun",
    "Machine learning models require large datasets",
    "Deep neural networks learn hierarchical representations",
    "Natural language processing enables machine translation",
    "The fox is quick and the dog is lazy",
    "Datasets are essential for training machine learning models",
    "Hierarchical representations emerge in deep networks",
    "Machine translation relies on natural language processing",
  ]

  var ids: seq[uint64]
  for t in texts:
    ids.add(insert(service, t))

  let graphPath = dataDir / "graph.bin"
  let graphSize = getFileSize(graphPath)
  let expectedRecordSize = uint64(8 + cfg.hnswMaxLayers * cfg.hnswMaxNeighbors * 4)
  let expectedSize = int(256'u64 + service.storage.graphCapacity * expectedRecordSize)

  echo &"Config: hnswMaxLayers={cfg.hnswMaxLayers}, hnswMaxNeighbors={cfg.hnswMaxNeighbors}"
  echo &"Expected graph record size: {expectedRecordSize} bytes"
  echo &"Old hardcoded record size: 1032 bytes"
  echo &"Savings: {float64(1032 - expectedRecordSize) / 1032.0 * 100.0:.1f}%"
  echo &"Graph file size: {graphSize} bytes"
  echo &"Graph capacity: {service.storage.graphCapacity} slots"
  echo &"Expected file size (header + capacity * record): {expectedSize} bytes"

  doAssert graphSize == expectedSize,
    &"Graph size mismatch: got {graphSize}, expected {expectedSize}"

  # Verify search works
  let results = search(service, "quick fox", 3)
  echo &"Search 'quick fox' returned {results.len} results"
  for r in results:
    echo &"  id={r.id} score={r.score:.4f} text='{r.content}'"
  doAssert results.len > 0, "Search should return results"

  # Verify delete works
  deleteMemory(service, ids[0])
  let afterDelete = search(service, "quick fox", 3)
  echo &"After delete, search returned {afterDelete.len} results"
  doAssert afterDelete.len > 0, "Search should still return results after delete"

  echo "\nValidation PASSED"

when isMainModule:
  main()

## Validation script for fingerprint compression.
## Creates a small index, measures the compressed fingerprint data size,
## and verifies search/delete still work correctly.

import std/[os, strformat]
import ../domain/services/memory/init
import ../domain/services/memory/insert
import ../domain/services/memory/search
import ../domain/services/memory/delete
import ../domain/entities/fingerprint
import ../infrastructure/mmapped_storage

proc main() =
  let dataDir = "benchmarks/validate_fp_compression_data"
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

  let idxPath = dataDir / "fingerprints.idx"
  let dataPath = dataDir / "fingerprints.bin"
  let idxFileSize = getFileSize(idxPath)
  let dataFileSize = getFileSize(dataPath)
  let actualDataSize = int(service.storage.fpDataSize)

  var totalSegments = 0
  for id in 0'u64 ..< service.storage.recordCount:
    var fp: Fingerprint
    service.storage.readFingerprint(id, addr fp)
    totalSegments += fingerprintActiveSegments(fp)

  let avgSegments = totalSegments.float / service.storage.recordCount.float
  let avgCompressedSize = actualDataSize.float / service.storage.recordCount.float

  echo &"Inserted {texts.len} memories ({service.storage.recordCount} total slots incl. chunks)"
  echo &"fingerprints.idx file size: {idxFileSize} bytes (preallocated)"
  echo &"fingerprints.bin file size: {dataFileSize} bytes (preallocated)"
  echo &"Actual compressed data bytes: {actualDataSize} bytes"
  echo &"Avg active segments per fingerprint: {avgSegments:.1f} / 160"
  echo &"Avg compressed size per fingerprint: {avgCompressedSize:.1f} bytes"
  echo &"Old fixed size per fingerprint: 1280 bytes"
  echo &"Compression ratio: {1280.0 / avgCompressedSize:.2f}x"
  echo &"Space savings: {(1280.0 - avgCompressedSize) / 1280.0 * 100.0:.1f}%"

  # Verify search works
  let results = search(service, "quick fox", 3)
  echo &"\nSearch 'quick fox' returned {results.len} results"
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

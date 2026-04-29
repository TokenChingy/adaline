## Unit tests for Checkpoint service.


import unittest
import ../../domain/services/memory/init
import ../../domain/services/memory/insert
import ../../domain/services/memory/checkpoint
import ../../domain/entities/config
import std/os

suite "Checkpoint":
  let cfg = defaultEngineConfig()
  let testDir = getCurrentDir() / "tests" / "tmp_checkpoint"

  setup:
    removeDir(testDir)

  teardown:
    removeDir(testDir)

  test "checkpoint creates persisted index files":
    var svc = initMemoryService(testDir, cfg)
    discard svc.insert("checkpoint test content")

    checkpoint(svc)

    check fileExists(testDir / "lsh.bin")
    check fileExists(testDir / "lexical.bin")
    check fileExists(testDir / "corpus.bin")

  test "checkpoint files are non-empty after indexing":
    var svc = initMemoryService(testDir, cfg)
    discard svc.insert("another checkpoint test")

    checkpoint(svc)

    let lshSize = getFileSize(testDir / "lsh.bin")
    let lexicalSize = getFileSize(testDir / "lexical.bin")
    let corpusSize = getFileSize(testDir / "corpus.bin")
    check lshSize > 0
    check lexicalSize > 0
    check corpusSize > 0

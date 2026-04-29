## Unit tests for Init memory service.


import unittest
import ../../domain/services/memory/init
import ../../domain/services/memory/insert
import ../../domain/services/memory/search
import ../../domain/entities/config
import std/[os, tables]

suite "Init memory service":
  let cfg = defaultEngineConfig()
  let testDir = getCurrentDir() / "tests" / "tmp_init_svc"

  setup:
    removeDir(testDir)

  teardown:
    removeDir(testDir)

  test "initMemoryService creates storage directory":
    discard initMemoryService(testDir, cfg)
    check dirExists(testDir)

  test "initMemoryService creates required files":
    discard initMemoryService(testDir, cfg)
    check fileExists(testDir / "wal.bin")
    check fileExists(testDir / "fingerprints.bin")
    check fileExists(testDir / "fingerprints.idx")
    check fileExists(testDir / "chunks.bin")

  test "initMemoryService replays WAL on restart":
    block:
      var svc = initMemoryService(testDir, cfg)
      discard svc.insert("wal replay test")

    block:
      var svc2 = initMemoryService(testDir, cfg)
      check svc2.textCache.len == 1
      let results = svc2.search("wal", 5)
      check results.len == 1
      check results[0].content == "wal replay test"

  test "initMemoryService replays chunk mappings on restart":
    block:
      var svc = initMemoryService(testDir, cfg)
      let id = svc.insert("chunk mapping test")
      check svc.parentToChunks.hasKey(id)

    block:
      var svc2 = initMemoryService(testDir, cfg)
      check svc2.parentToChunks.len == 1

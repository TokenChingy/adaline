import unittest
import ../../domain/services/memory/init
import ../../domain/services/memory/search
import ../../domain/entities/config
import ../../use_cases/insert_memory
import std/os

suite "Insert memory use case":
  let cfg = defaultEngineConfig()
  let testDir = getCurrentDir() / "tests" / "tmp_data_insert"

  setup:
    removeDir(testDir)

  teardown:
    removeDir(testDir)

  test "inserts a memory through the use case":
    var svc = initMemoryService(testDir, cfg)
    let output = insertMemory(svc, InsertMemoryInput(content: "integration test"))
    check output.memoryId == 0'u64
    let results = svc.search("integration", 5)
    check results.len == 1
    check results[0].content == "integration test"

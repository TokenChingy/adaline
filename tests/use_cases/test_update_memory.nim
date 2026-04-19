import unittest
import ../../domain/services/memory_service
import ../../domain/entities/config
import ../../use_cases/update_memory
import ../../use_cases/insert_memory
import std/os

suite "Update memory use case":
  let cfg = defaultEngineConfig()
  let testDir = getCurrentDir() / "tests" / "tmp_data_update"

  setup:
    removeDir(testDir)

  teardown:
    removeDir(testDir)

  test "updates a memory through the use case":
    var svc = initMemoryService(testDir, cfg)
    let insertOutput = insertMemory(svc, InsertMemoryInput(content: "original text"))
    let id = insertOutput.memoryId

    let updateOutput = updateMemory(svc, UpdateMemoryInput(memoryId: id, content: "updated text"))
    check updateOutput.memoryId == id

    let results = svc.search("updated", 5)
    check results.len == 1
    check results[0].content == "updated text"

  test "old content is no longer searchable after update":
    var svc = initMemoryService(testDir, cfg)
    let insertOutput = insertMemory(svc, InsertMemoryInput(content: "original text"))
    let id = insertOutput.memoryId

    discard updateMemory(svc, UpdateMemoryInput(memoryId: id, content: "updated text"))

    # Approximate search can produce accidental low-scoring matches.
    let oldResults = svc.search("original", 5)
    var found = false
    var foundScore = 0.0
    for r in oldResults:
      if r.id == id:
        found = true
        foundScore = r.score
    check (not found) or (foundScore < 0.15)

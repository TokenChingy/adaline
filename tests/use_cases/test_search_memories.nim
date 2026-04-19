import unittest
import std/[strutils, os]
import ../../domain/services/memory/init
import ../../domain/entities/config
import ../../use_cases/insert_memory
import ../../use_cases/search_memories

suite "Search memories use case":
  let cfg = defaultEngineConfig()
  let testDir = getCurrentDir() / "tests" / "tmp_data_search"

  setup:
    removeDir(testDir)

  teardown:
    removeDir(testDir)

  test "returns memories matching query":
    var svc = initMemoryService(testDir, cfg)
    discard insertMemory(svc, InsertMemoryInput(content: "the quick brown fox"))
    discard insertMemory(svc, InsertMemoryInput(content: "lazy dog sleeping"))
    discard insertMemory(svc, InsertMemoryInput(content: "foxes are quick"))
    let output = searchMemories(svc, SearchMemoriesInput(query: "quick fox", topK: 2))
    check output.memories.len == 2
    var foundFox = false
    for mem in output.memories:
      if mem.content.contains("fox"):
        foundFox = true
    check foundFox == true

  test "returns empty when no memories indexed":
    var svc = initMemoryService(testDir, cfg)
    let output = searchMemories(svc, SearchMemoriesInput(query: "anything", topK: 5))
    check output.memories.len == 0

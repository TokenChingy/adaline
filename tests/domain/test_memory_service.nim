import unittest
import ../../domain/services/memory_service
import ../../domain/entities/config
import std/[os, tables]

suite "Memory service":
  let cfg = defaultEngineConfig()
  let testDir = getCurrentDir() / "tests" / "tmp_data"

  setup:
    removeDir(testDir)

  teardown:
    removeDir(testDir)

  test "insert returns a uint64 id":
    var svc = initMemoryService(testDir, cfg)
    let id = svc.insert("hello world")
    check id == 0'u64

  test "inserted memory is searchable":
    var svc = initMemoryService(testDir, cfg)
    let id = svc.insert("hello world")
    let results = svc.search("hello", 5)
    check results.len > 0
    check results[0].id == id
    check results[0].content == "hello world"

  test "search returns relevant memory":
    var svc = initMemoryService(testDir, cfg)
    discard svc.insert("the quick brown fox")
    discard svc.insert("lazy dog sleeping")
    discard svc.insert("nim programming language")
    let results = svc.search("quick fox", 2)
    check results.len > 0
    check results[0].content == "the quick brown fox"

  test "short memory is not chunked":
    var svc = initMemoryService(testDir, cfg)
    let id = svc.insert("hello world")
    check svc.chunkToParent.hasKey(id)
    check svc.chunkToParent[id] == id

  test "long memory is chunked into multiple chunks":
    var svc = initMemoryService(testDir, cfg)
    var longText = ""
    for i in 0 ..< 100:
      longText.add("Machine learning is a subset of artificial intelligence. ")
    let parentId = svc.insert(longText)
    var chunkCount = 0
    for chunkId, pid in svc.chunkToParent:
      if pid == parentId:
        chunkCount.inc
    check chunkCount > 1

  test "search deduplicates chunks from same parent":
    var svc = initMemoryService(testDir, cfg)
    var longText = ""
    for i in 0 ..< 100:
      longText.add("Machine learning is a subset of artificial intelligence. ")
    let parentId = svc.insert(longText)
    discard svc.insert("completely unrelated text about cats and dogs")
    let results = svc.search("machine learning", 10)
    var parentCount = 0
    for r in results:
      if r.id == parentId:
        parentCount.inc
    check parentCount == 1

  test "search returns parent content not chunk content":
    var svc = initMemoryService(testDir, cfg)
    var longText = ""
    for i in 0 ..< 100:
      longText.add("Machine learning is a subset of artificial intelligence. ")
    let parentId = svc.insert(longText)
    let results = svc.search("machine learning", 5)
    check results.len > 0
    check results[0].id == parentId
    check results[0].content == longText

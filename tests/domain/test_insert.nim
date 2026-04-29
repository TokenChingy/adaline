## Unit tests for Insert memory service.


import unittest
import ../../domain/services/memory/init
import ../../domain/services/memory/insert
import ../../domain/services/memory/search
import ../../domain/entities/config
import std/[os, tables, sets]

suite "Insert memory service":
  let cfg = defaultEngineConfig()
  let testDir = getCurrentDir() / "tests" / "tmp_insert_svc"

  setup:
    removeDir(testDir)

  teardown:
    removeDir(testDir)

  test "insert returns a valid memory ID":
    var svc = initMemoryService(testDir, cfg)
    let id = svc.insert("hello world")
    check id >= 0'u64

  test "insert populates textCache":
    var svc = initMemoryService(testDir, cfg)
    let id = svc.insert("populate cache")
    check svc.textCache[id] == "populate cache"
    check svc.lowerTextCache[id] == "populate cache"

  test "insert populates tokenCache":
    var svc = initMemoryService(testDir, cfg)
    let id = svc.insert("token cache test")
    check svc.tokenCache.hasKey(id)
    check svc.tokenCache[id].contains("token")
    check svc.tokenCache[id].contains("cache")
    check svc.tokenCache[id].contains("test")

  test "insert populates timestampCache":
    var svc = initMemoryService(testDir, cfg)
    let id = svc.insert("timestamp test")
    check svc.timestampCache.hasKey(id)
    check svc.timestampCache[id] > 0'u64

  test "insert makes content searchable":
    var svc = initMemoryService(testDir, cfg)
    discard svc.insert("searchable content")
    let results = svc.search("searchable", 5)
    check results.len == 1
    check results[0].content == "searchable content"

  test "insert creates single chunk for short text":
    var svc = initMemoryService(testDir, cfg)
    let id = svc.insert("short text")
    check svc.parentToChunks[id].len == 1

  test "insert creates multiple chunks for long text":
    var svc = initMemoryService(testDir, cfg)
    var longText = ""
    for i in 0 ..< 100:
      longText.add("Machine learning is a subset of artificial intelligence. ")
    let id = svc.insert(longText)
    check svc.parentToChunks[id].len > 1

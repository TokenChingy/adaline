import unittest
import ../../domain/services/memory_service
import ../../domain/entities/config
import std/os

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

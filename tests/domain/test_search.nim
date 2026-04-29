## Unit tests for Search memory service.


import unittest
import ../../domain/services/memory/init
import ../../domain/services/memory/insert
import ../../domain/services/memory/search
import ../../domain/entities/config
import std/os

suite "Search memory service":
  let cfg = defaultEngineConfig()
  let testDir = getCurrentDir() / "tests" / "tmp_search_svc"

  setup:
    removeDir(testDir)

  teardown:
    removeDir(testDir)

  test "search finds inserted content":
    var svc = initMemoryService(testDir, cfg)
    discard svc.insert("find this content")
    let results = svc.search("find", 5)
    check results.len == 1
    check results[0].content == "find this content"

  test "search respects top-k limit":
    var svc = initMemoryService(testDir, cfg)
    for i in 0 ..< 5:
      discard svc.insert("document number " & $i)
    let results = svc.search("document", 3)
    check results.len == 3

  test "search returns low score for unrelated query":
    var svc = initMemoryService(testDir, cfg)
    discard svc.insert("some content")
    let results = svc.search("xyzunknown", 5)
    if results.len > 0:
      check results[0].score < 0.3

  test "search with semantic disabled returns lexical results":
    var semanticOnly = cfg
    semanticOnly.semanticSearchEnabled = false
    semanticOnly.lexicalSearchEnabled = true
    var svc = initMemoryService(testDir, semanticOnly)
    discard svc.insert("semantic off test")
    let results = svc.search("semantic", 5)
    check results.len == 1

  test "search with lexical disabled returns semantic results":
    var lexicalOnly = cfg
    lexicalOnly.semanticSearchEnabled = true
    lexicalOnly.lexicalSearchEnabled = false
    var svc = initMemoryService(testDir, lexicalOnly)
    discard svc.insert("lexical off test")
    let results = svc.search("lexical", 5)
    check results.len == 1

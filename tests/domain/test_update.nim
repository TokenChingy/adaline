## Unit tests for Update memory service.


import unittest
import ../../domain/services/memory/init
import ../../domain/services/memory/insert
import ../../domain/services/memory/update
import ../../domain/services/memory/search
import ../../domain/entities/config
import std/[os, tables]

suite "Update memory service":
  let cfg = defaultEngineConfig()
  let testDir = getCurrentDir() / "tests" / "tmp_update_svc"

  setup:
    removeDir(testDir)

  teardown:
    removeDir(testDir)

  test "update changes memory content":
    var svc = initMemoryService(testDir, cfg)
    let id = svc.insert("old content")
    updateMemory(svc, id, "new content")
    check svc.textCache[id] == "new content"
    check svc.lowerTextCache[id] == "new content"

  test "update makes new content searchable":
    var svc = initMemoryService(testDir, cfg)
    let id = svc.insert("old content")
    updateMemory(svc, id, "new searchable content")
    let results = svc.search("new", 5)
    check results.len == 1
    check results[0].content == "new searchable content"

  test "update makes old content unsearchable":
    var svc = initMemoryService(testDir, cfg)
    let id = svc.insert("old content")
    updateMemory(svc, id, "new content")
    let results = svc.search("old", 5)
    var found = false
    var foundScore = 0.0
    for r in results:
      if r.id == id:
        found = true
        foundScore = r.score
    check (not found) or (foundScore < 0.25)

  test "update preserves parent ID":
    var svc = initMemoryService(testDir, cfg)
    let id = svc.insert("original")
    updateMemory(svc, id, "updated")
    check svc.textCache.hasKey(id)

  test "update is safe for unknown ID":
    var svc = initMemoryService(testDir, cfg)
    updateMemory(svc, 9999'u64, "ghost content")
    check not svc.textCache.hasKey(9999'u64)

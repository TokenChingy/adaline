## Unit tests for Delete memory service.


import unittest
import ../../domain/services/memory/init
import ../../domain/services/memory/insert
import ../../domain/services/memory/delete
import ../../domain/services/memory/search
import ../../domain/entities/config
import std/[os, tables]

suite "Delete memory service":
  let cfg = defaultEngineConfig()
  let testDir = getCurrentDir() / "tests" / "tmp_delete_svc"

  setup:
    removeDir(testDir)

  teardown:
    removeDir(testDir)

  test "delete removes memory from caches":
    var svc = initMemoryService(testDir, cfg)
    let id = svc.insert("delete me")
    check svc.textCache.hasKey(id)

    deleteMemory(svc, id)
    check not svc.textCache.hasKey(id)
    check not svc.lowerTextCache.hasKey(id)
    check not svc.tokenCache.hasKey(id)
    check not svc.timestampCache.hasKey(id)

  test "delete makes memory unsearchable":
    var svc = initMemoryService(testDir, cfg)
    let id = svc.insert("find and delete")
    let before = svc.search("find", 5)
    check before.len == 1

    deleteMemory(svc, id)
    let after = svc.search("find", 5)
    check after.len == 0

  test "delete is idempotent":
    var svc = initMemoryService(testDir, cfg)
    let id = svc.insert("delete once")
    deleteMemory(svc, id)
    deleteMemory(svc, id)
    check not svc.textCache.hasKey(id)

  test "delete removes chunk mappings":
    var svc = initMemoryService(testDir, cfg)
    let id = svc.insert("delete me")
    check svc.parentToChunks.hasKey(id)

    deleteMemory(svc, id)
    check not svc.parentToChunks.hasKey(id)

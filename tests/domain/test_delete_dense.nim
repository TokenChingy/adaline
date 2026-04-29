## Unit tests for Delete Dense memory service.


import unittest
import ../../domain/services/memory/init
import ../../domain/services/memory/insert
import ../../domain/services/memory/insert_dense
import ../../domain/services/memory/delete_dense
import ../../domain/services/memory/search_dense
import ../../domain/entities/config
import std/[os, tables]

suite "Delete dense service":
  let cfg = defaultEngineConfig()
  let testDir = getCurrentDir() / "tests" / "tmp_delete_dense_svc"

  setup:
    removeDir(testDir)

  teardown:
    removeDir(testDir)

  test "deleteDense removes vector from search":
    var svc = initMemoryService(testDir, cfg)
    let vec = @[1.0'f32, 0.0'f32, 0.0'f32, 0.0'f32]
    let id = svc.insertDense(vec)

    let before = svc.searchDense(vec, 5)
    check before.len > 0

    deleteDense(svc, id)
    let after = svc.searchDense(vec, 5)
    check after.len == 0

  test "deleteDense guards against deleting text memories":
    var svc = initMemoryService(testDir, cfg)
    let textId = svc.insert("text memory")

    deleteDense(svc, textId)
    check svc.textCache.hasKey(textId)

  test "deleteDense is safe on unknown IDs":
    var svc = initMemoryService(testDir, cfg)
    deleteDense(svc, 9999'u64)
    check true

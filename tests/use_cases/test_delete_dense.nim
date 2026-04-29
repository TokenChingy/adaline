## Unit tests for Delete Dense use case.


import unittest
import ../../domain/services/memory/init
import ../../domain/services/memory/insert_dense
import ../../domain/services/memory/search_dense
import ../../domain/entities/config
import ../../use_cases/delete_dense
import std/os

suite "Delete dense use case":
  let cfg = defaultEngineConfig()
  let testDir = getCurrentDir() / "tests" / "tmp_uc_delete_dense"

  setup:
    removeDir(testDir)

  teardown:
    removeDir(testDir)

  test "deleteDense use case removes vector":
    var svc = initMemoryService(testDir, cfg)
    let vec = @[1.0'f32, 0.0'f32, 0.0'f32, 0.0'f32]
    let id = svc.insertDense(vec)

    deleteDense(svc, DeleteDenseInput(memoryId: id))

    let results = svc.searchDense(vec, 5)
    check results.len == 0

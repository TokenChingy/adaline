## Unit tests for Insert Dense memory service.


import unittest
import ../../domain/services/memory/init
import ../../domain/services/memory/insert_dense
import ../../domain/services/memory/search_dense
import ../../domain/entities/config
import std/[os, tables]

suite "Insert dense service":
  let cfg = defaultEngineConfig()
  let testDir = getCurrentDir() / "tests" / "tmp_insert_dense_svc"

  setup:
    removeDir(testDir)

  teardown:
    removeDir(testDir)

  test "insertDense returns a valid ID":
    var svc = initMemoryService(testDir, cfg)
    let vec = @[1.0'f32, 0.0'f32, 0.0'f32, 0.0'f32]
    let id = svc.insertDense(vec)
    check id >= 0'u64

  test "insertDense makes vector searchable":
    var svc = initMemoryService(testDir, cfg)
    let vecA = @[1.0'f32, 0.0'f32, 0.0'f32, 0.0'f32]
    let vecB = @[0.0'f32, 1.0'f32, 0.0'f32, 0.0'f32]
    let idA = svc.insertDense(vecA)
    discard svc.insertDense(vecB)

    let results = svc.searchDense(vecA, 5)
    check results.len > 0
    check results[0].memoryId == idA

  test "insertDense does not populate textCache":
    var svc = initMemoryService(testDir, cfg)
    let vec = @[1.0'f32, 0.0'f32, 0.0'f32, 0.0'f32]
    let id = svc.insertDense(vec)
    check not svc.textCache.hasKey(id)

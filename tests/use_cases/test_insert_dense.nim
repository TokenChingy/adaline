## Unit tests for Insert Dense use case.


import unittest
import ../../domain/services/memory/init
import ../../domain/services/memory/search_dense
import ../../domain/entities/config
import ../../use_cases/insert_dense
import std/os

suite "Insert dense use case":
  let cfg = defaultEngineConfig()
  let testDir = getCurrentDir() / "tests" / "tmp_uc_insert_dense"

  setup:
    removeDir(testDir)

  teardown:
    removeDir(testDir)

  test "insertDense use case returns memoryId":
    var svc = initMemoryService(testDir, cfg)
    let vec = @[1.0'f32, 0.0'f32, 0.0'f32, 0.0'f32]
    let output = insertDense(svc, InsertDenseInput(vec: vec))
    check output.memoryId >= 0'u64

  test "insertDense use case makes vector searchable":
    var svc = initMemoryService(testDir, cfg)
    let vec = @[1.0'f32, 0.0'f32, 0.0'f32, 0.0'f32]
    let output = insertDense(svc, InsertDenseInput(vec: vec))
    let results = svc.searchDense(vec, 5)
    check results.len > 0
    check results[0].memoryId == output.memoryId

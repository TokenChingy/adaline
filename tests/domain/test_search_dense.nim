## Unit tests for Search Dense memory service.


import unittest
import ../../domain/services/memory/init
import ../../domain/services/memory/insert_dense
import ../../domain/services/memory/search_dense
import ../../domain/entities/config
import std/os

suite "Search dense service":
  let cfg = defaultEngineConfig()
  let testDir = getCurrentDir() / "tests" / "tmp_search_dense_svc"

  setup:
    removeDir(testDir)

  teardown:
    removeDir(testDir)

  test "searchDense finds similar vectors":
    var svc = initMemoryService(testDir, cfg)
    let vecA = @[1.0'f32, 0.0'f32, 0.0'f32, 0.0'f32]
    let vecB = @[0.0'f32, 1.0'f32, 0.0'f32, 0.0'f32]
    let idA = svc.insertDense(vecA)
    discard svc.insertDense(vecB)

    let results = svc.searchDense(vecA, 5)
    check results.len > 0
    check results[0].memoryId == idA
    check results[0].score > 0.0

  test "searchDense respects topK limit":
    var svc = initMemoryService(testDir, cfg)
    for i in 0 ..< 5:
      var vec = @[0.0'f32, 0.0'f32, 0.0'f32, 0.0'f32]
      vec[i mod 4] = 1.0'f32
      discard svc.insertDense(vec)

    let query = @[1.0'f32, 0.0'f32, 0.0'f32, 0.0'f32]
    let results = svc.searchDense(query, 3)
    check results.len == 3

  test "searchDense returns empty when nothing indexed":
    var svc = initMemoryService(testDir, cfg)
    let query = @[1.0'f32, 0.0'f32, 0.0'f32, 0.0'f32]
    let results = svc.searchDense(query, 5)
    check results.len == 0

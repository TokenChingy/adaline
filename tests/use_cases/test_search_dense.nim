## Unit tests for Search Dense use case.


import unittest
import ../../domain/services/memory/init
import ../../domain/services/memory/insert_dense
import ../../domain/entities/config
import ../../use_cases/search_dense
import std/os

suite "Search dense use case":
  let cfg = defaultEngineConfig()
  let testDir = getCurrentDir() / "tests" / "tmp_uc_search_dense"

  setup:
    removeDir(testDir)

  teardown:
    removeDir(testDir)

  test "searchDense use case returns results":
    var svc = initMemoryService(testDir, cfg)
    let vec = @[1.0'f32, 0.0'f32, 0.0'f32, 0.0'f32]
    discard svc.insertDense(vec)

    let output = searchDense(svc, SearchDenseInput(vec: vec, topK: 5))
    check output.results.len > 0
    check output.results[0].score > 0.0

  test "searchDense use case respects topK":
    var svc = initMemoryService(testDir, cfg)
    for i in 0 ..< 5:
      var v = @[0.0'f32, 0.0'f32, 0.0'f32, 0.0'f32]
      v[i mod 4] = 1.0'f32
      discard svc.insertDense(v)

    let query = @[1.0'f32, 0.0'f32, 0.0'f32, 0.0'f32]
    let output = searchDense(svc, SearchDenseInput(vec: query, topK: 3))
    check output.results.len == 3

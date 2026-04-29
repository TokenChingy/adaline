## Unit tests for MemoryService types.


import unittest
import ../../domain/services/memory/types
import ../../domain/entities/config
import std/[tables, sets]

suite "MemoryService types":
  test "MemoryService has expected fields":
    let cfg = defaultEngineConfig()
    var svc = MemoryService(
      cfg: cfg,
      textCache: initTable[uint64, string](),
      lowerTextCache: initTable[uint64, string](),
      tokenCache: initTable[uint64, HashSet[string]](),
      timestampCache: initTable[uint64, uint64](),
      chunkToParent: initTable[uint64, uint64](),
      parentToChunks: initTable[uint64, seq[uint64]]()
    )
    check svc.cfg.fingerprintBits == 10240
    check svc.textCache.len == 0
    check svc.lowerTextCache.len == 0
    check svc.tokenCache.len == 0
    check svc.timestampCache.len == 0
    check svc.chunkToParent.len == 0
    check svc.parentToChunks.len == 0

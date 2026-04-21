# Unit tests for Reranker algorithm.


import unittest
import ../../domain/entities/memory
import ../../domain/entities/config
import ../../domain/algorithms/reranker
import std/tables

suite "Reranker":
  test "coverageBoost is 1.0 for exact match":
    check coverageBoost("quick fox", "quick fox") == 1.0

  test "coverageBoost is 0.0 for no overlap":
    check coverageBoost("quick fox", "lazy dog") == 0.0

  test "coverageBoost is partial for some overlap":
    let boost = coverageBoost("quick brown fox", "quick fox")
    check boost > 0.0
    check boost < 1.0

  test "rerank boosts exact-match memories to top":
    var candidates = @[
      Memory(id: 0, content: "quick brown fox", score: 0.5, createdAt: 0),
      Memory(id: 1, content: "lazy dog sleeping", score: 0.6, createdAt: 0),
    ]
    var cache = initTable[uint64, string]()
    cache[0'u64] = "quick brown fox"
    cache[1'u64] = "lazy dog sleeping"
    let cfg = defaultEngineConfig()
    rerank("quick fox", candidates, cache, cfg)
    check candidates[0].id == 0'u64
    check candidates[0].score > 0.5

  test "rerank preserves order when no coverage":
    var candidates = @[
      Memory(id: 0, content: "aaa", score: 0.3, createdAt: 0),
      Memory(id: 1, content: "bbb", score: 0.5, createdAt: 0),
    ]
    var cache = initTable[uint64, string]()
    cache[0'u64] = "aaa"
    cache[1'u64] = "bbb"
    let cfg = defaultEngineConfig()
    rerank("xyz", candidates, cache, cfg)
    check candidates[0].id == 1'u64
    check candidates[0].score == 0.5

## Unit tests for Reranker algorithm.


import unittest
import ../../domain/entities/memory
import ../../domain/entities/config
import ../../domain/algorithms/reranker
import std/[tables, sets]

suite "Reranker":
  test "coverageBoost is 1.0 for exact match":
    var docTokens = initHashSet[string]()
    for t in tokenize("quick fox"):
      docTokens.incl(t)
    check coverageBoost(tokenize("quick fox"), docTokens) == 1.0

  test "coverageBoost is 0.0 for no overlap":
    var docTokens = initHashSet[string]()
    for t in tokenize("lazy dog"):
      docTokens.incl(t)
    check coverageBoost(tokenize("quick fox"), docTokens) == 0.0

  test "coverageBoost is partial for some overlap":
    var docTokens = initHashSet[string]()
    for t in tokenize("quick fox"):
      docTokens.incl(t)
    let boost = coverageBoost(tokenize("quick brown fox"), docTokens)
    check boost > 0.0
    check boost < 1.0

  test "rerank boosts exact-match memories to top":
    var candidates = @[
      Memory(id: 0, content: "quick brown fox", score: 0.5, createdAt: 0),
      Memory(id: 1, content: "lazy dog sleeping", score: 0.6, createdAt: 0),
    ]
    var cache = initTable[uint64, HashSet[string]]()
    cache[0'u64] = initHashSet[string]()
    for t in tokenize("quick brown fox"):
      cache[0'u64].incl(t)
    cache[1'u64] = initHashSet[string]()
    for t in tokenize("lazy dog sleeping"):
      cache[1'u64].incl(t)
    let cfg = defaultEngineConfig()
    rerank("quick fox", candidates, cache, cfg)
    check candidates[0].id == 0'u64
    check candidates[0].score > 0.5

  test "rerank preserves order when no coverage":
    var candidates = @[
      Memory(id: 0, content: "aaa", score: 0.3, createdAt: 0),
      Memory(id: 1, content: "bbb", score: 0.5, createdAt: 0),
    ]
    var cache = initTable[uint64, HashSet[string]]()
    cache[0'u64] = initHashSet[string]()
    for t in tokenize("aaa"):
      cache[0'u64].incl(t)
    cache[1'u64] = initHashSet[string]()
    for t in tokenize("bbb"):
      cache[1'u64].incl(t)
    let cfg = defaultEngineConfig()
    rerank("xyz", candidates, cache, cfg)
    check candidates[0].id == 1'u64
    check candidates[0].score == 0.5

# Unit tests for RRF merger algorithm.


import unittest
import ../../domain/algorithms/rrf_merger

suite "RRF merger":
  test "mergeRrf returns empty for empty inputs":
    let semantic: seq[tuple[memoryId: uint64, score: float]] = @[]
    let lexical: seq[tuple[memoryId: uint64, score: float]] = @[]
    let merged = mergeRrf(semantic, lexical, 10, 60)
    check merged.len == 0

  test "mergeRrf prefers memories in both lanes":
    let semantic = @[(memoryId: 0'u64, score: 1.0), (memoryId: 1'u64, score: 0.8)]
    let lexical = @[(memoryId: 0'u64, score: 0.9), (memoryId: 2'u64, score: 0.7)]
    let merged = mergeRrf(semantic, lexical, 10, 60)
    check merged.len == 3
    check merged[0].memoryId == 0'u64

  test "mergeRrf respects topK":
    let semantic = @[
      (memoryId: 0'u64, score: 1.0),
      (memoryId: 1'u64, score: 0.9),
      (memoryId: 2'u64, score: 0.8),
    ]
    let lexical: seq[tuple[memoryId: uint64, score: float]] = @[]
    let merged = mergeRrf(semantic, lexical, 2, 60)
    check merged.len == 2

  test "mergeRrf includes lexical-only memories":
    let semantic: seq[tuple[memoryId: uint64, score: float]] = @[]
    let lexical = @[(memoryId: 5'u64, score: 0.5)]
    let merged = mergeRrf(semantic, lexical, 10, 60)
    check merged.len == 1
    check merged[0].memoryId == 5'u64

  test "mergeRrf deduplicates across lanes":
    let semantic = @[(memoryId: 0'u64, score: 1.0)]
    let lexical = @[(memoryId: 0'u64, score: 0.9)]
    let merged = mergeRrf(semantic, lexical, 10, 60)
    var count = 0
    for m in merged:
      if m.memoryId == 0'u64:
        count.inc
    check count == 1

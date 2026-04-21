## Unit tests for Chunk entity.


import unittest
import ../../domain/entities/chunk

suite "Chunk entity":
  test "ChunkMapping stores parent and chunk ids":
    let cm = ChunkMapping(parentMemoryId: 42'u64, chunkId: 7'u64)
    check cm.parentMemoryId == 42'u64
    check cm.chunkId == 7'u64

  test "default ChunkMapping has zero ids":
    let cm = ChunkMapping()
    check cm.parentMemoryId == 0'u64
    check cm.chunkId == 0'u64

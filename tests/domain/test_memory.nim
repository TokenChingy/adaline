import unittest
import ../../domain/entities/memory

suite "Memory entity":
  test "Memory stores id, content, score, and timestamp":
    let mem = Memory(
      id: 42'u64,
      content: "hello world",
      score: 0.95,
      createdAt: 1234567890'u64
    )
    check mem.id == 42'u64
    check mem.content == "hello world"
    check mem.score == 0.95
    check mem.createdAt == 1234567890'u64

  test "default Memory has zero values":
    let mem = Memory()
    check mem.id == 0'u64
    check mem.content == ""
    check mem.score == 0.0
    check mem.createdAt == 0'u64

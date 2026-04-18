import unittest
import std/tables
import ../../domain/algorithms/lexical_index

suite "Lexical index":
  test "addMemory builds postings list":
    var idx = LexicalIndex(mu: 2000.0)
    idx.addMemory(0'u64, "the quick brown fox")
    check idx.postings.hasKey("quick")
    check idx.postings["quick"][0].memoryId == 0'u64
    check idx.postings["quick"][0].freq == 1

  test "addMemory tracks term frequency":
    var idx = LexicalIndex(mu: 2000.0)
    idx.addMemory(0'u64, "the the the")
    check idx.postings["the"][0].freq == 3

  test "searchLexical returns matching memory":
    var idx = LexicalIndex(mu: 2000.0)
    idx.addMemory(0'u64, "the quick brown fox")
    let results = searchLexical(idx, "quick", 5)
    check results.len > 0
    check results[0].memoryId == 0'u64

  test "searchLexical ranks exact match higher":
    var idx = LexicalIndex(mu: 2000.0)
    idx.addMemory(0'u64, "the quick brown fox")
    idx.addMemory(1280'u64, "the quick lazy dog")
    let results = searchLexical(idx, "quick fox", 5)
    check results.len == 2
    check results[0].memoryId == 0'u64
    check results[0].score > results[1].score

  test "searchLexical returns empty for unknown query":
    var idx = LexicalIndex(mu: 2000.0)
    let results = searchLexical(idx, "xyzabc", 5)
    check results.len == 0

  test "searchLexical respects topK":
    var idx = LexicalIndex(mu: 2000.0)
    idx.addMemory(0'u64, "the quick brown fox")
    idx.addMemory(1280'u64, "the lazy dog")
    idx.addMemory(2560'u64, "nim programming language")
    let results = searchLexical(idx, "the", 2)
    check results.len == 2

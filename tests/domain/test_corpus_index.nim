import unittest
import std/tables
import ../../domain/algorithms/corpus_index

suite "Corpus index":
  test "tokenize splits on non-alphanumeric":
    let tokens = tokenize("The quick-brown fox!!!")
    check tokens == @["the", "quick", "brown", "fox"]

  test "addMemory increments numMemories":
    var idx = CorpusIndex()
    check idx.numMemories == 0
    idx.addMemory("hello world")
    check idx.numMemories == 1
    idx.addMemory("hello again")
    check idx.numMemories == 2

  test "addMemory tracks document frequency":
    var idx = CorpusIndex()
    idx.addMemory("the quick brown fox")
    idx.addMemory("the lazy dog")
    check idx.memFreqs["the"] == 2
    check idx.memFreqs["quick"] == 1
    check idx.memFreqs["dog"] == 1

  test "idf increases for rare tokens":
    var idx = CorpusIndex()
    idx.addMemory("the quick brown fox")
    idx.addMemory("the lazy dog")
    let idfThe = idx.idf["the"]
    let idfLazy = idx.idf["lazy"]
    check idfLazy > idfThe

  test "scaledProbes returns baseProbes for empty index":
    var idx = CorpusIndex()
    check scaledProbes(idx, "anything", 4) == 4

  test "scaledProbes gives max probes for rarest token":
    var idx = CorpusIndex()
    idx.addMemory("the quick brown fox")
    idx.addMemory("the lazy dog")
    let rare = scaledProbes(idx, "lazy", 4)
    let common = scaledProbes(idx, "the", 4)
    check rare >= common
    check rare == 4

  test "scaledProbes clamps to at least 1":
    var idx = CorpusIndex()
    idx.addMemory("the the the")
    idx.addMemory("quick brown fox")
    idx.addMemory("lazy dog sleeps")
    check scaledProbes(idx, "the", 4) == 1

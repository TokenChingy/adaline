import unittest
import ../../domain/algorithms/chunker
import ../../domain/entities/config

suite "Chunker":
  let cfg = defaultEngineConfig()

  test "short text does not trigger chunking":
    let text = "hello world"
    check not shouldChunk(text, cfg)
    let chunks = splitIntoChunks(text, cfg)
    check chunks.len == 1
    check chunks[0] == text

  test "long text triggers chunking":
    var longText = ""
    for i in 0 ..< 100:
      longText.add("Machine learning is a subset of artificial intelligence. ")
    check shouldChunk(longText, cfg)

  test "splitIntoChunks returns single chunk for short text":
    let text = "The quick brown fox."
    let chunks = splitIntoChunks(text, cfg)
    check chunks.len == 1
    check chunks[0] == text

  test "splitIntoChunks splits long text into multiple chunks":
    var longText = ""
    for i in 0 ..< 100:
      longText.add("Machine learning is a subset of artificial intelligence. ")
    let chunks = splitIntoChunks(longText, cfg)
    check chunks.len > 1

  test "chunks stay below saturation threshold":
    var longText = ""
    for i in 0 ..< 100:
      longText.add("Machine learning is a subset of artificial intelligence. ")
    let chunks = splitIntoChunks(longText, cfg)
    for chunk in chunks:
      check not shouldChunk(chunk, cfg)

  test "adjacent chunks overlap by one sentence":
    var text = ""
    for i in 0 ..< 50:
      text.add("Sentence " & $i & " is about machine learning and artificial intelligence. ")
    let chunks = splitIntoChunks(text, cfg)
    if chunks.len > 1:
      let lastSentenceOfFirst = splitIntoSentences(chunks[0])[^1]
      let firstSentenceOfSecond = splitIntoSentences(chunks[1])[0]
      check lastSentenceOfFirst == firstSentenceOfSecond

  test "splitIntoSentences handles multiple delimiters":
    let text = "Hello world. How are you! This is great? Yes it is."
    let sentences = splitIntoSentences(text)
    check sentences.len == 4
    check sentences[0] == "Hello world."
    check sentences[1] == "How are you!"
    check sentences[2] == "This is great?"
    check sentences[3] == "Yes it is."

  test "empty text returns single empty chunk":
    let chunks = splitIntoChunks("", cfg)
    check chunks.len == 1
    check chunks[0] == ""

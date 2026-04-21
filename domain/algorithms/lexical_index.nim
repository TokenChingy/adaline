# Lexical inverted index with Query Likelihood Model.
# Tokenizes text on non-alphanumeric boundaries, builds postings lists,
# and scores queries using QLM with Dirichlet smoothing.


import std/[tables, strutils, math, algorithm]

type
  LexicalIndex* = object
    postings*: Table[string, seq[tuple[memoryId: uint64, freq: uint32]]]
    memLengths*: Table[uint64, uint32]
    corpusTermFreqs*: Table[string, uint64]
    totalCorpusTokens*: uint64
    mu*: float

proc tokenize*(text: string): seq[string] =
  let normalised = text.toLowerAscii()
  for token in normalised.split(AllChars - Letters - Digits):
    if token.len > 0:
      result.add(token)

proc removeMemory*(index: var LexicalIndex; memoryId: uint64; text: string) =
  let tokens = tokenize(text)
  index.totalCorpusTokens -= uint64(tokens.len)
  index.memLengths.del(memoryId)

  var termFreqs = initCountTable[string]()
  for token in tokens:
    termFreqs.inc(token)

  for term, freq in termFreqs:
    if index.corpusTermFreqs.hasKey(term):
      index.corpusTermFreqs[term] -= uint64(freq)
      if index.corpusTermFreqs[term] == 0:
        index.corpusTermFreqs.del(term)
    if index.postings.hasKey(term):
      var newPostings = newSeq[tuple[memoryId: uint64, freq: uint32]]()
      for (mid, f) in index.postings[term]:
        if mid != memoryId:
          newPostings.add((mid, f))
      index.postings[term] = newPostings
      if index.postings[term].len == 0:
        index.postings.del(term)

proc addMemory*(index: var LexicalIndex; memoryId: uint64; text: string) =
  let tokens = tokenize(text)
  index.memLengths[memoryId] = uint32(tokens.len)
  index.totalCorpusTokens += uint64(tokens.len)

  var termFreqs = initCountTable[string]()
  for token in tokens:
    termFreqs.inc(token)

  for term, freq in termFreqs:
    index.postings.mgetOrPut(term, @[]).add((memoryId, uint32(freq)))
    index.corpusTermFreqs.mgetOrPut(term, 0'u64) += uint64(freq)

proc searchLexical*(index: LexicalIndex; query: string; k: int): seq[tuple[memoryId: uint64, score: float]] =
  let qTokens = tokenize(query)
  var docScores = initTable[uint64, float]()

  for token in qTokens:
    let corpusFreq = index.corpusTermFreqs.getOrDefault(token, 0'u64)
    if corpusFreq == 0 or index.totalCorpusTokens == 0:
      continue
    let pqc = float(corpusFreq) / float(index.totalCorpusTokens)
    let denominator = index.mu * pqc

    if index.postings.hasKey(token):
      for (mid, freq) in index.postings[token]:
        docScores[mid] = docScores.getOrDefault(mid, 0.0) + ln(1.0 + float(freq) / denominator)

  var scored = newSeq[tuple[score: float, memoryId: uint64]]()
  let queryLen = float(qTokens.len)
  for mid, termScore in docScores:
    let memLen = float(index.memLengths.getOrDefault(mid, 0))
    let finalScore = termScore + queryLen * ln(index.mu / (memLen + index.mu))
    scored.add((finalScore, mid))

  scored.sort(proc(a, b: auto): int =
    if a.score > b.score: return -1
    if a.score < b.score: return 1
    return 0
  )

  let topK = min(k, scored.len)
  result = newSeq[tuple[memoryId: uint64, score: float]](topK)
  for i in 0 ..< topK:
    result[i] = (scored[i].memoryId, scored[i].score)


proc saveLexical*(index: LexicalIndex; path: string; walOffset: uint64 = 0) =
  var f = open(path, fmWrite)
  let magic = "ADLLEX01"
  discard f.writeBuffer(unsafeAddr magic[0], 8)
  discard f.writeBuffer(unsafeAddr walOffset, 8)
  let numTerms = uint64(index.postings.len)
  discard f.writeBuffer(unsafeAddr numTerms, 8)
  for term, postings in index.postings:
    let termLen = uint32(term.len)
    discard f.writeBuffer(unsafeAddr termLen, 4)
    f.write(term)
    let corpusFreq = index.corpusTermFreqs.getOrDefault(term, 0'u64)
    discard f.writeBuffer(unsafeAddr corpusFreq, 8)
    let numPostings = uint32(postings.len)
    discard f.writeBuffer(unsafeAddr numPostings, 4)
    for (mid, freq) in postings:
      discard f.writeBuffer(unsafeAddr mid, 8)
      discard f.writeBuffer(unsafeAddr freq, 4)
  let numMemLengths = uint64(index.memLengths.len)
  discard f.writeBuffer(unsafeAddr numMemLengths, 8)
  for mid, len in index.memLengths:
    discard f.writeBuffer(unsafeAddr mid, 8)
    discard f.writeBuffer(unsafeAddr len, 4)
  discard f.writeBuffer(unsafeAddr index.totalCorpusTokens, 8)
  discard f.writeBuffer(unsafeAddr index.mu, 8)
  f.close()

proc loadLexical*(path: string; walOffset: var uint64): LexicalIndex =
  walOffset = 0
  var f = open(path, fmRead)
  var magic = newString(8)
  if f.readBuffer(addr magic[0], 8) != 8 or magic != "ADLLEX01":
    f.close()
    return
  discard f.readBuffer(addr walOffset, 8)
  var numTerms: uint64
  discard f.readBuffer(addr numTerms, 8)
  for i in 0 ..< int(numTerms):
    var termLen: uint32
    discard f.readBuffer(addr termLen, 4)
    var term = newString(int(termLen))
    discard f.readBuffer(addr term[0], int(termLen))
    var corpusFreq: uint64
    discard f.readBuffer(addr corpusFreq, 8)
    result.corpusTermFreqs[term] = corpusFreq
    var numPostings: uint32
    discard f.readBuffer(addr numPostings, 4)
    var postings = newSeq[tuple[memoryId: uint64, freq: uint32]]()
    for j in 0 ..< int(numPostings):
      var mid: uint64
      var freq: uint32
      discard f.readBuffer(addr mid, 8)
      discard f.readBuffer(addr freq, 4)
      postings.add((mid, freq))
    result.postings[term] = postings
  var numMemLengths: uint64
  discard f.readBuffer(addr numMemLengths, 8)
  for i in 0 ..< int(numMemLengths):
    var mid: uint64
    var len: uint32
    discard f.readBuffer(addr mid, 8)
    discard f.readBuffer(addr len, 4)
    result.memLengths[mid] = len
  discard f.readBuffer(addr result.totalCorpusTokens, 8)
  discard f.readBuffer(addr result.mu, 8)
  f.close()

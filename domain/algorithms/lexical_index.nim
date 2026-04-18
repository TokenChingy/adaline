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

proc scoreMemory*(index: LexicalIndex; query: string; memoryId: uint64): float =
  let qTokens = tokenize(query)
  let memLen = float(index.memLengths.getOrDefault(memoryId, 0))
  let mu = index.mu

  result = 0.0
  for token in qTokens:
    let corpusFreq = index.corpusTermFreqs.getOrDefault(token, 0'u64)
    if corpusFreq == 0 or index.totalCorpusTokens == 0:
      continue
    let pqc = float(corpusFreq) / float(index.totalCorpusTokens)

    var tf = 0'u32
    if index.postings.hasKey(token):
      for (mid, freq) in index.postings[token]:
        if mid == memoryId:
          tf = freq
          break

    result += ln(1.0 + float(tf) / (mu * pqc))

  result += float(qTokens.len) * ln(mu / (memLen + mu))

proc searchLexical*(index: LexicalIndex; query: string; k: int): seq[tuple[memoryId: uint64, score: float]] =
  let qTokens = tokenize(query)
  var docScores = initTable[uint64, float]()

  # Accumulate per-memory term contributions by iterating postings directly
  for token in qTokens:
    let corpusFreq = index.corpusTermFreqs.getOrDefault(token, 0'u64)
    if corpusFreq == 0 or index.totalCorpusTokens == 0:
      continue
    let pqc = float(corpusFreq) / float(index.totalCorpusTokens)
    let denominator = index.mu * pqc

    if index.postings.hasKey(token):
      for (mid, freq) in index.postings[token]:
        docScores[mid] = docScores.getOrDefault(mid, 0.0) + ln(1.0 + float(freq) / denominator)

  # Apply length normalization and collect results
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

## Corpus-wide term frequency index.
## Tracks document frequencies across the entire corpus to compute
## IDF-squared scaling for SDR probe counts.


import std/[tables, sets, strutils, math]

type
  CorpusIndex* = object
    memFreqs*: Table[string, int]
    idf*: Table[string, float]
    maxIdf*: float
    numMemories*: int

proc tokenize*(text: string): seq[string] =
  let normalised = text.toLowerAscii()
  for token in normalised.split(AllChars - Letters - Digits):
    if token.len > 0:
      result.add(token)

proc addMemory*(index: var CorpusIndex; text: string) =
  inc index.numMemories
  var seen = initHashSet[string]()
  let tokens = tokenize(text)
  for token in tokens:
    seen.incl(token)
  for token in seen:
    index.memFreqs.mgetOrPut(token, 0).inc

  for token in seen:
    let df = index.memFreqs[token]
    let val = ln(float(index.numMemories) / float(df))
    index.idf[token] = val
    if val > index.maxIdf:
      index.maxIdf = val

proc scaledProbes*(index: CorpusIndex; feature: string; baseProbes: int): int =
  if index.numMemories == 0 or index.maxIdf <= 0.0:
    return baseProbes
  let idfVal = index.idf.getOrDefault(feature, 0.0)
  let ratio = idfVal / index.maxIdf
  let s = max(1.0, float(baseProbes) * (ratio * ratio))
  result = int(s)


proc saveCorpus*(index: CorpusIndex; path: string; walOffset: uint64 = 0) =
  var f = open(path, fmWrite)
  let magic = "ADLCORP1"
  discard f.writeBuffer(unsafeAddr magic[0], 8)
  discard f.writeBuffer(unsafeAddr walOffset, 8)
  let numMemories = int64(index.numMemories)
  let maxIdf = index.maxIdf
  discard f.writeBuffer(unsafeAddr numMemories, 8)
  discard f.writeBuffer(unsafeAddr maxIdf, 8)
  let numFreqs = uint64(index.memFreqs.len)
  discard f.writeBuffer(unsafeAddr numFreqs, 8)
  for term, freq in index.memFreqs:
    let termLen = uint32(term.len)
    discard f.writeBuffer(unsafeAddr termLen, 4)
    f.write(term)
    let freq64 = int64(freq)
    discard f.writeBuffer(unsafeAddr freq64, 8)
  f.close()

proc loadCorpus*(path: string; walOffset: var uint64): CorpusIndex =
  walOffset = 0
  var f = open(path, fmRead)
  var magic = newString(8)
  if f.readBuffer(addr magic[0], 8) != 8 or magic != "ADLCORP1":
    f.close()
    return
  discard f.readBuffer(addr walOffset, 8)
  var numMemories: int64
  var maxIdf: float
  discard f.readBuffer(addr numMemories, 8)
  discard f.readBuffer(addr maxIdf, 8)
  result.numMemories = int(numMemories)
  result.maxIdf = maxIdf
  var numFreqs: uint64
  discard f.readBuffer(addr numFreqs, 8)
  for i in 0 ..< int(numFreqs):
    var termLen: uint32
    discard f.readBuffer(addr termLen, 4)
    var term = newString(int(termLen))
    discard f.readBuffer(addr term[0], int(termLen))
    var freq64: int64
    discard f.readBuffer(addr freq64, 8)
    result.memFreqs[term] = int(freq64)
    let val = ln(float(result.numMemories) / float(freq64))
    result.idf[term] = val
  f.close()

import std/[tables, sets, strutils, math]

type
  CorpusIndex* = object
    docFreqs*: Table[string, int]
    idf*: Table[string, float]
    maxIdf*: float
    numDocs*: int

proc tokenize*(text: string): seq[string] =
  let normalised = text.toLowerAscii()
  for word in normalised.split(AllChars - Letters - Digits):
    if word.len > 0:
      result.add(word)

proc addDocument*(index: var CorpusIndex; text: string) =
  inc index.numDocs
  var seen = initHashSet[string]()
  let tokens = tokenize(text)
  for token in tokens:
    seen.incl(token)
  for token in seen:
    index.docFreqs.mgetOrPut(token, 0).inc

  # Update IDF for seen terms
  for token in seen:
    let df = index.docFreqs[token]
    let val = ln(float(index.numDocs) / float(df))
    index.idf[token] = val
    if val > index.maxIdf:
      index.maxIdf = val

proc scaledProbes*(index: CorpusIndex; feature: string; baseProbes: int): int =
  if index.numDocs == 0 or index.maxIdf <= 0.0:
    return baseProbes
  let idfVal = index.idf.getOrDefault(feature, 0.0)
  let ratio = idfVal / index.maxIdf
  let s = max(1.0, float(baseProbes) * (ratio * ratio))
  result = int(s)

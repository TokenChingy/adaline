## Sentence-aware conditional chunker.
## Splits long text into multiple chunks when any fingerprint block
## (tokens, bigrams, or XOR-context) approaches saturation.
## Splits on sentence boundaries with one-sentence overlap.


import ../entities/config
import std/strutils

proc estimateTokenCount*(text: string): int =
  result = 0
  let normalised = text.toLowerAscii()
  for token in normalised.split(AllChars - Letters - Digits):
    if token.len > 0:
      result.inc

proc estimateCharBigramCount*(text: string): int =
  let norm = text.toLowerAscii()
  if norm.len < 2:
    return 0
  result = 0
  for i in 0 ..< norm.len - 1:
    if norm[i] in Letters + Digits and norm[i + 1] in Letters + Digits:
      result.inc

proc shouldChunk*(text: string; cfg: EngineConfig): bool =
  let tokens = estimateTokenCount(text)
  let charBigrams = estimateCharBigramCount(text)
  let contextPairs = max(0, tokens - 1) * 2

  let tokenSat = float(tokens * cfg.tokenProbes + max(0, tokens - 1) * cfg.tokenBigramProbes) / float(cfg.tokenBits)
  let bigramSat = float(charBigrams * cfg.bigramProbes) / float(cfg.bigramBits)
  let contextSat = float(contextPairs * cfg.contextProbes) / float(cfg.contextBits)

  let threshold = cfg.chunkSaturationThreshold
  result = tokenSat > threshold or bigramSat > threshold or contextSat > threshold

proc splitIntoSentences*(text: string): seq[string] =
  result = @[]
  var current = ""
  for i in 0 ..< text.len:
    current.add(text[i])
    if text[i] in {'.', '!', '?'}:
      if i + 1 >= text.len or text[i + 1] in {' ', '\n', '\t', '\r'}:
        let stripped = current.strip()
        if stripped.len > 0:
          result.add(stripped)
        current = ""
  let stripped = current.strip()
  if stripped.len > 0:
    result.add(stripped)

proc splitIntoChunks*(text: string; cfg: EngineConfig): seq[string] =
  if not shouldChunk(text, cfg):
    return @[text]

  let sentences = splitIntoSentences(text)
  if sentences.len == 0:
    return @[text]
  if sentences.len == 1:
    return @[text]

  result = @[]
  var currentChunk = sentences[0]

  for i in 1 ..< sentences.len:
    let testChunk = currentChunk & " " & sentences[i]
    if shouldChunk(testChunk, cfg) and currentChunk.len > 0:
      result.add(currentChunk)
      if i > 0:
        currentChunk = sentences[i - 1] & " " & sentences[i]
        if shouldChunk(currentChunk, cfg):
          currentChunk = sentences[i]
      else:
        currentChunk = sentences[i]
    else:
      currentChunk = testChunk

  if currentChunk.len > 0:
    result.add(currentChunk)

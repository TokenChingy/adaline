## Sparse Distributed Representation (SDR) encoder.
## Converts text into a 10,240-bit fingerprint using three blocks:
## tokens (50% weight), character-bigrams (25% weight), and
## XOR-context features (25% weight). Probe counts are scaled by
## IDF-squared so rare terms receive more active bits.


import ../entities/fingerprint
import ../entities/config
import corpus_index
import std/strutils

proc hashFeature*(feature: string; seed: uint64 = 0): uint64 =
  var h = seed
  for c in feature:
    h += uint64(c)
    h += h shl 10
    h = h xor (h shr 6)
  h += h shl 3
  h = h xor (h shr 11)
  h += h shl 15
  result = h

proc probeBlock*(fp: var Fingerprint; feature: string; count, baseBit, sizeBits: int) =
  let h0 = hashFeature(feature, 0)
  for i in 0 ..< count:
    let h = h0 + uint64(i) * 0x9e3779b97f4a7c15'u64
    let pos = int(h mod uint64(sizeBits)) + baseBit
    setBit(fp, pos)

proc encodeSdr*(text: string; cfg: EngineConfig; index: CorpusIndex = CorpusIndex(); isQuery: bool = false): Fingerprint =
  result = initFingerprint()
  let normalised = text.toLowerAscii()

  let probeMult = if isQuery: cfg.queryProbeMultiplier else: 1.0

  var tokens: seq[string] = @[]
  for token in normalised.split(AllChars - Letters - Digits):
    if token.len > 0:
      tokens.add(token)

  for token in tokens:
    let baseN = scaledProbes(index, token, cfg.tokenProbes)
    let n = max(1, int(float(baseN) * probeMult))
    probeBlock(result, token, n, 0, cfg.tokenBits)
    if token.len >= 4:
      let prefix = token[0..3]
      let suffix = token[^4..^1]
      let np = max(1, int(float(scaledProbes(index, prefix, cfg.tokenProbes)) * probeMult * 0.5))
      let ns = max(1, int(float(scaledProbes(index, suffix, cfg.tokenProbes)) * probeMult * 0.5))
      probeBlock(result, prefix, np, 0, cfg.tokenBits)
      probeBlock(result, suffix, ns, 0, cfg.tokenBits)

  for i in 0 ..< tokens.len - 1:
    let bigram = tokens[i] & "_" & tokens[i + 1]
    let baseN = scaledProbes(index, bigram, cfg.tokenBigramProbes)
    let n = max(1, int(float(baseN) * probeMult))
    probeBlock(result, bigram, n, 0, cfg.tokenBits)

  if normalised.len >= 2:
    for i in 0 ..< normalised.len - 1:
      let bg = normalised[i ..< i + 2]
      if bg[0] in Letters + Digits and bg[1] in Letters + Digits:
        let baseN = scaledProbes(index, bg, cfg.bigramProbes)
        let n = max(1, int(float(baseN) * probeMult))
        probeBlock(result, bg, n, cfg.tokenBits, cfg.bigramBits)

  if tokens.len >= 2:
    var tokenHashes = newSeq[uint64](tokens.len)
    for i in 0 ..< tokens.len:
      tokenHashes[i] = hashFeature(tokens[i], 0)

    for i in 0 ..< tokens.len:
      if i > 0:
        let xorVal = tokenHashes[i] xor tokenHashes[i - 1]
        let baseN = scaledProbes(index, "L_" & tokens[i - 1], cfg.contextProbes)
        let n = max(1, int(float(baseN) * probeMult))
        for k in 0 ..< n:
          let h = xorVal + uint64(k) * 0x9e3779b97f4a7c15'u64
          let pos = int(h mod uint64(cfg.contextBits)) + cfg.tokenBits + cfg.bigramBits
          setBit(result, pos)
      if i < tokens.len - 1:
        let xorVal = tokenHashes[i] xor tokenHashes[i + 1]
        let baseN = scaledProbes(index, "R_" & tokens[i + 1], cfg.contextProbes)
        let n = max(1, int(float(baseN) * probeMult))
        for k in 0 ..< n:
          let h = xorVal + uint64(k) * 0x9e3779b97f4a7c15'u64
          let pos = int(h mod uint64(cfg.contextBits)) + cfg.tokenBits + cfg.bigramBits
          setBit(result, pos)

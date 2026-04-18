import ../entities/fingerprint
import ../entities/config
import corpus_index
import std/strutils

const
  GoldenRatio64* = 0x9e3779b97f4a7c15'u64

type SplitMix64* = uint64

proc next*(sm: var SplitMix64): uint64 =
  var z = sm + 0x9e3779b97f4a7c15'u64
  sm = z
  z = (z xor (z shr 30)) * 0xbf58476d1ce4e5b9'u64
  z = (z xor (z shr 27)) * 0x94d049bb133111eb'u64
  return z xor (z shr 31)

proc fnv1a64(s: string): uint64 =
  result = 0xcbf29ce484222325'u64
  for c in s:
    result = result xor uint64(c)
    result = result * 0x100000001b3'u64

proc hashFeature*(feature: string): uint64 =
  var sm = SplitMix64(fnv1a64(feature) + GoldenRatio64)
  result = next(sm)

proc probeBlock*(fp: var Fingerprint; feature: string; count, baseBit, sizeBits: int) =
  let h = hashFeature(feature)
  for i in 0 ..< count:
    var sm = SplitMix64(h + uint64(i) * GoldenRatio64)
    let pos = int(next(sm) mod uint64(sizeBits)) + baseBit
    setBit(fp, pos)

proc encodeSdr*(text: string; cfg: EngineConfig; index: CorpusIndex = CorpusIndex()): Fingerprint =
  result = initFingerprint()
  let normalised = text.toLowerAscii()

  # --- Block A: Tokens ---
  var tokens: seq[string] = @[]
  for token in normalised.split(AllChars - Letters - Digits):
    if token.len > 0:
      tokens.add(token)

  for token in tokens:
    let n = scaledProbes(index, token, cfg.tokenProbes)
    probeBlock(result, token, n, 0, cfg.tokenBits)

  # --- Token bigrams in token block ---
  for i in 0 ..< tokens.len - 1:
    let bigram = tokens[i] & "_" & tokens[i + 1]
    let n = scaledProbes(index, bigram, cfg.tokenBigramProbes)
    probeBlock(result, bigram, n, 0, cfg.tokenBits)

  # --- Block B: Character bigrams ---
  if normalised.len >= 2:
    for i in 0 ..< normalised.len - 1:
      let bg = normalised[i ..< i + 2]
      if bg[0] in Letters + Digits and bg[1] in Letters + Digits:
        let n = scaledProbes(index, bg, cfg.bigramProbes)
        probeBlock(result, bg, n, cfg.tokenBits, cfg.bigramBits)

  # --- Block C: XOR Bounded Neighbors ---
  if tokens.len >= 2:
    var tokenHashes = newSeq[uint64](tokens.len)
    for i in 0 ..< tokens.len:
      tokenHashes[i] = hashFeature(tokens[i])

    for i in 0 ..< tokens.len:
      if i > 0:
        let xorVal = tokenHashes[i] xor tokenHashes[i - 1]
        let n = scaledProbes(index, "L_" & tokens[i - 1], cfg.contextProbes)
        for k in 0 ..< n:
          var sm = SplitMix64(xorVal + uint64(k) * GoldenRatio64)
          let pos = int(next(sm) mod uint64(cfg.contextBits)) + cfg.tokenBits + cfg.bigramBits
          setBit(result, pos)
      if i < tokens.len - 1:
        let xorVal = tokenHashes[i] xor tokenHashes[i + 1]
        let n = scaledProbes(index, "R_" & tokens[i + 1], cfg.contextProbes)
        for k in 0 ..< n:
          var sm = SplitMix64(xorVal + uint64(k) * GoldenRatio64)
          let pos = int(next(sm) mod uint64(cfg.contextBits)) + cfg.tokenBits + cfg.bigramBits
          setBit(result, pos)

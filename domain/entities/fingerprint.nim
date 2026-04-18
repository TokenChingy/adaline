import std/bitops

const
  FingerprintBytes* = 1280
  FingerprintWords* = 160

  # Block A: Tokens — 4096 bits = 64 words (0..63)
  TokenWordStart* = 0
  TokenWordCount* = 64

  # Block B: Bigrams — 3072 bits = 48 words (64..111)
  BigramWordStart* = 64
  BigramWordCount* = 48

  # Block C: XOR Context — 3072 bits = 48 words (112..159)
  ContextWordStart* = 112
  ContextWordCount* = 48

type
  Fingerprint* = object
    bits*: array[FingerprintWords, uint64]

proc initFingerprint*(): Fingerprint =
  result.bits = default(array[FingerprintWords, uint64])

proc setBit*(fp: var Fingerprint; pos: int) =
  let word = pos shr 6
  let bit  = pos and 63
  fp.bits[word] = fp.bits[word] or (1'u64 shl bit)

proc popcountRegion*(fp: ptr Fingerprint; wordStart, wordCount: int): int =
  for i in wordStart ..< wordStart + wordCount:
    result += popcount(fp.bits[i])

proc popcountRegionAnd*(a, b: ptr Fingerprint; wordStart, wordCount: int): int =
  for i in wordStart ..< wordStart + wordCount:
    result += popcount(a.bits[i] and b.bits[i])

proc popcountRegionOr*(a, b: ptr Fingerprint; wordStart, wordCount: int): int =
  for i in wordStart ..< wordStart + wordCount:
    result += popcount(a.bits[i] or b.bits[i])

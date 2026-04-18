import std/bitops

const
  FingerprintBytes* = 1280
  FingerprintSegments* = 160

  # Block A: Tokens — 4096 bits = 64 segments (0..63)
  TokenSegmentStart* = 0
  TokenSegmentCount* = 64

  # Block B: Bigrams — 3072 bits = 48 segments (64..111)
  BigramSegmentStart* = 64
  BigramSegmentCount* = 48

  # Block C: XOR Context — 3072 bits = 48 segments (112..159)
  ContextSegmentStart* = 112
  ContextSegmentCount* = 48

type
  Fingerprint* = object
    bits*: array[FingerprintSegments, uint64]

proc initFingerprint*(): Fingerprint =
  result.bits = default(array[FingerprintSegments, uint64])

proc setBit*(fp: var Fingerprint; pos: int) =
  let segment = pos shr 6
  let bit     = pos and 63
  fp.bits[segment] = fp.bits[segment] or (1'u64 shl bit)

proc popcountRegion*(fp: ptr Fingerprint; segmentStart, segmentCount: int): int =
  for i in segmentStart ..< segmentStart + segmentCount:
    result += popcount(fp.bits[i])

proc popcountRegionAnd*(a, b: ptr Fingerprint; segmentStart, segmentCount: int): int =
  for i in segmentStart ..< segmentStart + segmentCount:
    result += popcount(a.bits[i] and b.bits[i])

proc popcountRegionOr*(a, b: ptr Fingerprint; segmentStart, segmentCount: int): int =
  for i in segmentStart ..< segmentStart + segmentCount:
    result += popcount(a.bits[i] or b.bits[i])

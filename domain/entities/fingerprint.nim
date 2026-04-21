## Fingerprint entity.
## A fixed-size 10,240-bit (1,280-byte) bitmap representing a sparse
## distributed representation. Provides bit-level get/set operations
## and population count.


import std/bitops
import config

const
  FingerprintBytes* = 1280
  FingerprintSegments* = 160

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

proc tokenSegmentStart*(cfg: EngineConfig): int = 0
proc tokenSegmentCount*(cfg: EngineConfig): int = cfg.tokenBits shr 6

proc bigramSegmentStart*(cfg: EngineConfig): int = tokenSegmentCount(cfg)
proc bigramSegmentCount*(cfg: EngineConfig): int = cfg.bigramBits shr 6

proc contextSegmentStart*(cfg: EngineConfig): int = bigramSegmentStart(cfg) + bigramSegmentCount(cfg)
proc contextSegmentCount*(cfg: EngineConfig): int = cfg.contextBits shr 6

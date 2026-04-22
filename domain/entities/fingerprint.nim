## Fingerprint entity.
## A fixed-size 10,240-bit (1,280-byte) bitmap representing a sparse
## distributed representation. Provides bit-level get/set operations,
## population count, and sparse segment compression so that inactive
## (zero) uint64 segments are not stored on disk.


import std/bitops
import config

const
  FingerprintBytes* = 1280
  FingerprintSegments* = 160
  FingerprintSegmentBytes* = 9        ## 1 byte index + 8 byte value
  FingerprintMaxCompressed* = 1 + FingerprintSegments * FingerprintSegmentBytes

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

# ---------------------------------------------------------------------------
# Sparse segment compression.
# Format: [count: uint8] [(segment_idx: uint8, value: uint64)...]
# ---------------------------------------------------------------------------

proc compressFingerprint*(fp: Fingerprint; dst: pointer): int =
  ## Compress a fingerprint by eliding zero uint64 segments.
  ## Returns the number of bytes written to ``dst``.
  var count = 0
  for i in 0 ..< FingerprintSegments:
    if fp.bits[i] != 0:
      count.inc
  cast[ptr uint8](dst)[] = uint8(count)
  var p = cast[pointer](cast[uint](dst) + 1)
  for i in 0 ..< FingerprintSegments:
    if fp.bits[i] != 0:
      cast[ptr uint8](p)[] = uint8(i)
      p = cast[pointer](cast[uint](p) + 1)
      cast[ptr uint64](p)[] = fp.bits[i]
      p = cast[pointer](cast[uint](p) + 8)
  result = 1 + count * FingerprintSegmentBytes

proc decompressFingerprint*(src: pointer; fp: ptr Fingerprint) =
  ## Decompress a segment-sparse encoding back into a full ``Fingerprint``.
  zeroMem(fp, sizeof(Fingerprint))
  let count = int(cast[ptr uint8](src)[])
  var p = cast[pointer](cast[uint](src) + 1)
  for i in 0 ..< count:
    let idx = int(cast[ptr uint8](p)[])
    p = cast[pointer](cast[uint](p) + 1)
    fp.bits[idx] = cast[ptr uint64](p)[]
    p = cast[pointer](cast[uint](p) + 8)

proc fingerprintActiveSegments*(fp: Fingerprint): int =
  ## Count how many uint64 segments are non-zero.
  for i in 0 ..< FingerprintSegments:
    if fp.bits[i] != 0:
      result.inc

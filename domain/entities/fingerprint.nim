## Fingerprint entity.
## A fixed-size 10,240-bit (1,280-byte) bitmap representing a sparse
## distributed representation. Provides bit-level get/set operations,
## population count, and adaptive compression:
##
## * Sparse format  (<= 20 active segments): [count] + [(idx, uint64)...]
## * Bitmap format  (21–157 active): [0xFE] + [20-byte bitmap] + [uint64 values...]
## * Raw format     (>= 158 active): 1,280 bytes verbatim


import std/bitops
import config

const
  FingerprintBytes* = 1280
  FingerprintSegments* = 160
  FingerprintSegmentBytes* = 9        ## 1 byte index + 8 byte value
  FingerprintMaxCompressed* = 1 + FingerprintSegments * FingerprintSegmentBytes
  FingerprintBitmapBytes* = 20        ## 160 bits = 20 bytes

  ## Format sentinels
  FpFmtBitmap* = 0xFE'u8

  ## Thresholds for format selection (cross-over points)
  FpSparseThreshold* = 20             ## <= 20: sparse wins over bitmap
  FpRawThreshold* = 158               ## >= 158: raw wins over bitmap

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
# Adaptive compression.
#
# Three formats are selected based on active-segment count:
#   Sparse : [count: uint8] [(idx: uint8, value: uint64)...]
#   Bitmap : [0xFE] [bitmap: 20 bytes] [values: uint64...]
#   Raw    : 1,280 bytes copied verbatim (size == 1280, no header)
# ---------------------------------------------------------------------------

proc compressFingerprint*(fp: Fingerprint; dst: pointer): int =
  ## Compress a fingerprint using the smallest representation.
  ## Returns the number of bytes written to ``dst``.
  var count = 0
  for i in 0 ..< FingerprintSegments:
    if fp.bits[i] != 0:
      count.inc

  # Raw format: verbatim copy when denser than bitmap
  if count >= FpRawThreshold:
    copyMem(dst, addr fp.bits[0], FingerprintBytes)
    return FingerprintBytes

  # Sparse format: count + (idx, value) pairs
  if count <= FpSparseThreshold:
    cast[ptr uint8](dst)[] = uint8(count)
    var p = cast[pointer](cast[uint](dst) + 1)
    for i in 0 ..< FingerprintSegments:
      if fp.bits[i] != 0:
        cast[ptr uint8](p)[] = uint8(i)
        p = cast[pointer](cast[uint](p) + 1)
        cast[ptr uint64](p)[] = fp.bits[i]
        p = cast[pointer](cast[uint](p) + 8)
    return 1 + count * FingerprintSegmentBytes

  # Bitmap format: 0xFE sentinel + 20-byte bitmap + values
  cast[ptr uint8](dst)[] = FpFmtBitmap
  var bitmapP = cast[ptr UncheckedArray[byte]](cast[pointer](cast[uint](dst) + 1))
  zeroMem(bitmapP, FingerprintBitmapBytes)
  for i in 0 ..< FingerprintSegments:
    if fp.bits[i] != 0:
      bitmapP[i shr 3] = bitmapP[i shr 3] or (1'u8 shl (i and 7))
  var p = cast[pointer](cast[uint](dst) + 1 + FingerprintBitmapBytes)
  for i in 0 ..< FingerprintSegments:
    if fp.bits[i] != 0:
      cast[ptr uint64](p)[] = fp.bits[i]
      p = cast[pointer](cast[uint](p) + 8)
  result = 1 + FingerprintBitmapBytes + count * 8

proc decompressFingerprint*(src: pointer; fp: ptr Fingerprint) =
  ## Decompress an adaptive encoding back into a full ``Fingerprint``.
  zeroMem(fp, sizeof(Fingerprint))
  let fmt = cast[ptr uint8](src)[]

  # Sparse format
  if fmt < FpFmtBitmap:
    let count = int(fmt)
    var p = cast[pointer](cast[uint](src) + 1)
    for i in 0 ..< count:
      let idx = int(cast[ptr uint8](p)[])
      p = cast[pointer](cast[uint](p) + 1)
      fp.bits[idx] = cast[ptr uint64](p)[]
      p = cast[pointer](cast[uint](p) + 8)
    return

  # Bitmap format: iterate set bits via CTZ instead of testing all 160
  if fmt == FpFmtBitmap:
    let bitmap = cast[ptr UncheckedArray[byte]](cast[pointer](cast[uint](src) + 1))
    var p = cast[pointer](cast[uint](src) + 1 + FingerprintBitmapBytes)
    for byteIdx in 0 ..< FingerprintBitmapBytes:
      var b = bitmap[byteIdx]
      while b != 0:
        let bit = countTrailingZeroBits(b)
        let seg = (byteIdx shl 3) + bit
        fp.bits[seg] = cast[ptr uint64](p)[]
        p = cast[pointer](cast[uint](p) + 8)
        b = b and (b - 1)
    return

  # Raw format: caller should have checked size == 1280 and copied directly.
  # This path should not normally be reached.
  copyMem(fp, src, FingerprintBytes)

proc fingerprintActiveSegments*(fp: Fingerprint): int =
  ## Count how many uint64 segments are non-zero.
  for i in 0 ..< FingerprintSegments:
    if fp.bits[i] != 0:
      result.inc

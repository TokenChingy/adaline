import ../entities/fingerprint
import ../entities/config
import std/[tables, sets]

# GoldFinger-style direct fingerprint banding.
# Instead of MinHash signatures, we partition the fingerprint's uint64 segments
# into bands and hash each band directly. Two similar fingerprints will share
# many identical segments, giving high collision probability in LSH buckets.

type
  FingerprintLshIndex* = object
    cfg*: EngineConfig
    buckets*: Table[(int, uint64), seq[uint64]]

# Build a band hash directly from fingerprint segments.
proc bandHash*(fp: ptr Fingerprint; bandId, rows: int): uint64 =
  var h: uint64 = uint64(bandId) * 0x9e3779b97f4a7c15'u64
  for r in 0 ..< rows:
    let segIdx = (bandId * rows + r) mod FingerprintSegments
    h = h xor fp.bits[segIdx]
    h = h * 0xbf58476d1ce4e5b9'u64
  result = h

proc initLshIndex*(cfg: EngineConfig): FingerprintLshIndex =
  result.cfg = cfg
  result.buckets = initTable[(int, uint64), seq[uint64]]()

proc insertLsh*(index: var FingerprintLshIndex; fp: ptr Fingerprint; memoryId: uint64) =
  for b in 0 ..< index.cfg.lshBands:
    let bh = bandHash(fp, b, index.cfg.lshRows)
    let key = (b, bh)
    index.buckets.mgetOrPut(key, @[]).add(memoryId)

proc removeLsh*(index: var FingerprintLshIndex; fp: ptr Fingerprint; memoryId: uint64) =
  for b in 0 ..< index.cfg.lshBands:
    let bh = bandHash(fp, b, index.cfg.lshRows)
    let key = (b, bh)
    if index.buckets.hasKey(key):
      var newIds = newSeq[uint64]()
      for id in index.buckets[key]:
        if id != memoryId:
          newIds.add(id)
      index.buckets[key] = newIds
      if index.buckets[key].len == 0:
        index.buckets.del(key)

proc queryLsh*(index: FingerprintLshIndex; fp: ptr Fingerprint): seq[uint64] =
  var seen = initHashSet[uint64]()
  for b in 0 ..< index.cfg.lshBands:
    let bh = bandHash(fp, b, index.cfg.lshRows)
    let key = (b, bh)
    if index.buckets.hasKey(key):
      for id in index.buckets[key]:
        if id notin seen:
          seen.incl(id)
          result.add(id)

import ../entities/fingerprint
import ../entities/config
import sdr_encoder
import std/[tables, sets]

const
  MaxHashFunctions = 100

type
  MinHashLshIndex* = object
    cfg*: EngineConfig
    buckets*: Table[(int, uint64), seq[uint64]]

proc minHashValue(funcId, bitPos: int): uint64 =
  var sm = SplitMix64(uint64(funcId) * 0x9e3779b97f4a7c15'u64 + uint64(bitPos) * 0xbf58476d1ce4e5b9'u64)
  result = next(sm)

proc computeSignature*(fp: ptr Fingerprint; cfg: EngineConfig): array[MaxHashFunctions, uint64] =
  for i in 0 ..< cfg.minHashFunctions:
    result[i] = high(uint64)

  for segmentIdx in 0 ..< FingerprintSegments:
    let segment = fp.bits[segmentIdx]
    if segment == 0:
      continue
    for bitOffset in 0 ..< 64:
      if (segment and (1'u64 shl bitOffset)) != 0:
        let pos = segmentIdx * 64 + bitOffset
        for h in 0 ..< cfg.minHashFunctions:
          let hv = minHashValue(h, pos)
          if hv < result[h]:
            result[h] = hv

proc bandHash*(signature: array[MaxHashFunctions, uint64]; bandId, rows: int): uint64 =
  var h: uint64 = uint64(bandId) * 0x9e3779b97f4a7c15'u64
  for r in 0 ..< rows:
    h = h xor signature[bandId * rows + r]
    h = h * 0xbf58476d1ce4e5b9'u64
  result = h

proc initLshIndex*(cfg: EngineConfig): MinHashLshIndex =
  result.cfg = cfg
  result.buckets = initTable[(int, uint64), seq[uint64]]()

proc insertLsh*(index: var MinHashLshIndex; signature: array[MaxHashFunctions, uint64]; memoryId: uint64) =
  for b in 0 ..< index.cfg.lshBands:
    let bh = bandHash(signature, b, index.cfg.lshRows)
    let key = (b, bh)
    index.buckets.mgetOrPut(key, @[]).add(memoryId)

proc queryLsh*(index: MinHashLshIndex; signature: array[MaxHashFunctions, uint64]): seq[uint64] =
  var seen = initHashSet[uint64]()
  for b in 0 ..< index.cfg.lshBands:
    let bh = bandHash(signature, b, index.cfg.lshRows)
    let key = (b, bh)
    if index.buckets.hasKey(key):
      for id in index.buckets[key]:
        if id notin seen:
          seen.incl(id)
          result.add(id)

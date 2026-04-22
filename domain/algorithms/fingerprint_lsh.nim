## Banded Fingerprint LSH (GoldFinger-style).
## Partitions the 10,240-bit fingerprint into bands and rows for
## locality-sensitive hashing. Each band hash seeds candidate retrieval.
## AND-construction (minHits >= 2) filters weak single-band collisions.


import ../entities/fingerprint
import ../entities/config
import std/tables

type
  FingerprintLshIndex* = object
    cfg*: EngineConfig
    buckets*: Table[(int, uint64), seq[uint64]]
    visited*: seq[int]    ## epoch array for O(1) query deduplication
    counts*: seq[uint8]   ## collision counts per ID for current epoch
    epoch*: int

proc bandHash*(fp: ptr Fingerprint; bandId, rows: int): uint64 =
  var h: uint64 = uint64(bandId) * 0x9e3779b97f4a7c15'u64
  ## Fast path for default config where bands * rows == segments (no wrap).
  if rows == 2 and bandId * 2 + 2 <= FingerprintSegments:
    h = h xor fp.bits[bandId * 2]
    h = h * 0xbf58476d1ce4e5b9'u64
    h = h xor fp.bits[bandId * 2 + 1]
    h = h * 0xbf58476d1ce4e5b9'u64
  else:
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
  let id = int(memoryId)
  if id >= index.visited.len:
    index.visited.setLen(max(id + 1, index.visited.len * 2))
  if id >= index.counts.len:
    index.counts.setLen(max(id + 1, index.counts.len * 2))

proc removeLsh*(index: var FingerprintLshIndex; fp: ptr Fingerprint; memoryId: uint64) =
  for b in 0 ..< index.cfg.lshBands:
    let bh = bandHash(fp, b, index.cfg.lshRows)
    let key = (b, bh)
    if index.buckets.hasKey(key):
      var ids = index.buckets[key]
      var writePos = 0
      for readPos in 0 ..< ids.len:
        if ids[readPos] != memoryId:
          ids[writePos] = ids[readPos]
          inc writePos
      ids.setLen(writePos)
      if ids.len == 0:
        index.buckets.del(key)

proc queryLsh*(index: var FingerprintLshIndex; fp: ptr Fingerprint): seq[uint64] =
  inc index.epoch
  if index.epoch == 0:
    index.epoch = 1
    for i in 0 ..< index.visited.len:
      index.visited[i] = 0

  for b in 0 ..< index.cfg.lshBands:
    let bh = bandHash(fp, b, index.cfg.lshRows)
    let ids = index.buckets.getOrDefault((b, bh))
    for id in ids:
      let slot = int(id)
      if slot >= index.visited.len:
        index.visited.setLen(max(slot + 1, index.visited.len * 2))
      if slot >= index.counts.len:
        index.counts.setLen(max(slot + 1, index.counts.len * 2))
      if index.visited[slot] != index.epoch:
        index.visited[slot] = index.epoch
        index.counts[slot] = 1
      else:
        index.counts[slot] += 1
        if index.counts[slot] == 2:
          result.add(id)

proc saveLsh*(index: FingerprintLshIndex; path: string; walOffset: uint64 = 0) =
  var f = open(path, fmWrite)
  let magic = "ADLSH001"
  discard f.writeBuffer(unsafeAddr magic[0], 8)
  discard f.writeBuffer(unsafeAddr walOffset, 8)
  let numBuckets = uint64(index.buckets.len)
  discard f.writeBuffer(unsafeAddr numBuckets, 8)
  for key, ids in index.buckets:
    let (bandId, hash) = key
    let bandId32 = int32(bandId)
    let numIds = uint32(ids.len)
    discard f.writeBuffer(unsafeAddr bandId32, 4)
    discard f.writeBuffer(unsafeAddr hash, 8)
    discard f.writeBuffer(unsafeAddr numIds, 4)
    for id in ids:
      discard f.writeBuffer(unsafeAddr id, 8)
  f.close()

proc loadLsh*(cfg: EngineConfig; path: string; walOffset: var uint64): FingerprintLshIndex =
  result = initLshIndex(cfg)
  walOffset = 0
  var f = open(path, fmRead)
  var magic = newString(8)
  if f.readBuffer(addr magic[0], 8) != 8 or magic != "ADLSH001":
    f.close()
    return
  discard f.readBuffer(addr walOffset, 8)
  var numBuckets: uint64
  discard f.readBuffer(addr numBuckets, 8)
  result.buckets = initTable[(int, uint64), seq[uint64]](initialSize = int(numBuckets))
  for i in 0 ..< int(numBuckets):
    var bandId32: int32
    var hash: uint64
    var numIds: uint32
    discard f.readBuffer(addr bandId32, 4)
    discard f.readBuffer(addr hash, 8)
    discard f.readBuffer(addr numIds, 4)
    var ids = newSeq[uint64](int(numIds))
    for j in 0 ..< int(numIds):
      discard f.readBuffer(addr ids[j], 8)
    result.buckets[(int(bandId32), hash)] = ids
  f.close()

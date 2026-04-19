import std/bitops
import ../entities/fingerprint
import ../entities/config

type RegionStats = object
  inter: int
  aPop: int
  bPop: int
  unionPop: int

proc regionStats(a, b: ptr Fingerprint; segmentStart, segmentCount: int): RegionStats =
  for i in segmentStart ..< segmentStart + segmentCount:
    let abits = a.bits[i]
    let bbits = b.bits[i]
    result.inter += popcount(abits and bbits)
    result.aPop += popcount(abits)
    result.bPop += popcount(bbits)
    result.unionPop += popcount(abits or bbits)

proc jaccardRegion*(a, b: ptr Fingerprint; segmentStart, segmentCount: int): float =
  let s = regionStats(a, b, segmentStart, segmentCount)
  if s.unionPop == 0:
    return 1.0
  return float(s.inter) / float(s.unionPop)

proc weightedJaccard*(a, b: ptr Fingerprint; cfg: EngineConfig): float =
  let tStart = tokenSegmentStart(cfg)
  let tCount = tokenSegmentCount(cfg)
  let bStart = bigramSegmentStart(cfg)
  let bCount = bigramSegmentCount(cfg)
  let cStart = contextSegmentStart(cfg)
  let cCount = contextSegmentCount(cfg)

  let jToken = jaccardRegion(a, b, tStart, tCount)
  let jBigram = jaccardRegion(a, b, bStart, bCount)
  let jContext = jaccardRegion(a, b, cStart, cCount)

  result = cfg.tokenWeight * jToken + cfg.bigramWeight * jBigram + cfg.contextWeight * jContext

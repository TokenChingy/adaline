import std/bitops
import ../entities/fingerprint
import ../entities/config
import std/math

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

proc diceRegion*(a, b: ptr Fingerprint; segmentStart, segmentCount: int): float =
  let s = regionStats(a, b, segmentStart, segmentCount)
  let denom = s.aPop + s.bPop
  if denom == 0:
    return 1.0
  return float(2 * s.inter) / float(denom)

proc cosineRegion*(a, b: ptr Fingerprint; segmentStart, segmentCount: int): float =
  let s = regionStats(a, b, segmentStart, segmentCount)
  let denom = sqrt(float(s.aPop) * float(s.bPop))
  if denom == 0.0:
    return 0.0
  return float(s.inter) / denom

proc overlapRegion*(a, b: ptr Fingerprint; segmentStart, segmentCount: int): float =
  let s = regionStats(a, b, segmentStart, segmentCount)
  let denom = min(s.aPop, s.bPop)
  if denom == 0:
    # Both empty → identical; one empty and one non-empty → 0 overlap
    if s.aPop == 0 and s.bPop == 0:
      return 1.0
    return 0.0
  return float(s.inter) / float(denom)

proc weightedJaccard*(a, b: ptr Fingerprint; cfg: EngineConfig): float =
  let tStart = tokenSegmentStart(cfg)
  let tCount = tokenSegmentCount(cfg)
  let bStart = bigramSegmentStart(cfg)
  let bCount = bigramSegmentCount(cfg)
  let cStart = contextSegmentStart(cfg)
  let cCount = contextSegmentCount(cfg)

  var jToken, jBigram, jContext: float

  case cfg.similarityMetric:
    of "dice":
      jToken = diceRegion(a, b, tStart, tCount)
      jBigram = diceRegion(a, b, bStart, bCount)
      jContext = diceRegion(a, b, cStart, cCount)
    of "cosine":
      jToken = cosineRegion(a, b, tStart, tCount)
      jBigram = cosineRegion(a, b, bStart, bCount)
      jContext = cosineRegion(a, b, cStart, cCount)
    of "overlap":
      jToken = overlapRegion(a, b, tStart, tCount)
      jBigram = overlapRegion(a, b, bStart, bCount)
      jContext = overlapRegion(a, b, cStart, cCount)
    else:
      jToken = jaccardRegion(a, b, tStart, tCount)
      jBigram = jaccardRegion(a, b, bStart, bCount)
      jContext = jaccardRegion(a, b, cStart, cCount)

  result = cfg.tokenWeight * jToken + cfg.bigramWeight * jBigram + cfg.contextWeight * jContext

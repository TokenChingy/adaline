## Regional weighted Jaccard similarity.
## Computes Jaccard over three fingerprint regions (tokens, bigrams,
## context) with configurable block weights, using hardware popcount
## for fast bit-level intersection and union.


import std/bitops
import ../entities/fingerprint
import ../entities/config

proc jaccardRegion*(a, b: ptr Fingerprint; segmentStart, segmentCount: int): float =
  var inter = 0
  var unionPop = 0
  for i in segmentStart ..< segmentStart + segmentCount:
    let abits = a.bits[i]
    let bbits = b.bits[i]
    inter += popcount(abits and bbits)
    unionPop += popcount(abits or bbits)
  if unionPop == 0:
    return 1.0
  return float(inter) / float(unionPop)

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

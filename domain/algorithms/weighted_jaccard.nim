import std/bitops
import ../entities/fingerprint
import ../entities/config

proc jaccardRegion*(a, b: ptr Fingerprint; wordStart, wordCount: int): float =
  var inter = 0
  var unionPop = 0
  for i in wordStart ..< wordStart + wordCount:
    inter += popcount(a.bits[i] and b.bits[i])
    unionPop += popcount(a.bits[i] or b.bits[i])
  if unionPop == 0:
    return 1.0
  return float(inter) / float(unionPop)

proc weightedJaccard*(a, b: ptr Fingerprint; cfg: EngineConfig): float =
  let jToken = jaccardRegion(a, b, TokenWordStart, TokenWordCount)
  let jBigram = jaccardRegion(a, b, BigramWordStart, BigramWordCount)
  let jContext = jaccardRegion(a, b, ContextWordStart, ContextWordCount)
  result = cfg.tokenWeight * jToken + cfg.bigramWeight * jBigram + cfg.contextWeight * jContext

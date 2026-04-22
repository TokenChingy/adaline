## Sparse Distributed Representation (SDR) encoder.
## Converts text into a 10,240-bit fingerprint using three blocks:
## tokens (50% weight), character-bigrams (25% weight), and
## XOR-context features (25% weight). Probe counts are scaled by
## IDF-squared so rare terms receive more active bits.
##
## Documents are optionally filtered to the top-K tokens by IDF
## (``cfg.maxTokenFeatures``). Adjacency features (token bigrams,
## context) are kept only when both neighbours survive the cut.
## Character bigrams are skipped in filtered mode because they have
## no natural per-token IDF score. Queries are never filtered.


import ../entities/fingerprint
import ../entities/config
import corpus_index
import std/[strutils, tables]

proc hashFeature*(feature: string; seed: uint64 = 0): uint64 =
  var h = seed
  for c in feature:
    h += uint64(c)
    h += h shl 10
    h = h xor (h shr 6)
  h += h shl 3
  h = h xor (h shr 11)
  h += h shl 15
  result = h

proc probeBlock*(fp: var Fingerprint; feature: string; count, baseBit, sizeBits: int) =
  let h0 = hashFeature(feature, 0)
  for i in 0 ..< count:
    let h = h0 + uint64(i) * 0x9e3779b97f4a7c15'u64
    let pos = int(h mod uint64(sizeBits)) + baseBit
    setBit(fp, pos)

proc encodeSdr*(text: string; cfg: EngineConfig; index: CorpusIndex = CorpusIndex(); isQuery: bool = false): Fingerprint =
  result = initFingerprint()
  let normalised = text.toLowerAscii()

  let probeMult = if isQuery: cfg.queryProbeMultiplier else: 1.0

  var tokens: seq[string] = @[]
  for token in normalised.split(AllChars - Letters - Digits):
    if token.len > 0:
      tokens.add(token)

  # ---------------------------------------------------------------------------
  # Optional top-K token filtering (documents only).
  # ---------------------------------------------------------------------------
  var selected = newSeq[bool](tokens.len)
  for i in 0 ..< tokens.len:
    selected[i] = true

  let doFilter = cfg.maxTokenFeatures > 0 and not isQuery and index.numMemories > 0
  if doFilter:
    # Bounded min-heap selection: O(n log K) instead of O(n log n) full sort.
    type HeapItem = tuple[idx: int, score: float]
    var heap = newSeq[HeapItem]()  # min-heap by score

    proc siftUp(h: var seq[HeapItem]; pos: int) =
      var i = pos
      while i > 0:
        let parent = (i - 1) shr 1
        if h[parent].score <= h[i].score: break
        swap(h[parent], h[i])
        i = parent

    proc siftDown(h: var seq[HeapItem]; pos: int) =
      let n = h.len
      var i = pos
      while true:
        let left = (i shl 1) + 1
        let right = left + 1
        var smallest = i
        if left < n and h[left].score < h[smallest].score:
          smallest = left
        if right < n and h[right].score < h[smallest].score:
          smallest = right
        if smallest == i: break
        swap(h[i], h[smallest])
        i = smallest

    let keep = cfg.maxTokenFeatures
    for i in 0 ..< tokens.len:
      let idfVal = index.idf.getOrDefault(tokens[i], 0.0)
      if heap.len < keep:
        heap.add((i, idfVal))
        siftUp(heap, heap.len - 1)
      elif idfVal > heap[0].score:
        heap[0] = (i, idfVal)
        siftDown(heap, 0)

    for i in 0 ..< selected.len:
      selected[i] = false
    for item in heap:
      selected[item.idx] = true

  # ---------------------------------------------------------------------------
  # Token block (individual tokens + prefix/suffix).
  # ---------------------------------------------------------------------------
  for i in 0 ..< tokens.len:
    if not selected[i]:
      continue
    let token = tokens[i]
    let baseN = scaledProbes(index, token, cfg.tokenProbes)
    let n = max(1, int(float(baseN) * probeMult))
    probeBlock(result, token, n, 0, cfg.tokenBits)
    if token.len >= 4:
      let prefix = token[0..3]
      let suffix = token[^4..^1]
      let np = max(1, int(float(scaledProbes(index, prefix, cfg.tokenProbes)) * probeMult * 0.5))
      let ns = max(1, int(float(scaledProbes(index, suffix, cfg.tokenProbes)) * probeMult * 0.5))
      probeBlock(result, prefix, np, 0, cfg.tokenBits)
      probeBlock(result, suffix, ns, 0, cfg.tokenBits)

  # ---------------------------------------------------------------------------
  # Token bigrams (adjacent tokens).
  # ---------------------------------------------------------------------------
  for i in 0 ..< tokens.len - 1:
    if doFilter and (not selected[i] or not selected[i + 1]):
      continue
    let bigram = tokens[i] & "_" & tokens[i + 1]
    let baseN = scaledProbes(index, bigram, cfg.tokenBigramProbes)
    let n = max(1, int(float(baseN) * probeMult))
    probeBlock(result, bigram, n, 0, cfg.tokenBits)

  # ---------------------------------------------------------------------------
  # Character bigrams — skipped when top-K filtering is active because
  # they have no natural per-token IDF score.
  # ---------------------------------------------------------------------------
  if not doFilter and normalised.len >= 2:
    for i in 0 ..< normalised.len - 1:
      let bg = normalised[i ..< i + 2]
      if bg[0] in Letters + Digits and bg[1] in Letters + Digits:
        let baseN = scaledProbes(index, bg, cfg.bigramProbes)
        let n = max(1, int(float(baseN) * probeMult))
        probeBlock(result, bg, n, cfg.tokenBits, cfg.bigramBits)

  # ---------------------------------------------------------------------------
  # XOR-context features.
  # ---------------------------------------------------------------------------
  if tokens.len >= 2:
    var tokenHashes = newSeq[uint64](tokens.len)
    for i in 0 ..< tokens.len:
      tokenHashes[i] = hashFeature(tokens[i], 0)

    for i in 0 ..< tokens.len:
      if i > 0:
        if doFilter and (not selected[i] or not selected[i - 1]):
          discard
        else:
          let xorVal = tokenHashes[i] xor tokenHashes[i - 1]
          let baseN = scaledProbes(index, "L_" & tokens[i - 1], cfg.contextProbes)
          let n = max(1, int(float(baseN) * probeMult))
          for k in 0 ..< n:
            let h = xorVal + uint64(k) * 0x9e3779b97f4a7c15'u64
            let pos = int(h mod uint64(cfg.contextBits)) + cfg.tokenBits + cfg.bigramBits
            setBit(result, pos)
      if i < tokens.len - 1:
        if doFilter and (not selected[i] or not selected[i + 1]):
          discard
        else:
          let xorVal = tokenHashes[i] xor tokenHashes[i + 1]
          let baseN = scaledProbes(index, "R_" & tokens[i + 1], cfg.contextProbes)
          let n = max(1, int(float(baseN) * probeMult))
          for k in 0 ..< n:
            let h = xorVal + uint64(k) * 0x9e3779b97f4a7c15'u64
            let pos = int(h mod uint64(cfg.contextBits)) + cfg.tokenBits + cfg.bigramBits
            setBit(result, pos)

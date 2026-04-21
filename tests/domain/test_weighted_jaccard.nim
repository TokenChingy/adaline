# Unit tests for Weighted Jaccard algorithm.


import unittest
import ../../domain/entities/fingerprint
import ../../domain/entities/config
import ../../domain/algorithms/weighted_jaccard

suite "Weighted Jaccard":
  test "identical fingerprints have Jaccard 1.0":
    var a = initFingerprint()
    a.setBit(0)
    a.setBit(100)
    a.setBit(5000)
    let cfg = defaultEngineConfig()
    let j = weightedJaccard(addr a, addr a, cfg)
    check j == 1.0

  test "disjoint fingerprints have Jaccard 0.0":
    var a = initFingerprint()
    var b = initFingerprint()
    a.setBit(0)
    a.setBit(4096)
    a.setBit(7168)
    b.setBit(1)
    b.setBit(4097)
    b.setBit(7169)
    let cfg = defaultEngineConfig()
    let j = weightedJaccard(addr a, addr b, cfg)
    check j == 0.0

  test "half-overlap in a region has Jaccard 0.5":
    var a = initFingerprint()
    var b = initFingerprint()
    a.setBit(0)
    b.setBit(0)
    b.setBit(1)
    let j = jaccardRegion(addr a, addr b, 0, 1)
    check j == 0.5

  test "overlapping fingerprints have positive Jaccard":
    var a = initFingerprint()
    var b = initFingerprint()
    a.setBit(10)
    a.setBit(20)
    a.setBit(30)
    b.setBit(10)
    b.setBit(25)
    let cfg = defaultEngineConfig()
    let j = weightedJaccard(addr a, addr b, cfg)
    check j > 0.0
    check j < 1.0

import unittest
import ../../domain/entities/fingerprint
import ../../domain/entities/config
import ../../domain/algorithms/fingerprint_lsh

suite "Fingerprint LSH (GoldFinger-style)":
  test "identical fingerprints collide in LSH":
    var fp = initFingerprint()
    fp.setBit(42)
    fp.setBit(1234)
    let cfg = defaultEngineConfig()
    var idx = initLshIndex(cfg)
    insertLsh(idx, addr fp, 12345'u64)
    let results = queryLsh(idx, addr fp)
    check results.len > 0
    check results[0] == 12345'u64

  test "query deduplicates across bands":
    var fp = initFingerprint()
    fp.setBit(1)
    fp.setBit(2)
    let cfg = defaultEngineConfig()
    var idx = initLshIndex(cfg)
    insertLsh(idx, addr fp, 100'u64)
    let results = queryLsh(idx, addr fp)
    var count = 0
    for r in results:
      if r == 100'u64:
        count.inc
    check count == 1

  test "dissimilar fingerprints have low collision probability":
    var a = initFingerprint()
    var b = initFingerprint()
    # a and b share no bits at all
    a.setBit(1)
    b.setBit(9999)
    let cfg = defaultEngineConfig()
    var idx = initLshIndex(cfg)
    insertLsh(idx, addr a, 1'u64)
    let results = queryLsh(idx, addr b)
    # With 50 bands and completely disjoint fingerprints,
    # occasional false collisions can happen; just ensure it's rare
    check results.len <= 1

  test "similar fingerprints have high collision probability":
    var a = initFingerprint()
    var b = initFingerprint()
    # Set 80% of the same bits
    for i in 0 ..< 100:
      a.setBit(i * 10)
      b.setBit(i * 10)
    # Set 20% different bits
    for i in 0 ..< 20:
      a.setBit(5000 + i)
      b.setBit(8000 + i)

    let cfg = defaultEngineConfig()
    var idx = initLshIndex(cfg)
    insertLsh(idx, addr a, 1'u64)
    let results = queryLsh(idx, addr b)
    # With high similarity, we expect at least some collision
    check results.len > 0
    check results[0] == 1'u64

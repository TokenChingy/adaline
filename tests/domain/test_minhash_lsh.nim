import unittest
import ../../domain/entities/fingerprint
import ../../domain/entities/config
import ../../domain/algorithms/minhash_lsh

suite "MinHash LSH":
  test "computeSignature is deterministic":
    var fp = initFingerprint()
    fp.setBit(10)
    fp.setBit(100)
    fp.setBit(500)
    let cfg = defaultEngineConfig()
    let sig1 = computeSignature(addr fp, cfg)
    let sig2 = computeSignature(addr fp, cfg)
    for i in 0 ..< cfg.minHashFunctions:
      check sig1[i] == sig2[i]

  test "identical fingerprints have identical signatures":
    var a = initFingerprint()
    var b = initFingerprint()
    a.setBit(10)
    a.setBit(20)
    b.setBit(10)
    b.setBit(20)
    let cfg = defaultEngineConfig()
    let sigA = computeSignature(addr a, cfg)
    let sigB = computeSignature(addr b, cfg)
    for i in 0 ..< cfg.minHashFunctions:
      check sigA[i] == sigB[i]

  test "different fingerprints have different signatures":
    var a = initFingerprint()
    var b = initFingerprint()
    a.setBit(10)
    b.setBit(9999)
    let cfg = defaultEngineConfig()
    let sigA = computeSignature(addr a, cfg)
    let sigB = computeSignature(addr b, cfg)
    var same = 0
    for i in 0 ..< cfg.minHashFunctions:
      if sigA[i] == sigB[i]:
        same.inc
    check same < cfg.minHashFunctions

  test "insert and query return inserted memory":
    var fp = initFingerprint()
    fp.setBit(42)
    fp.setBit(1234)
    let cfg = defaultEngineConfig()
    let sig = computeSignature(addr fp, cfg)
    var idx = initLshIndex(cfg)
    insertLsh(idx, sig, 12345'u64)
    let results = queryLsh(idx, sig)
    check results.len > 0
    check results[0] == 12345'u64

  test "query deduplicates across bands":
    var fp = initFingerprint()
    fp.setBit(1)
    fp.setBit(2)
    let cfg = defaultEngineConfig()
    let sig = computeSignature(addr fp, cfg)
    var idx = initLshIndex(cfg)
    insertLsh(idx, sig, 100'u64)
    let results = queryLsh(idx, sig)
    var count = 0
    for r in results:
      if r == 100'u64:
        count.inc
    check count == 1

  test "dissimilar fingerprints may not collide":
    var a = initFingerprint()
    var b = initFingerprint()
    a.setBit(1)
    b.setBit(9999)
    let cfg = defaultEngineConfig()
    let sigA = computeSignature(addr a, cfg)
    let sigB = computeSignature(addr b, cfg)
    var idx = initLshIndex(cfg)
    insertLsh(idx, sigA, 1'u64)
    let results = queryLsh(idx, sigB)
    check results.len == 0

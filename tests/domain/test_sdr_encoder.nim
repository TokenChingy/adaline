# Unit tests for SDR encoder algorithm.


import unittest
import std/bitops
import ../../domain/entities/fingerprint
import ../../domain/entities/config
import ../../domain/algorithms/sdr_encoder
import ../../domain/algorithms/weighted_jaccard

suite "SDR encoder":
  let cfg = defaultEngineConfig()

  test "encodeSdr is deterministic":
    let a = encodeSdr("hello world", cfg)
    let b = encodeSdr("hello world", cfg)
    var diff = 0
    for i in 0 ..< FingerprintSegments:
      diff += popcount(a.bits[i] xor b.bits[i])
    check diff == 0

  test "different strings produce different fingerprints":
    let a = encodeSdr("nim programming language", cfg)
    let b = encodeSdr("python programming language", cfg)
    var diff = 0
    for i in 0 ..< FingerprintSegments:
      diff += popcount(a.bits[i] xor b.bits[i])
    check diff > 0

  test "similar strings have higher Jaccard than unrelated strings":
    let fox = encodeSdr("the quick brown fox", cfg)
    let quickFox = encodeSdr("quick brown fox", cfg)
    let lazyDog = encodeSdr("lazy dog sleeping", cfg)
    let sim1 = weightedJaccard(addr fox, addr quickFox, cfg)
    let sim2 = weightedJaccard(addr fox, addr lazyDog, cfg)
    check sim1 > sim2

  test "XOR-bound context distinguishes token sense":
    let a = encodeSdr("fox quick", cfg)
    let b = encodeSdr("fox lazy", cfg)
    let c = encodeSdr("fox quick", cfg)
    let sim1 = weightedJaccard(addr a, addr c, cfg)
    let sim2 = weightedJaccard(addr a, addr b, cfg)
    check sim1 > sim2

## Unit tests for Dense Encoder algorithm.


import unittest
import ../../domain/entities/fingerprint
import ../../domain/entities/config
import ../../domain/algorithms/dense_encoder

suite "Dense encoder":
  let cfg = defaultEngineConfig()

  test "l2Norm of empty vector is zero":
    let vec: seq[float32] = @[]
    check l2Norm(vec) == 0.0'f32

  test "l2Norm of unit vector is one":
    let vec = @[1.0'f32, 0.0'f32, 0.0'f32]
    check abs(l2Norm(vec) - 1.0'f32) < 1e-6'f32

  test "l2Norm scales correctly":
    let vec = @[3.0'f32, 4.0'f32]
    check abs(l2Norm(vec) - 5.0'f32) < 1e-6'f32

  test "encodeDense returns empty fingerprint for zero vector":
    let vec = @[0.0'f32, 0.0'f32, 0.0'f32]
    let fp = encodeDense(vec, cfg)
    var allZero = true
    for w in fp.bits:
      if w != 0'u64:
        allZero = false
        break
    check allZero

  test "encodeDense is deterministic":
    let vec = @[0.1'f32, 0.3'f32, 0.5'f32, 0.2'f32]
    let fp1 = encodeDense(vec, cfg)
    let fp2 = encodeDense(vec, cfg)
    for i in 0 ..< fp1.bits.len:
      check fp1.bits[i] == fp2.bits[i]

  test "encodeDense sets bits in fingerprint":
    let vec = @[0.1'f32, 0.3'f32, 0.5'f32, 0.2'f32]
    let fp = encodeDense(vec, cfg)
    var active = 0
    for w in fp.bits:
      if w != 0'u64:
        active.inc
    check active > 0

  test "encodeDense produces different fingerprints for different inputs":
    let vecA = @[1.0'f32, 0.0'f32, 0.0'f32, 0.0'f32]
    let vecB = @[0.0'f32, 0.0'f32, 0.0'f32, 1.0'f32]
    let fpA = encodeDense(vecA, cfg)
    let fpB = encodeDense(vecB, cfg)
    var same = true
    for i in 0 ..< fpA.bits.len:
      if fpA.bits[i] != fpB.bits[i]:
        same = false
        break
    check not same

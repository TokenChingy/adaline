import unittest
import ../../domain/entities/fingerprint
import ../../domain/entities/config

suite "Fingerprint entity":
  test "initFingerprint creates zeroed bitmap":
    let fp = initFingerprint()
    for w in fp.bits:
      check w == 0'u64

  test "setBit sets correct segment and bit":
    var fp = initFingerprint()
    setBit(fp, 0)
    check fp.bits[0] == 1'u64
    setBit(fp, 65)
    check fp.bits[1] == 2'u64
    setBit(fp, 10239)
    check fp.bits[159] == (1'u64 shl 63)

  test "popcountRegionAnd/Or work correctly":
    let cfg = defaultEngineConfig()
    var a = initFingerprint()
    var b = initFingerprint()
    setBit(a, 0)
    setBit(a, 1)
    setBit(b, 1)
    setBit(b, 2)
    let inter = popcountRegionAnd(addr a, addr b, tokenSegmentStart(cfg), tokenSegmentCount(cfg))
    let unionPop = popcountRegionOr(addr a, addr b, tokenSegmentStart(cfg), tokenSegmentCount(cfg))
    check inter == 1
    check unionPop == 3

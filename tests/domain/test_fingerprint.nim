import unittest
import ../../domain/entities/fingerprint

suite "Fingerprint entity":
  test "initFingerprint creates zeroed bitmap":
    let fp = initFingerprint()
    for w in fp.bits:
      check w == 0'u64

  test "setBit sets correct word and bit":
    var fp = initFingerprint()
    setBit(fp, 0)
    check fp.bits[0] == 1'u64
    setBit(fp, 65)
    check fp.bits[1] == 2'u64
    setBit(fp, 10239)
    check fp.bits[159] == (1'u64 shl 63)

  test "popcountRegionAnd/Or work correctly":
    var a = initFingerprint()
    var b = initFingerprint()
    setBit(a, 0)
    setBit(a, 1)
    setBit(b, 1)
    setBit(b, 2)
    let inter = popcountRegionAnd(addr a, addr b, TokenWordStart, TokenWordCount)
    let unionPop = popcountRegionOr(addr a, addr b, TokenWordStart, TokenWordCount)
    check inter == 1
    check unionPop == 3

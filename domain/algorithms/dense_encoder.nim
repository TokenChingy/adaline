import ../entities/fingerprint
import ../entities/config
import sdr_encoder
import std/algorithm
import std/math

proc l2Norm*(vec: openArray[float32]): float32 =
  var s = 0.0'f32
  for v in vec: s += v * v
  sqrt(s)

proc encodeDense*(vec: openArray[float32]; cfg: EngineConfig;
                  k: int = 128; probes: int = 4): Fingerprint =
  ## Converts a dense float32 feature vector into an Adaline fingerprint.
  ##
  ## Uses k-WTA (k-Winners Take All): the top-k dimensions by absolute value
  ## are treated as active features and probed into the fingerprint space
  ## using the same hashFeature/probeBlock primitives as the text SDR encoder.
  ## This makes the encoder domain-agnostic — vision, audio, tabular, or any
  ## L2-normalised float32 vector can be indexed and searched with identical
  ## infrastructure.
  ##
  ## k:      number of winning dimensions (top-k by |value|); default 128
  ## probes: base probes per winner, scaled up for high-magnitude activations
  result = initFingerprint()
  let norm = l2Norm(vec)
  if norm < 1e-8'f32:
    return

  var indexed = newSeq[(float32, int)](vec.len)
  for i in 0 ..< vec.len:
    indexed[i] = (abs(vec[i] / norm), i)
  indexed.sort(proc(a, b: (float32, int)): int = cmp(b[0], a[0]))

  let winners = min(k, indexed.len)
  for rank in 0 ..< winners:
    let (mag, idx) = indexed[rank]
    let sign = if vec[idx] / norm >= 0'f32: "p" else: "n"
    let feature = $idx & sign
    let n = max(1, min(probes, int(mag * float32(probes) * 3.0'f32)))
    probeBlock(result, feature, n, 0, cfg.fingerprintBits)

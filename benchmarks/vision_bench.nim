## Vision / dense-vector benchmark using the proper dense-vector use-cases.
## Loads CIFAR-10 CNN features from benchmarks/data/cifar10_features.bin
## and benchmarks classification, few-shot scaling, incremental class addition,
## and open-set detection via insertDense / searchDense.

import std/[os, strutils, times, math, random, bitops]
import ../domain/services/memory/types
import ../domain/services/memory/init
import ../domain/services/memory/insert_dense
import ../domain/services/memory/search_dense
import ../domain/entities/config
import ../domain/entities/fingerprint
import ../infrastructure/mmapped_storage
import ../use_cases/insert_dense
import ../use_cases/search_dense

const
  FeatureFilePath = "benchmarks/data/cifar10_features.bin"
  Magic = "ADLV"
  HeaderSize = 32

type
  FeatureDataset = object
    trainFeatures: seq[float32]
    trainLabels: seq[uint32]
    testFeatures: seq[float32]
    testLabels: seq[uint32]
    nTrain: int
    nTest: int
    featDim: int
    nClasses: int

proc loadFeatureFile(path: string): FeatureDataset =
  if not fileExists(path):
    raise newException(IOError, "Feature file not found: " & path &
      "\nGenerate it with: python3 benchmarks/dump_features.py")

  let f = open(path, fmRead)
  defer: f.close()

  var header: array[HeaderSize, uint8]
  if f.readBuffer(addr header[0], HeaderSize) != HeaderSize:
    raise newException(IOError, "Feature file too small for header")

  proc readU32(offset: int): uint32 =
    result = uint32(header[offset]) or
             (uint32(header[offset+1]) shl 8) or
             (uint32(header[offset+2]) shl 16) or
             (uint32(header[offset+3]) shl 24)

  if cast[string](header[0..<4]) != Magic:
    raise newException(IOError, "Invalid feature file magic")

  let version = readU32(4)
  if version != 1:
    raise newException(IOError, "Unsupported feature file version: " & $version)

  result.nTrain   = int(readU32(8))
  result.nTest    = int(readU32(12))
  result.featDim  = int(readU32(16))
  result.nClasses = int(readU32(20))

  let trainFeatBytes  = result.nTrain * result.featDim * 4
  let trainLabelBytes = result.nTrain * 4
  let testFeatBytes   = result.nTest * result.featDim * 4
  let testLabelBytes  = result.nTest * 4
  let totalBytes = HeaderSize + trainFeatBytes + trainLabelBytes + testFeatBytes + testLabelBytes

  let fileSize = getFileSize(path)
  if fileSize != totalBytes:
    raise newException(IOError, "Feature file size mismatch: expected " & $totalBytes & " got " & $fileSize)

  result.trainFeatures = newSeq[float32](result.nTrain * result.featDim)
  result.trainLabels   = newSeq[uint32](result.nTrain)
  result.testFeatures  = newSeq[float32](result.nTest * result.featDim)
  result.testLabels    = newSeq[uint32](result.nTest)

  if f.readBuffer(addr result.trainFeatures[0], trainFeatBytes) != trainFeatBytes:
    raise newException(IOError, "Failed to read train features")
  if f.readBuffer(addr result.trainLabels[0], trainLabelBytes) != trainLabelBytes:
    raise newException(IOError, "Failed to read train labels")
  if f.readBuffer(addr result.testFeatures[0], testFeatBytes) != testFeatBytes:
    raise newException(IOError, "Failed to read test features")
  if f.readBuffer(addr result.testLabels[0], testLabelBytes) != testLabelBytes:
    raise newException(IOError, "Failed to read test labels")

proc getFeatureVec(ds: FeatureDataset; idx: int): seq[float32] =
  let offset = idx * ds.featDim
  result = ds.trainFeatures[offset ..< offset + ds.featDim]

proc getTestVec(ds: FeatureDataset; idx: int): seq[float32] =
  let offset = idx * ds.featDim
  result = ds.testFeatures[offset ..< offset + ds.featDim]

proc insertVector(service: var MemoryService; vec: openArray[float32];
                  labels: var seq[int]; label: int): uint64 =
  let id = insertDense(service, InsertDenseInput(vec: @vec)).memoryId
  if int(id) >= labels.len:
    labels.setLen(int(id) + 1)
  labels[int(id)] = label
  result = id

proc searchVector(service: var MemoryService; vec: openArray[float32]; k: int): seq[tuple[memoryId: uint64, score: float]] =
  result = searchDense(service, SearchDenseInput(vec: @vec, topK: k)).results

proc top1Accuracy(predictions, truth: seq[int]): float =
  var correct = 0
  for i in 0 ..< predictions.len:
    if predictions[i] == truth[i]:
      correct.inc
  return float(correct) / float(max(predictions.len, 1))

proc auroc(posScores, negScores: seq[float]): float =
  if posScores.len == 0 or negScores.len == 0:
    return 0.5
  var count = 0.0
  var ties = 0.0
  for p in posScores:
    for n in negScores:
      if p > n: count += 1.0
      elif p == n: ties += 1.0
  return (count + 0.5 * ties) / float(posScores.len * negScores.len)

proc buildClassIndices(labels: seq[uint32]; nClasses: int): seq[seq[int]] =
  result = newSeq[seq[int]](nClasses)
  for i in 0 ..< labels.len:
    let c = int(labels[i])
    if c < nClasses:
      result[c].add(i)

proc benchClassify(ds: FeatureDataset; cfg: EngineConfig; nQueryPerClass: int = 50; seed: int = 42) =
  echo "\n=== Benchmark 1: classify (1-shot, ", nQueryPerClass, " queries/class) ===\n"
  var rng = initRand(seed)
  let trainClassIdx = buildClassIndices(ds.trainLabels, ds.nClasses)
  let testClassIdx  = buildClassIndices(ds.testLabels,  ds.nClasses)

  var protoVectors: seq[seq[float32]]
  var protoLabels: seq[int]
  for c in 0 ..< ds.nClasses:
    if trainClassIdx[c].len > 0:
      let chosen = trainClassIdx[c][rng.rand(trainClassIdx[c].len - 1)]
      protoVectors.add(getFeatureVec(ds, chosen))
      protoLabels.add(c)

  var queries: seq[seq[float32]]
  var truth: seq[int]
  for c in 0 ..< ds.nClasses:
    let n = min(nQueryPerClass, testClassIdx[c].len)
    var idxCopy = testClassIdx[c]
    rng.shuffle(idxCopy)
    for i in 0 ..< n:
      queries.add(getTestVec(ds, idxCopy[i]))
      truth.add(c)

  var densePreds = newSeq[int](queries.len)
  let tDense = cpuTime()
  for qi, qvec in queries:
    var bestScore = -1.0
    var bestLabel = -1
    for pi in 0 ..< protoVectors.len:
      var dot = 0.0'f32
      for j in 0 ..< ds.featDim:
        dot += qvec[j] * protoVectors[pi][j]
      if float(dot) > bestScore:
        bestScore = float(dot)
        bestLabel = protoLabels[pi]
    densePreds[qi] = bestLabel
  let msDense = (cpuTime() - tDense) * 1000.0 / float(queries.len)

  var benchDir = getCurrentDir() / "benchmarks" / "data" / "vision_classify"
  removeDir(benchDir)
  var service = initMemoryService(benchDir, cfg)
  var labels: seq[int] = @[]
  for pi in 0 ..< protoVectors.len:
    discard insertVector(service, protoVectors[pi], labels, protoLabels[pi])

  var sparsePreds = newSeq[int](queries.len)
  let tSparse = cpuTime()
  for qi, qvec in queries:
    let res = searchVector(service, qvec, ds.nClasses)
    var classBest = newSeq[float](ds.nClasses)
    for i in 0 ..< ds.nClasses:
      classBest[i] = -1.0
    for (mid, score) in res:
      let lbl = labels[int(mid)]
      if lbl >= 0 and lbl < ds.nClasses:
        if score > classBest[lbl]:
          classBest[lbl] = score
    var bestLabel = 0
    var bestScore = classBest[0]
    for c in 1 ..< ds.nClasses:
      if classBest[c] > bestScore:
        bestScore = classBest[c]
        bestLabel = c
    sparsePreds[qi] = bestLabel
  let msSparse = (cpuTime() - tSparse) * 1000.0 / float(queries.len)

  echo "  Method                Accuracy    ms/query"
  echo "  -----------------------------------------"
  echo "  dense (cosine)        ", formatFloat(top1Accuracy(densePreds, truth) * 100, ffDecimal, 1).align(6), "%     ", formatFloat(msDense, ffDecimal, 3)
  echo "  sparse (LSH)          ", formatFloat(top1Accuracy(sparsePreds, truth) * 100, ffDecimal, 1).align(6), "%     ", formatFloat(msSparse, ffDecimal, 3)


proc benchFewshot(ds: FeatureDataset; cfg: EngineConfig;
                  shots: seq[int] = @[1, 2, 5, 10, 20];
                  nQueryPerClass: int = 50; seed: int = 42) =
  echo "\n=== Benchmark 2: fewshot (accuracy vs prototype count) ===\n"
  var rng = initRand(seed)
  let trainClassIdx = buildClassIndices(ds.trainLabels, ds.nClasses)
  let testClassIdx  = buildClassIndices(ds.testLabels,  ds.nClasses)

  var queries: seq[seq[float32]]
  var truth: seq[int]
  for c in 0 ..< ds.nClasses:
    let n = min(nQueryPerClass, testClassIdx[c].len)
    var idxCopy = testClassIdx[c]
    rng.shuffle(idxCopy)
    for i in 0 ..< n:
      queries.add(getTestVec(ds, idxCopy[i]))
      truth.add(c)

  echo "  Shots     dense       sparse (LSH)"
  echo "  -----------------------------------"
  for nProto in shots:
    var denseProtos: seq[seq[float32]]
    var denseLabels: seq[int]
    for c in 0 ..< ds.nClasses:
      let n = min(nProto, trainClassIdx[c].len)
      var idxCopy = trainClassIdx[c]
      rng.shuffle(idxCopy)
      for i in 0 ..< n:
        denseProtos.add(getFeatureVec(ds, idxCopy[i]))
        denseLabels.add(c)

    var densePreds = newSeq[int](queries.len)
    for qi, qvec in queries:
      var bestScore = -1.0
      var bestLabel = -1
      for pi in 0 ..< denseProtos.len:
        var dot = 0.0'f32
        for j in 0 ..< ds.featDim:
          dot += qvec[j] * denseProtos[pi][j]
        if float(dot) > bestScore:
          bestScore = float(dot)
          bestLabel = denseLabels[pi]
      densePreds[qi] = bestLabel

    var benchDir = getCurrentDir() / "benchmarks" / "data" / ("vision_fewshot_" & $nProto)
    removeDir(benchDir)
    var service = initMemoryService(benchDir, cfg)
    var labels: seq[int] = @[]
    for c in 0 ..< ds.nClasses:
      let n = min(nProto, trainClassIdx[c].len)
      var idxCopy = trainClassIdx[c]
      rng.shuffle(idxCopy)
      for i in 0 ..< n:
        discard insertVector(service, getFeatureVec(ds, idxCopy[i]), labels, c)

    var sparsePreds = newSeq[int](queries.len)
    for qi, qvec in queries:
      let res = searchVector(service, qvec, ds.nClasses)
      var classBest = newSeq[float](ds.nClasses)
      for i in 0 ..< ds.nClasses:
        classBest[i] = -1.0
      for (mid, score) in res:
        let lbl = labels[int(mid)]
        if lbl >= 0 and lbl < ds.nClasses:
          if score > classBest[lbl]:
            classBest[lbl] = score
      var bestLabel = 0
      var bestScore = classBest[0]
      for c in 1 ..< ds.nClasses:
        if classBest[c] > bestScore:
          bestScore = classBest[c]
          bestLabel = c
      sparsePreds[qi] = bestLabel

    echo "  ", align($nProto, 4),
         "      ", formatFloat(top1Accuracy(densePreds, truth) * 100, ffDecimal, 1).align(5), "%",
         "     ", formatFloat(top1Accuracy(sparsePreds, truth) * 100, ffDecimal, 1).align(5), "%"


proc benchIncremental(ds: FeatureDataset; cfg: EngineConfig;
                      nProto: int = 5; nQueryPerClass: int = 50;
                      steps: seq[int] = @[2, 4, 6, 8, 10]; seed: int = 42) =
  echo "\n=== Benchmark 3: incremental (", nProto, " prototypes/class, ", nQueryPerClass,
       " queries/class, classes ", steps[0], "->", steps[^1], ") ===\n"
  var rng = initRand(seed)
  let trainClassIdx = buildClassIndices(ds.trainLabels, ds.nClasses)
  let testClassIdx  = buildClassIndices(ds.testLabels,  ds.nClasses)

  echo "  Top-1 Accuracy:"
  echo "  Classes    dense       sparse (LSH)      ms/query"
  echo "  -------------------------------------------------"
  for nCls in steps:
    var queries: seq[seq[float32]]
    var truth: seq[int]
    for c in 0 ..< nCls:
      let n = min(nQueryPerClass, testClassIdx[c].len)
      var idxCopy = testClassIdx[c]
      rng.shuffle(idxCopy)
      for i in 0 ..< n:
        queries.add(getTestVec(ds, idxCopy[i]))
        truth.add(c)

    var denseProtos: seq[seq[float32]]
    var denseLabels: seq[int]
    for c in 0 ..< nCls:
      let n = min(nProto, trainClassIdx[c].len)
      var idxCopy = trainClassIdx[c]
      rng.shuffle(idxCopy)
      for i in 0 ..< n:
        denseProtos.add(getFeatureVec(ds, idxCopy[i]))
        denseLabels.add(c)

    var densePreds = newSeq[int](queries.len)
    let tDense = cpuTime()
    for qi, qvec in queries:
      var bestScore = -1.0
      var bestLabel = -1
      for pi in 0 ..< denseProtos.len:
        var dot = 0.0'f32
        for j in 0 ..< ds.featDim:
          dot += qvec[j] * denseProtos[pi][j]
        if float(dot) > bestScore:
          bestScore = float(dot)
          bestLabel = denseLabels[pi]
      densePreds[qi] = bestLabel
    discard (cpuTime() - tDense) * 1000.0 / float(queries.len)

    var benchDir = getCurrentDir() / "benchmarks" / "data" / ("vision_inc_" & $nCls)
    removeDir(benchDir)
    var service = initMemoryService(benchDir, cfg)
    var labels: seq[int] = @[]
    for c in 0 ..< nCls:
      let n = min(nProto, trainClassIdx[c].len)
      var idxCopy = trainClassIdx[c]
      rng.shuffle(idxCopy)
      for i in 0 ..< n:
        discard insertVector(service, getFeatureVec(ds, idxCopy[i]), labels, c)

    var sparsePreds = newSeq[int](queries.len)
    let tSparse = cpuTime()
    for qi, qvec in queries:
      let res = searchVector(service, qvec, nCls)
      var classBest = newSeq[float](nCls)
      for i in 0 ..< nCls:
        classBest[i] = -1.0
      for (mid, score) in res:
        let lbl = labels[int(mid)]
        if lbl >= 0 and lbl < nCls:
          if score > classBest[lbl]:
            classBest[lbl] = score
      var bestLabel = 0
      var bestScore = classBest[0]
      for c in 1 ..< nCls:
        if classBest[c] > bestScore:
          bestScore = classBest[c]
          bestLabel = c
      sparsePreds[qi] = bestLabel
    let msSparse = (cpuTime() - tSparse) * 1000.0 / float(queries.len)

    echo "  ", align($nCls, 4),
         "       ", formatFloat(top1Accuracy(densePreds, truth) * 100, ffDecimal, 1).align(5), "%",
         "     ", formatFloat(top1Accuracy(sparsePreds, truth) * 100, ffDecimal, 1).align(5), "%",
         "      ", formatFloat(msSparse, ffDecimal, 3)


proc benchOpenset(ds: FeatureDataset; cfg: EngineConfig;
                  nKnown: int = 6; nProto: int = 5;
                  nQueryPerClass: int = 50; seed: int = 42) =
  echo "\n=== Benchmark 4: openset (enrol ", nKnown, " classes, query all ", ds.nClasses, ") ===\n"
  var rng = initRand(seed)
  let trainClassIdx = buildClassIndices(ds.trainLabels, ds.nClasses)
  let testClassIdx  = buildClassIndices(ds.testLabels,  ds.nClasses)

  var denseProtos: seq[seq[float32]]
  var denseLabels: seq[int]
  for c in 0 ..< nKnown:
    let n = min(nProto, trainClassIdx[c].len)
    var idxCopy = trainClassIdx[c]
    rng.shuffle(idxCopy)
    for i in 0 ..< n:
      denseProtos.add(getFeatureVec(ds, idxCopy[i]))
      denseLabels.add(c)

  var queries: seq[seq[float32]]
  var truth: seq[int]
  var isKnown: seq[bool]
  for c in 0 ..< ds.nClasses:
    let n = min(nQueryPerClass, testClassIdx[c].len)
    var idxCopy = testClassIdx[c]
    rng.shuffle(idxCopy)
    for i in 0 ..< n:
      queries.add(getTestVec(ds, idxCopy[i]))
      truth.add(c)
      isKnown.add(c < nKnown)

  var densePos, denseNeg: seq[float]
  var denseKnownPreds, denseKnownTruth: seq[int]
  for qi, qvec in queries:
    var bestScore = -1.0
    var bestLabel = -1
    for pi in 0 ..< denseProtos.len:
      var dot = 0.0'f32
      for j in 0 ..< ds.featDim:
        dot += qvec[j] * denseProtos[pi][j]
      if float(dot) > bestScore:
        bestScore = float(dot)
        bestLabel = denseLabels[pi]
    if isKnown[qi]:
      densePos.add(bestScore)
      denseKnownPreds.add(bestLabel)
      denseKnownTruth.add(truth[qi])
    else:
      denseNeg.add(bestScore)

  var benchDir = getCurrentDir() / "benchmarks" / "data" / "vision_openset"
  removeDir(benchDir)
  var service = initMemoryService(benchDir, cfg)
  var labels: seq[int] = @[]
  for c in 0 ..< nKnown:
    let n = min(nProto, trainClassIdx[c].len)
    var idxCopy = trainClassIdx[c]
    rng.shuffle(idxCopy)
    for i in 0 ..< n:
      discard insertVector(service, getFeatureVec(ds, idxCopy[i]), labels, c)

  var sparsePos, sparseNeg: seq[float]
  var sparseKnownPreds, sparseKnownTruth: seq[int]
  for qi, qvec in queries:
    let res = searchVector(service, qvec, nKnown)
    var classBest = newSeq[float](nKnown)
    for i in 0 ..< nKnown:
      classBest[i] = -1.0
    for (mid, score) in res:
      let lbl = labels[int(mid)]
      if lbl >= 0 and lbl < nKnown:
        if score > classBest[lbl]:
          classBest[lbl] = score
    var bestScore = classBest[0]
    var bestLabel = 0
    for c in 1 ..< nKnown:
      if classBest[c] > bestScore:
        bestScore = classBest[c]
        bestLabel = c
    if isKnown[qi]:
      sparsePos.add(bestScore)
      sparseKnownPreds.add(bestLabel)
      sparseKnownTruth.add(truth[qi])
    else:
      sparseNeg.add(bestScore)

  let denseAuc = auroc(densePos, denseNeg)
  let sparseAuc = auroc(sparsePos, sparseNeg)
  let denseKnownAcc = top1Accuracy(denseKnownPreds, denseKnownTruth)
  let sparseKnownAcc = top1Accuracy(sparseKnownPreds, sparseKnownTruth)

  echo "  Known classes: 0..", nKnown-1, "  (", densePos.len, " known queries, ", denseNeg.len, " novel queries)\n"
  echo "  Method                AUROC    Known acc"
  echo "  ----------------------------------------"
  echo "  dense (cosine)        ", formatFloat(denseAuc, ffDecimal, 3).align(6),
       "   ", formatFloat(denseKnownAcc * 100, ffDecimal, 1).align(5), "%"
  echo "  sparse (LSH)          ", formatFloat(sparseAuc, ffDecimal, 3).align(6),
       "   ", formatFloat(sparseKnownAcc * 100, ffDecimal, 1).align(5), "%"


proc benchFootprint(ds: FeatureDataset; cfg: EngineConfig) =
  echo "\n=== Benchmark 5: footprint (memory comparison) ===\n"
  let denseBytes = ds.featDim * 4
  echo "  Input dimension : ", ds.featDim
  echo "  Fingerprint size: ", cfg.fingerprintBits, " bits = ", cfg.fingerprintBytes, " bytes"
  echo "  Dense storage   : ", ds.featDim, " dims x 4 bytes = ", denseBytes, " bytes/vector\n"

  echo "  Encoder          Active bits    Sparse bytes    Ratio"
  echo "  -------------------------------------------------------"

  var rng = initRand(42)
  var sampleVecs = newSeq[seq[float32]](50)
  for i in 0 ..< 50:
    var v = newSeq[float32](ds.featDim)
    var norm = 0.0'f32
    for j in 0 ..< ds.featDim:
      v[j] = float32(rng.rand(2.0) - 1.0)
      norm += v[j] * v[j]
    norm = sqrt(norm)
    if norm > 1e-8:
      for j in 0 ..< ds.featDim:
        v[j] = v[j] / norm
    sampleVecs[i] = v

  var service = initMemoryService(getCurrentDir() / "benchmarks" / "data" / "vision_footprint", defaultEngineConfig())
  var totalBits = 0
  for v in sampleVecs:
    let id = insertDense(service, InsertDenseInput(vec: v)).memoryId
    var fp: Fingerprint
    service.storage.readFingerprint(id, addr fp)
    var bits = 0
    for i in 0 ..< FingerprintSegments:
      bits += popcount(fp.bits[i])
    totalBits += bits
  let avgBits = float(totalBits) / float(sampleVecs.len)
  let sparseBytes = avgBits * 2.0
  let ratio = float(denseBytes) / max(sparseBytes, 1.0)

  echo "  kwta_k128        ", formatFloat(avgBits, ffDecimal, 1).align(11),
       " ", formatFloat(sparseBytes, ffDecimal, 0).align(13),
       " ", formatFloat(ratio, ffDecimal, 1).align(6), "x"
  echo "\n  Sparse bytes assumes uint16 active-bit position list."
  echo "  Adaline stores fingerprints as packed bit arrays (",
       cfg.fingerprintBytes, " bytes each — fixed width)."


proc printUsage() =
  echo """
Adaline Vision Benchmark (Nim)

Uses the proper dense-vector use-cases (insertDense, searchDense)
to benchmark retrieval through the actual Adaline engine.

Usage:
  vision_bench <feature-file>

Options:
  --help        Show this help message.

Real features (requires Python + PyTorch):
  python3 benchmarks/dump_features.py
  vision_bench benchmarks/data/cifar10_features.bin
"""

proc main() =
  let args = commandLineParams()
  if args.len > 0 and args[0] in ["help", "--help", "-h"]:
    printUsage()
    return

  var ds: FeatureDataset
  if args.len > 0:
    echo "Loading feature file: ", args[0]
    ds = loadFeatureFile(args[0])
  else:
    if fileExists(FeatureFilePath):
      echo "Loading feature file: ", FeatureFilePath
      ds = loadFeatureFile(FeatureFilePath)
    else:
      echo "No feature file found. Generate it with:"
      echo "  python3 benchmarks/dump_features.py"
      quit(1)

  echo "Train: ", ds.nTrain, " x ", ds.featDim, "-dim"
  echo "Test:  ", ds.nTest,  " x ", ds.featDim, "-dim"
  echo "Classes: ", ds.nClasses

  let cfg = defaultEngineConfig()

  benchFootprint(ds, cfg)
  benchClassify(ds, cfg)
  benchFewshot(ds, cfg)
  benchIncremental(ds, cfg)
  benchOpenset(ds, cfg)

  echo "\nDone."

when isMainModule:
  main()

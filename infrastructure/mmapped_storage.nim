import ../domain/entities/fingerprint
import ../domain/entities/hnsw_node
import std/[memfiles, os]

const
  InitialFpCount = 1024
  InitialGraphCount = 1024

type
  MmappedStorage* = ref object
    dataDir*: string
    walFile*: File
    walSize*: uint64
    fpMemFile*: MemFile
    fpMem*: pointer
    fpCapacity*: uint64
    graphMemFile*: MemFile
    graphMem*: pointer
    graphCapacity*: uint64

proc extendFile(path: string; newSize: uint64) =
  var f = system.open(path, fmReadWrite)
  f.setFilePos(int64(newSize - 1))
  f.write('\0')
  f.close()

proc growFpStore*(storage: MmappedStorage; minCount: uint64) =
  if minCount <= storage.fpCapacity:
    return
  var newCapacity = storage.fpCapacity * 2
  if newCapacity == 0:
    newCapacity = InitialFpCount
  while newCapacity < minCount:
    newCapacity *= 2

  storage.fpMemFile.close()
  let fpPath = storage.dataDir / "fingerprints.bin"
  let newSize = newCapacity * uint64(FingerprintBytes)
  extendFile(fpPath, newSize)
  storage.fpMemFile = memfiles.open(fpPath, mode = fmReadWrite, mappedSize = int(newSize))
  storage.fpMem = storage.fpMemFile.mem
  storage.fpCapacity = newCapacity

proc growGraphStore*(storage: MmappedStorage; minCount: uint64) =
  if minCount <= storage.graphCapacity:
    return
  var newCapacity = storage.graphCapacity * 2
  if newCapacity == 0:
    newCapacity = InitialGraphCount
  while newCapacity < minCount:
    newCapacity *= 2

  storage.graphMemFile.close()
  let graphPath = storage.dataDir / "graph.bin"
  let newSize = newCapacity * uint64(sizeof(HnswNode))
  extendFile(graphPath, newSize)
  storage.graphMemFile = memfiles.open(graphPath, mode = fmReadWrite, mappedSize = int(newSize))
  storage.graphMem = storage.graphMemFile.mem
  storage.graphCapacity = newCapacity

proc initStorage*(dataDir: string): MmappedStorage =
  result = MmappedStorage(dataDir: dataDir)
  createDir(dataDir)

  # WAL (append-only, variable length — use standard File I/O)
  let walPath = dataDir / "wal.bin"
  result.walFile = system.open(walPath, fmAppend)
  result.walSize = uint64(getFileSize(walPath))

  # Fingerprint store: flat array of 1280-byte fingerprints
  let fpPath = dataDir / "fingerprints.bin"
  if not fileExists(fpPath):
    let initialSize = uint64(InitialFpCount * FingerprintBytes)
    extendFile(fpPath, initialSize)
  let fpSize = uint64(getFileSize(fpPath))
  result.fpMemFile = memfiles.open(fpPath, mode = fmReadWrite, mappedSize = int(fpSize))
  result.fpMem = result.fpMemFile.mem
  result.fpCapacity = fpSize div uint64(FingerprintBytes)

  # Graph store: flat array of HnswNode structs
  let graphPath = dataDir / "graph.bin"
  if not fileExists(graphPath):
    let initialSize = uint64(InitialGraphCount * sizeof(HnswNode))
    extendFile(graphPath, initialSize)
  let graphSize = uint64(getFileSize(graphPath))
  result.graphMemFile = memfiles.open(graphPath, mode = fmReadWrite, mappedSize = int(graphSize))
  result.graphMem = result.graphMemFile.mem
  result.graphCapacity = graphSize div uint64(sizeof(HnswNode))

proc appendWal*(storage: MmappedStorage; memoryId: uint64; text: string): uint64 =
  result = storage.walSize
  var textLen = uint32(text.len)
  discard storage.walFile.writeBuffer(unsafeAddr memoryId, sizeof(uint64))
  discard storage.walFile.writeBuffer(unsafeAddr textLen, sizeof(uint32))
  storage.walFile.write(text)
  storage.walFile.flushFile()
  storage.walSize += uint64(sizeof(uint64) + sizeof(uint32) + text.len)

proc replayWal*(storage: MmappedStorage): seq[tuple[memoryId: uint64, text: string]] =
  let walPath = storage.dataDir / "wal.bin"
  if not fileExists(walPath) or getFileSize(walPath) == 0:
    return
  var f = system.open(walPath, fmRead)
  while true:
    var memoryId: uint64
    var textLen: uint32
    if f.readBuffer(addr memoryId, sizeof(uint64)) != sizeof(uint64):
      break
    if f.readBuffer(addr textLen, sizeof(uint32)) != sizeof(uint32):
      break
    var text = newString(int(textLen))
    if f.readBuffer(addr text[0], int(textLen)) != int(textLen):
      break
    result.add((memoryId, text))
  f.close()

proc getFingerprintPtr*(storage: MmappedStorage; memoryId: uint64): ptr Fingerprint =
  let offset = memoryId
  result = cast[ptr Fingerprint](cast[pointer](cast[uint](storage.fpMem) + uint(offset)))

proc writeFingerprint*(storage: MmappedStorage; memoryId: uint64; fp: Fingerprint) =
  let idx = memoryId div uint64(FingerprintBytes)
  if idx >= storage.fpCapacity:
    storage.growFpStore(idx + 1)
  let offset = memoryId
  copyMem(cast[pointer](cast[uint](storage.fpMem) + uint(offset)),
          unsafeAddr fp, sizeof(Fingerprint))

proc getHnswNodePtr*(storage: MmappedStorage; memoryId: uint64): ptr HnswNode =
  let idx = memoryId div uint64(FingerprintBytes)
  let offset = idx * uint64(sizeof(HnswNode))
  result = cast[ptr HnswNode](cast[pointer](cast[uint](storage.graphMem) + uint(offset)))

proc ensureGraphCapacity*(storage: MmappedStorage; memoryId: uint64) =
  let idx = memoryId div uint64(FingerprintBytes)
  if idx >= storage.graphCapacity:
    storage.growGraphStore(idx + 1)

proc writeHnswNode*(storage: MmappedStorage; memoryId: uint64; node: HnswNode) =
  storage.ensureGraphCapacity(memoryId)
  let idx = memoryId div uint64(FingerprintBytes)
  let offset = idx * uint64(sizeof(HnswNode))
  copyMem(cast[pointer](cast[uint](storage.graphMem) + uint(offset)),
          unsafeAddr node, sizeof(HnswNode))

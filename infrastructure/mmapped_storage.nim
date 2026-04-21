# Memory-mapped storage adapter.
# Manages WAL, fingerprint store, graph store, and chunk mapping
# store as contiguous memory-mapped flat files with 256-byte
# self-describing headers. Provides dense slot addressing,
# freelist-based ID reuse, and 64 MiB pre-allocated growth.


import ../domain/entities/fingerprint
import ../domain/entities/hnsw_node
import std/[memfiles, os]

const
  StoreHeaderSize = 256
  StoreMagic = "ADLN"
  StoreVersion = 1'u16
  GrowthChunkBytes = 64 * 1024 * 1024

type
  StoreHeader = object
    magic: array[4, char]
    version: uint16
    recordSize: uint16
    recordCount: uint64
    capacity: uint64
    freelistHead: uint64
    freelistCount: uint64

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
    chunksFile*: File
    chunksSize*: uint64
    recordCount*: uint64
    freelistHead*: uint64
    freelistCount*: uint64


proc writeHeader(f: File; h: StoreHeader) =
  f.setFilePos(0)
  discard f.writeBuffer(unsafeAddr h.magic[0], 4)
  discard f.writeBuffer(unsafeAddr h.version, 2)
  discard f.writeBuffer(unsafeAddr h.recordSize, 2)
  discard f.writeBuffer(unsafeAddr h.recordCount, 8)
  discard f.writeBuffer(unsafeAddr h.capacity, 8)
  discard f.writeBuffer(unsafeAddr h.freelistHead, 8)
  discard f.writeBuffer(unsafeAddr h.freelistCount, 8)
  f.flushFile()

proc readHeader(f: File): StoreHeader =
  f.setFilePos(0)
  discard f.readBuffer(addr result.magic[0], 4)
  discard f.readBuffer(addr result.version, 2)
  discard f.readBuffer(addr result.recordSize, 2)
  discard f.readBuffer(addr result.recordCount, 8)
  discard f.readBuffer(addr result.capacity, 8)
  discard f.readBuffer(addr result.freelistHead, 8)
  discard f.readBuffer(addr result.freelistCount, 8)

const FreelistNull = uint64.high

proc makeHeader(recordSize: uint16): StoreHeader =
  result.magic = [StoreMagic[0], StoreMagic[1], StoreMagic[2], StoreMagic[3]]
  result.version = StoreVersion
  result.recordSize = recordSize
  result.recordCount = 0
  result.capacity = 0
  result.freelistHead = FreelistNull
  result.freelistCount = 0

proc isNewFormat(path: string): bool =
  if not fileExists(path) or getFileSize(path) < StoreHeaderSize:
    return false
  var f = system.open(path, fmRead)
  defer: f.close()
  var magic: array[4, char]
  discard f.readBuffer(addr magic[0], 4)
  result = (magic == [StoreMagic[0], StoreMagic[1], StoreMagic[2], StoreMagic[3]])


proc extendFile(path: string; newSize: uint64) =
  var f = system.open(path, fmReadWriteExisting)
  f.setFilePos(int64(newSize - 1))
  f.write('\0')
  f.close()

proc preallocateFile(path: string; recordSize: uint64; minSlots: uint64): uint64 =
  let bytesPerChunk = uint64(GrowthChunkBytes)
  let slotsPerChunk = bytesPerChunk div recordSize
  if slotsPerChunk == 0:
    result = minSlots
  else:
    result = ((minSlots + slotsPerChunk - 1) div slotsPerChunk) * slotsPerChunk

  let totalBytes = uint64(StoreHeaderSize) + result * recordSize
  extendFile(path, totalBytes)


proc growGraphStore*(storage: MmappedStorage; minCount: uint64) =
  if minCount <= storage.graphCapacity:
    return
  let graphPath = storage.dataDir / "graph.bin"
  let newCapacity = preallocateFile(graphPath, uint64(sizeof(HnswNode)), minCount)

  storage.graphMemFile.close()
  let newSize = StoreHeaderSize + int(newCapacity * uint64(sizeof(HnswNode)))
  storage.graphMemFile = memfiles.open(graphPath, mode = fmReadWrite, mappedSize = newSize)
  storage.graphMem = cast[pointer](cast[uint](storage.graphMemFile.mem) + uint(StoreHeaderSize))
  storage.graphCapacity = newCapacity

  var f = system.open(graphPath, fmReadWriteExisting)
  var h = readHeader(f)
  h.capacity = newCapacity
  writeHeader(f, h)
  f.close()

proc growFpStore*(storage: MmappedStorage; minCount: uint64) =
  if minCount <= storage.fpCapacity:
    return
  let fpPath = storage.dataDir / "fingerprints.bin"
  let newCapacity = preallocateFile(fpPath, uint64(FingerprintBytes), minCount)

  storage.fpMemFile.close()
  let newSize = StoreHeaderSize + int(newCapacity * uint64(FingerprintBytes))
  storage.fpMemFile = memfiles.open(fpPath, mode = fmReadWrite, mappedSize = newSize)
  storage.fpMem = cast[pointer](cast[uint](storage.fpMemFile.mem) + uint(StoreHeaderSize))
  storage.fpCapacity = newCapacity

  var f = system.open(fpPath, fmReadWriteExisting)
  var h = readHeader(f)
  h.capacity = newCapacity
  writeHeader(f, h)
  f.close()

  if newCapacity > storage.graphCapacity:
    storage.growGraphStore(newCapacity)


proc initStorage*(dataDir: string): MmappedStorage =
  result = MmappedStorage(dataDir: dataDir)
  createDir(dataDir)

  let walPath = dataDir / "wal.bin"
  result.walFile = system.open(walPath, fmAppend)
  result.walSize = uint64(getFileSize(walPath))

  let fpPath = dataDir / "fingerprints.bin"
  if not fileExists(fpPath):
    var h = makeHeader(uint16(FingerprintBytes))
    var f = system.open(fpPath, fmReadWrite)
    writeHeader(f, h)
    f.close()
    let initialSlots = preallocateFile(fpPath, uint64(FingerprintBytes), 1024)
    var f2 = system.open(fpPath, fmReadWriteExisting)
    var h2 = readHeader(f2)
    h2.capacity = initialSlots
    writeHeader(f2, h2)
    f2.close()
  elif not isNewFormat(fpPath):
    raise newException(IOError, "fingerprints.bin uses old format without header. Please re-index.")

  let fpSize = uint64(getFileSize(fpPath))
  result.fpMemFile = memfiles.open(fpPath, mode = fmReadWrite, mappedSize = int(fpSize))
  result.fpMem = cast[pointer](cast[uint](result.fpMemFile.mem) + uint(StoreHeaderSize))
  result.fpCapacity = (fpSize - uint64(StoreHeaderSize)) div uint64(FingerprintBytes)

  let graphPath = dataDir / "graph.bin"
  if not fileExists(graphPath):
    var h = makeHeader(uint16(sizeof(HnswNode)))
    var f = system.open(graphPath, fmReadWrite)
    writeHeader(f, h)
    f.close()
    let initialSlots = preallocateFile(graphPath, uint64(sizeof(HnswNode)), 1024)
    var f2 = system.open(graphPath, fmReadWriteExisting)
    var h2 = readHeader(f2)
    h2.capacity = initialSlots
    writeHeader(f2, h2)
    f2.close()
  elif not isNewFormat(graphPath):
    raise newException(IOError, "graph.bin uses old format without header. Please re-index.")

  let graphSize = uint64(getFileSize(graphPath))
  result.graphMemFile = memfiles.open(graphPath, mode = fmReadWrite, mappedSize = int(graphSize))
  result.graphMem = cast[pointer](cast[uint](result.graphMemFile.mem) + uint(StoreHeaderSize))
  result.graphCapacity = (graphSize - uint64(StoreHeaderSize)) div uint64(sizeof(HnswNode))

  let chunksPath = dataDir / "chunks.bin"
  result.chunksFile = system.open(chunksPath, fmAppend)
  result.chunksSize = uint64(getFileSize(chunksPath))

  var fpHdr = readHeader(system.open(fpPath, fmRead))
  result.recordCount = fpHdr.recordCount
  result.freelistHead = fpHdr.freelistHead
  result.freelistCount = fpHdr.freelistCount


proc appendWal*(storage: MmappedStorage; memoryId: uint64; timestamp: uint64; text: string): uint64 =
  result = storage.walSize
  var textLen = uint32(text.len)
  discard storage.walFile.writeBuffer(unsafeAddr memoryId, sizeof(uint64))
  discard storage.walFile.writeBuffer(unsafeAddr timestamp, sizeof(uint64))
  discard storage.walFile.writeBuffer(unsafeAddr textLen, sizeof(uint32))
  storage.walFile.write(text)
  storage.walFile.flushFile()
  storage.walSize += uint64(sizeof(uint64) + sizeof(uint64) + sizeof(uint32) + text.len)

proc replayWal*(storage: MmappedStorage): seq[tuple[memoryId: uint64, timestamp: uint64, text: string]] =
  let walPath = storage.dataDir / "wal.bin"
  if not fileExists(walPath) or getFileSize(walPath) == 0:
    return
  var f = system.open(walPath, fmRead)
  while true:
    var memoryId: uint64
    var timestamp: uint64
    var textLen: uint32
    if f.readBuffer(addr memoryId, sizeof(uint64)) != sizeof(uint64):
      break
    if f.readBuffer(addr timestamp, sizeof(uint64)) != sizeof(uint64):
      break
    if f.readBuffer(addr textLen, sizeof(uint32)) != sizeof(uint32):
      break
    var text = newString(int(textLen))
    if f.readBuffer(addr text[0], int(textLen)) != int(textLen):
      break
    result.add((memoryId, timestamp, text))
  f.close()


proc appendChunkMapping*(storage: MmappedStorage; parentMemoryId: uint64; chunkId: uint64): uint64 =
  result = storage.chunksSize
  discard storage.chunksFile.writeBuffer(unsafeAddr parentMemoryId, sizeof(uint64))
  discard storage.chunksFile.writeBuffer(unsafeAddr chunkId, sizeof(uint64))
  storage.chunksFile.flushFile()
  storage.chunksSize += uint64(sizeof(uint64) + sizeof(uint64))

proc replayChunks*(storage: MmappedStorage): seq[tuple[parentMemoryId: uint64, chunkId: uint64]] =
  let chunksPath = storage.dataDir / "chunks.bin"
  if not fileExists(chunksPath) or getFileSize(chunksPath) == 0:
    return
  var f = system.open(chunksPath, fmRead)
  while true:
    var parentMemoryId: uint64
    var chunkId: uint64
    if f.readBuffer(addr parentMemoryId, sizeof(uint64)) != sizeof(uint64):
      break
    if f.readBuffer(addr chunkId, sizeof(uint64)) != sizeof(uint64):
      break
    result.add((parentMemoryId, chunkId))
  f.close()


proc getFingerprintPtr*(storage: MmappedStorage; memoryId: uint64): ptr Fingerprint {.inline.} =
  let offset = memoryId * uint64(FingerprintBytes)
  cast[ptr Fingerprint](cast[pointer](cast[uint](storage.fpMem) + uint(offset)))

proc writeFingerprint*(storage: MmappedStorage; memoryId: uint64; fp: Fingerprint) {.inline.} =
  if memoryId >= storage.fpCapacity:
    storage.growFpStore(memoryId + 1)
  let offset = memoryId * uint64(FingerprintBytes)
  copyMem(cast[pointer](cast[uint](storage.fpMem) + uint(offset)),
          unsafeAddr fp, sizeof(Fingerprint))

proc writeFingerprintUnsafe*(storage: MmappedStorage; memoryId: uint64; fp: Fingerprint) {.inline.} =
  let offset = memoryId * uint64(FingerprintBytes)
  copyMem(cast[pointer](cast[uint](storage.fpMem) + uint(offset)),
          unsafeAddr fp, sizeof(Fingerprint))

proc getHnswNodePtr*(storage: MmappedStorage; memoryId: uint64): ptr HnswNode {.inline.} =
  let offset = memoryId * uint64(sizeof(HnswNode))
  cast[ptr HnswNode](cast[pointer](cast[uint](storage.graphMem) + uint(offset)))

proc ensureGraphCapacity*(storage: MmappedStorage; memoryId: uint64) {.inline.} =
  if memoryId >= storage.graphCapacity:
    storage.growGraphStore(memoryId + 1)


proc syncHeader*(storage: MmappedStorage) =
  let fpPath = storage.dataDir / "fingerprints.bin"
  var f = system.open(fpPath, fmReadWriteExisting)
  var h = readHeader(f)
  h.recordCount = storage.recordCount
  h.freelistHead = storage.freelistHead
  h.freelistCount = storage.freelistCount
  writeHeader(f, h)
  f.close()

proc allocId*(storage: MmappedStorage): uint64 =
  if storage.freelistHead != FreelistNull:
    result = storage.freelistHead
    let nextPtr = cast[ptr uint64](cast[pointer](
      cast[uint](storage.fpMem) + uint(result * uint64(FingerprintBytes))))
    storage.freelistHead = nextPtr[]
    storage.freelistCount.dec
  else:
    result = storage.recordCount
    storage.recordCount.inc
    if result >= storage.fpCapacity:
      storage.growFpStore(result + 1)

proc freeId*(storage: MmappedStorage; id: uint64) =
  let fpPtr = storage.getFingerprintPtr(id)
  zeroMem(fpPtr, FingerprintBytes)
  let nextPtr = cast[ptr uint64](cast[pointer](
    cast[uint](storage.fpMem) + uint(id * uint64(FingerprintBytes))))
  nextPtr[] = storage.freelistHead
  storage.freelistHead = id
  storage.freelistCount.inc
  let node = storage.getHnswNodePtr(id)
  clearNode(node)

proc idCount*(storage: MmappedStorage): uint64 =
  result = storage.recordCount

proc freeIdCount*(storage: MmappedStorage): uint64 =
  result = storage.freelistCount

proc syncRecordCount*(storage: MmappedStorage; count: uint64) =
  if storage.recordCount < count:
    storage.recordCount = count
  if count > storage.graphCapacity:
    storage.growGraphStore(count)


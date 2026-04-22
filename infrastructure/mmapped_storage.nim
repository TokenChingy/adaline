## Memory-mapped storage adapter.
## Manages WAL, compressed fingerprint store, graph store, and chunk mapping
## store. Fingerprints are stored segment-sparse: only non-zero uint64
## segments are persisted, cutting disk usage by ~50-60% for typical SDR
## fingerprints. The graph store uses a runtime-configurable record size.
##
## File layout:
##   fingerprints.bin  – 256-byte header + append-only compressed data
##   fingerprints.idx  – 256-byte header + (offset: uint32, size: uint32) table
##   graph.bin         – 256-byte header + fixed-size HnswNode records
##   wal.bin           – append-only write-ahead log
##   chunks.bin        – append-only chunk lineage


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
    hnswMaxLayers: uint16
    hnswMaxNeighbors: uint16

  FpIdxEntry = object
    offset: uint32
    size: uint32

  MmappedStorage* = ref object
    dataDir*: string
    walFile*: File
    walSize*: uint64
    fpDataMemFile*: MemFile
    fpDataMem*: pointer
    fpDataSize*: uint64
    fpDataCapacity*: uint64
    fpIdxMemFile*: MemFile
    fpIdxMem*: pointer
    fpIdxCapacity*: uint64
    graphMemFile*: MemFile
    graphMem*: pointer
    graphCapacity*: uint64
    graphRecordSize*: uint64
    graphMaxLayers*: int
    graphMaxNeighbors*: int
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
  discard f.writeBuffer(unsafeAddr h.hnswMaxLayers, 2)
  discard f.writeBuffer(unsafeAddr h.hnswMaxNeighbors, 2)
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
  discard f.readBuffer(addr result.hnswMaxLayers, 2)
  discard f.readBuffer(addr result.hnswMaxNeighbors, 2)

const FreelistNull = uint64.high

proc makeHeader(recordSize: uint16; hnswMaxLayers: uint16 = 0; hnswMaxNeighbors: uint16 = 0): StoreHeader =
  result.magic = [StoreMagic[0], StoreMagic[1], StoreMagic[2], StoreMagic[3]]
  result.version = StoreVersion
  result.recordSize = recordSize
  result.recordCount = 0
  result.capacity = 0
  result.freelistHead = FreelistNull
  result.freelistCount = 0
  result.hnswMaxLayers = hnswMaxLayers
  result.hnswMaxNeighbors = hnswMaxNeighbors

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

proc preallocateBytes(path: string; minBytes: uint64): uint64 =
  let bytesPerChunk = uint64(GrowthChunkBytes)
  result = ((minBytes + bytesPerChunk - 1) div bytesPerChunk) * bytesPerChunk
  extendFile(path, result)


proc growGraphStore*(storage: MmappedStorage; minCount: uint64) =
  if minCount <= storage.graphCapacity:
    return
  let graphPath = storage.dataDir / "graph.bin"
  let newCapacity = preallocateFile(graphPath, storage.graphRecordSize, minCount)

  storage.graphMemFile.close()
  let newSize = StoreHeaderSize + int(newCapacity * storage.graphRecordSize)
  storage.graphMemFile = memfiles.open(graphPath, mode = fmReadWrite, mappedSize = newSize)
  storage.graphMem = cast[pointer](cast[uint](storage.graphMemFile.mem) + uint(StoreHeaderSize))
  storage.graphCapacity = newCapacity

  var f = system.open(graphPath, fmReadWriteExisting)
  var h = readHeader(f)
  h.capacity = newCapacity
  writeHeader(f, h)
  f.close()

proc growFpIdx*(storage: MmappedStorage; minCount: uint64) =
  if minCount <= storage.fpIdxCapacity:
    return
  let idxPath = storage.dataDir / "fingerprints.idx"
  let newCapacity = minCount
  let newSize = uint64(StoreHeaderSize) + newCapacity * uint64(sizeof(FpIdxEntry))
  extendFile(idxPath, newSize)

  storage.fpIdxMemFile.close()
  storage.fpIdxMemFile = memfiles.open(idxPath, mode = fmReadWrite, mappedSize = int(newSize))
  storage.fpIdxMem = cast[pointer](cast[uint](storage.fpIdxMemFile.mem) + uint(StoreHeaderSize))
  storage.fpIdxCapacity = newCapacity

  var f = system.open(idxPath, fmReadWriteExisting)
  var h = readHeader(f)
  h.capacity = newCapacity
  writeHeader(f, h)
  f.close()

proc growFpData*(storage: MmappedStorage; minBytes: uint64) =
  if minBytes <= storage.fpDataCapacity:
    return
  let dataPath = storage.dataDir / "fingerprints.bin"
  let newCapacity = preallocateBytes(dataPath, minBytes)

  storage.fpDataMemFile.close()
  storage.fpDataMemFile = memfiles.open(dataPath, mode = fmReadWrite, mappedSize = int(newCapacity))
  storage.fpDataMem = cast[pointer](cast[uint](storage.fpDataMemFile.mem) + uint(StoreHeaderSize))
  storage.fpDataCapacity = newCapacity

  var f = system.open(dataPath, fmReadWriteExisting)
  f.setFilePos(4)
  var ver: uint16
  discard f.readBuffer(addr ver, 2)
  f.setFilePos(6)
  discard f.writeBuffer(unsafeAddr newCapacity, 8)
  f.flushFile()
  f.close()


proc initStorage*(dataDir: string; hnswMaxLayers, hnswMaxNeighbors: int): MmappedStorage =
  result = MmappedStorage(dataDir: dataDir)
  createDir(dataDir)

  result.graphMaxLayers = hnswMaxLayers
  result.graphMaxNeighbors = hnswMaxNeighbors
  result.graphRecordSize = uint64(hnswNodeRecordSize(hnswMaxLayers, hnswMaxNeighbors))

  let walPath = dataDir / "wal.bin"
  result.walFile = system.open(walPath, fmAppend)
  result.walSize = uint64(getFileSize(walPath))

  let idxPath = dataDir / "fingerprints.idx"
  let dataPath = dataDir / "fingerprints.bin"

  # Detect old format (fingerprints.bin exists but fingerprints.idx does not)
  if fileExists(dataPath) and not fileExists(idxPath):
    raise newException(IOError,
      "fingerprints.bin exists without fingerprints.idx. Old dense slot format. Please re-index.")

  if not fileExists(idxPath):
    var h = makeHeader(uint16(sizeof(FpIdxEntry)))
    h.capacity = 1024
    var f = system.open(idxPath, fmReadWrite)
    writeHeader(f, h)
    f.close()
    let idxBytes = uint64(StoreHeaderSize) + 1024'u64 * uint64(sizeof(FpIdxEntry))
    extendFile(idxPath, idxBytes)

  let idxSize = uint64(getFileSize(idxPath))
  result.fpIdxMemFile = memfiles.open(idxPath, mode = fmReadWrite, mappedSize = int(idxSize))
  result.fpIdxMem = cast[pointer](cast[uint](result.fpIdxMemFile.mem) + uint(StoreHeaderSize))
  result.fpIdxCapacity = (idxSize - uint64(StoreHeaderSize)) div uint64(sizeof(FpIdxEntry))

  if not fileExists(dataPath):
    var f = system.open(dataPath, fmReadWrite)
    var magic = [StoreMagic[0], StoreMagic[1], StoreMagic[2], StoreMagic[3]]
    discard f.writeBuffer(unsafeAddr magic[0], 4)
    var ver = StoreVersion
    discard f.writeBuffer(unsafeAddr ver, 2)
    var dataCap = uint64(GrowthChunkBytes)
    discard f.writeBuffer(unsafeAddr dataCap, 8)
    f.flushFile()
    f.close()
    extendFile(dataPath, dataCap)

  let dataSize = uint64(getFileSize(dataPath))
  result.fpDataMemFile = memfiles.open(dataPath, mode = fmReadWrite, mappedSize = int(dataSize))
  result.fpDataMem = cast[pointer](cast[uint](result.fpDataMemFile.mem) + uint(StoreHeaderSize))
  result.fpDataCapacity = dataSize - uint64(StoreHeaderSize)

  var fHdr = system.open(dataPath, fmRead)
  fHdr.setFilePos(6)
  var storedCap: uint64
  discard fHdr.readBuffer(addr storedCap, 8)
  fHdr.close()
  result.fpDataCapacity = storedCap - uint64(StoreHeaderSize)

  let graphPath = dataDir / "graph.bin"
  if not fileExists(graphPath):
    let gRecordSize = uint16(result.graphRecordSize)
    var h = makeHeader(gRecordSize, uint16(hnswMaxLayers), uint16(hnswMaxNeighbors))
    var f = system.open(graphPath, fmReadWrite)
    writeHeader(f, h)
    f.close()
    let initialSlots = preallocateFile(graphPath, result.graphRecordSize, 1024)
    var f2 = system.open(graphPath, fmReadWriteExisting)
    var h2 = readHeader(f2)
    h2.capacity = initialSlots
    writeHeader(f2, h2)
    f2.close()
  elif not isNewFormat(graphPath):
    raise newException(IOError, "graph.bin uses old format without header. Please re-index.")
  else:
    var f = system.open(graphPath, fmRead)
    var h = readHeader(f)
    f.close()
    if int(h.hnswMaxLayers) != hnswMaxLayers or int(h.hnswMaxNeighbors) != hnswMaxNeighbors:
      raise newException(IOError,
        "graph.bin was created with hnswMaxLayers=" & $h.hnswMaxLayers &
        ", hnswMaxNeighbors=" & $h.hnswMaxNeighbors &
        " but config requests " & $hnswMaxLayers & ", " & $hnswMaxNeighbors &
        ". Please re-index.")
    result.graphRecordSize = uint64(h.recordSize)

  let graphSize = uint64(getFileSize(graphPath))
  result.graphMemFile = memfiles.open(graphPath, mode = fmReadWrite, mappedSize = int(graphSize))
  result.graphMem = cast[pointer](cast[uint](result.graphMemFile.mem) + uint(StoreHeaderSize))
  result.graphCapacity = (graphSize - uint64(StoreHeaderSize)) div result.graphRecordSize

  let chunksPath = dataDir / "chunks.bin"
  result.chunksFile = system.open(chunksPath, fmAppend)
  result.chunksSize = uint64(getFileSize(chunksPath))

  var idxHdr = readHeader(system.open(idxPath, fmRead))
  result.recordCount = idxHdr.recordCount
  result.freelistHead = idxHdr.freelistHead
  result.freelistCount = idxHdr.freelistCount


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


# ---------------------------------------------------------------------------
# Fingerprint storage: compressed variable-length records.
# ---------------------------------------------------------------------------

proc readFingerprint*(storage: MmappedStorage; memoryId: uint64; outFp: ptr Fingerprint) {.inline.} =
  let entry = cast[ptr UncheckedArray[FpIdxEntry]](storage.fpIdxMem)[memoryId]
  if entry.size == 0:
    zeroMem(outFp, sizeof(Fingerprint))
  else:
    let src = cast[pointer](cast[uint](storage.fpDataMem) + uint(entry.offset))
    decompressFingerprint(src, outFp)

proc readFingerprintWrapper*(ctx: pointer; memoryId: uint64; outFp: ptr Fingerprint) {.nimcall.} =
  let storage = cast[ptr MmappedStorage](ctx)
  storage[].readFingerprint(memoryId, outFp)

proc writeFingerprint*(storage: MmappedStorage; memoryId: uint64; fp: Fingerprint) {.inline.} =
  if memoryId >= storage.fpIdxCapacity:
    storage.growFpIdx(memoryId + 1)

  let offset = storage.fpDataSize
  let dst = cast[pointer](cast[uint](storage.fpDataMem) + uint(offset))
  let compressedSize = uint64(compressFingerprint(fp, dst))

  if offset + compressedSize > storage.fpDataCapacity:
    storage.growFpData(offset + compressedSize)
    # Recalculate dst after remap
    let dst2 = cast[pointer](cast[uint](storage.fpDataMem) + uint(offset))
    discard compressFingerprint(fp, dst2)

  storage.fpDataSize += compressedSize

  cast[ptr UncheckedArray[FpIdxEntry]](storage.fpIdxMem)[memoryId].offset = uint32(offset)
  cast[ptr UncheckedArray[FpIdxEntry]](storage.fpIdxMem)[memoryId].size = uint32(compressedSize)

proc writeFingerprintUnsafe*(storage: MmappedStorage; memoryId: uint64; fp: Fingerprint) {.inline.} =
  storage.writeFingerprint(memoryId, fp)

proc getHnswNodeView*(storage: MmappedStorage; memoryId: uint64): HnswNodeView {.inline.} =
  let offset = memoryId * storage.graphRecordSize
  makeView(cast[pointer](cast[uint](storage.graphMem) + uint(offset)),
           storage.graphMaxNeighbors, storage.graphMaxLayers)

proc ensureGraphCapacity*(storage: MmappedStorage; memoryId: uint64) {.inline.} =
  if memoryId >= storage.graphCapacity:
    storage.growGraphStore(memoryId + 1)


proc syncHeader*(storage: MmappedStorage) =
  let idxPath = storage.dataDir / "fingerprints.idx"
  var f = system.open(idxPath, fmReadWriteExisting)
  var h = readHeader(f)
  h.recordCount = storage.recordCount
  h.freelistHead = storage.freelistHead
  h.freelistCount = storage.freelistCount
  writeHeader(f, h)
  f.close()

proc allocId*(storage: MmappedStorage): uint64 =
  if storage.freelistHead != FreelistNull:
    result = storage.freelistHead
    let nextId = uint64(cast[ptr UncheckedArray[FpIdxEntry]](storage.fpIdxMem)[result].offset)
    storage.freelistHead = if nextId == uint64(uint32.high): FreelistNull else: nextId
    storage.freelistCount.dec
    # Reset the entry so it is ready for a new writeFingerprint call
    cast[ptr UncheckedArray[FpIdxEntry]](storage.fpIdxMem)[result].offset = 0
    cast[ptr UncheckedArray[FpIdxEntry]](storage.fpIdxMem)[result].size = 0
  else:
    result = storage.recordCount
    storage.recordCount.inc
    if result >= storage.fpIdxCapacity:
      storage.growFpIdx(result + 1)

proc freeId*(storage: MmappedStorage; id: uint64) =
  let idxArr = cast[ptr UncheckedArray[FpIdxEntry]](storage.fpIdxMem)
  idxArr[id].offset = uint32(storage.freelistHead)
  idxArr[id].size = 0
  storage.freelistHead = id
  storage.freelistCount.inc
  let node = storage.getHnswNodeView(id)
  clearNode(node)

proc idCount*(storage: MmappedStorage): uint64 =
  result = storage.recordCount

proc freeIdCount*(storage: MmappedStorage): uint64 =
  result = storage.freelistCount

proc syncRecordCount*(storage: MmappedStorage; count: uint64) =
  if storage.recordCount < count:
    storage.recordCount = count
  if count > storage.fpIdxCapacity:
    storage.growFpIdx(count)
  if count > storage.graphCapacity:
    storage.growGraphStore(count)

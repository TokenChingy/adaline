import nimpy
import std/tables
import ../domain/services/memory_service
import ../domain/entities/config
import ../domain/entities/memory
import ../use_cases/insert_memory
import ../use_cases/search_memories
import ../use_cases/update_memory
import ../use_cases/delete_memory

type
  Engine* = ref object of PyNimObjectExperimental
    service*: MemoryService

proc newEngine*(dataDir: string): Engine {.exportpy.} =
  let cfg = defaultEngineConfig()
  result = Engine(service: initMemoryService(dataDir, cfg))

proc insert*(self: Engine; content: string): uint64 {.exportpy.} =
  let output = insertMemory(self.service, InsertMemoryInput(content: content))
  result = output.memoryId

proc search*(self: Engine; query: string; topK: int = 10): seq[Memory] {.exportpy.} =
  let output = searchMemories(self.service, SearchMemoriesInput(query: query, topK: topK))
  result = output.memories

proc update*(self: Engine; memoryId: uint64; content: string): uint64 {.exportpy.} =
  let output = updateMemory(self.service, UpdateMemoryInput(memoryId: memoryId, content: content))
  result = output.memoryId

proc delete*(self: Engine; memoryId: uint64) {.exportpy.} =
  discard deleteMemory(self.service, DeleteMemoryInput(memoryId: memoryId))

proc checkpoint*(self: Engine) {.exportpy.} =
  checkpoint(self.service)

type
  StatsResult* = object
    memories*: int
    corpusMemories*: int
    hnswLayers*: int
    hnswEntryPoint*: uint64
    fingerprintBits*: int
    lshBands*: int
    lshRows*: int
    hnswMaxLayers*: int
    hnswEfSearch*: int
    rrfK*: int
    dirichletMu*: float

proc stats*(self: Engine): StatsResult {.exportpy.} =
  let cfg = self.service.cfg
  result = StatsResult(
    memories: self.service.textCache.len,
    corpusMemories: self.service.corpus.numMemories,
    hnswLayers: self.service.maxHnswLayer + 1,
    hnswEntryPoint: self.service.hnswEntryPoint,
    fingerprintBits: cfg.fingerprintBits,
    lshBands: cfg.lshBands,
    lshRows: cfg.lshRows,
    hnswMaxLayers: cfg.hnswMaxLayers,
    hnswEfSearch: cfg.hnswEfSearch,
    rrfK: cfg.rrfK,
    dirichletMu: cfg.dirichletMu
  )

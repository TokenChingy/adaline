import ../domain/services/memory_service

type
  InsertMemoryInput* = object
    content*: string

  InsertMemoryOutput* = object
    memoryId*: uint64

proc insertMemory*(service: var MemoryService; input: InsertMemoryInput): InsertMemoryOutput =
  result.memoryId = service.insert(input.content)

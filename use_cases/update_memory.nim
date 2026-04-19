import ../domain/services/memory_service

type
  UpdateMemoryInput* = object
    memoryId*: uint64
    content*: string

  UpdateMemoryOutput* = object
    memoryId*: uint64

proc updateMemory*(service: var MemoryService; input: UpdateMemoryInput): UpdateMemoryOutput =
  service.updateMemory(input.memoryId, input.content)
  result.memoryId = input.memoryId

import ../domain/services/memory/delete

type
  DeleteMemoryInput* = object
    memoryId*: uint64

  DeleteMemoryOutput* = object
    memoryId*: uint64

proc deleteMemory*(service: var MemoryService; input: DeleteMemoryInput): DeleteMemoryOutput =
  service.deleteMemory(input.memoryId)
  result.memoryId = input.memoryId

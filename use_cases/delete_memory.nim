## Delete memory use case.
## Input/output port for deleting a memory by ID.
## Delegates to the domain delete service.


import ../domain/services/memory/delete

type
  DeleteMemoryInput* = object
    memoryId*: uint64

  DeleteMemoryOutput* = object
    memoryId*: uint64

proc deleteMemory*(service: var MemoryService; input: DeleteMemoryInput): DeleteMemoryOutput =
  service.deleteMemory(input.memoryId)
  result.memoryId = input.memoryId

# Update memory use case.
# Input/output port for updating an existing memory by ID.
# Delegates to the domain update service.


import ../domain/services/memory/update

type
  UpdateMemoryInput* = object
    memoryId*: uint64
    content*: string

  UpdateMemoryOutput* = object
    memoryId*: uint64

proc updateMemory*(service: var MemoryService; input: UpdateMemoryInput): UpdateMemoryOutput =
  service.updateMemory(input.memoryId, input.content)
  result.memoryId = input.memoryId

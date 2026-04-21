## Delete dense-vector use case.
## Input/output port for deleting a dense-only vector by ID.
## Delegates to the domain delete_dense service.


import ../domain/services/memory/delete_dense

type
  DeleteDenseInput* = object
    memoryId*: uint64

proc deleteDense*(service: var MemoryService; input: DeleteDenseInput) =
  service.deleteDense(input.memoryId)

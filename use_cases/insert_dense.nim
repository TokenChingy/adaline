# Insert dense-vector use case.
# Input/output port for inserting a dense float32 vector.
# Delegates to the domain insert_dense service.


import ../domain/services/memory/insert_dense

type
  InsertDenseInput* = object
    vec*: seq[float32]

  InsertDenseOutput* = object
    memoryId*: uint64

proc insertDense*(service: var MemoryService; input: InsertDenseInput): InsertDenseOutput =
  result.memoryId = service.insertDense(input.vec)

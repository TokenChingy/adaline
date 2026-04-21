# Search dense-vector use case.
# Input/output port for searching by dense float32 vector.
# Delegates to the domain search_dense service.


import ../domain/services/memory/search_dense

type
  SearchDenseInput* = object
    vec*: seq[float32]
    topK*: int

  SearchDenseOutput* = object
    results*: seq[tuple[memoryId: uint64, score: float]]

proc searchDense*(service: var MemoryService; input: SearchDenseInput): SearchDenseOutput =
  result.results = service.searchDense(input.vec, input.topK)

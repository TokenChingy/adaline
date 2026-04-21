# Search memories use case.
# Input/output port for searching memories by query text.
# Delegates to the domain search service.


import ../domain/services/memory/search
import ../domain/entities/memory

type
  SearchMemoriesInput* = object
    query*: string
    topK*: int

  SearchMemoriesOutput* = object
    memories*: seq[Memory]

proc searchMemories*(service: var MemoryService; input: SearchMemoriesInput): SearchMemoriesOutput =
  result.memories = service.search(input.query, input.topK)

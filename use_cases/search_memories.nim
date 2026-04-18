import ../domain/services/memory_service
import ../domain/entities/memory

type
  SearchMemoriesInput* = object
    query*: string
    topK*: int

  SearchMemoriesOutput* = object
    memories*: seq[Memory]

proc searchMemories*(service: var MemoryService; input: SearchMemoriesInput): SearchMemoriesOutput =
  result.memories = service.search(input.query, input.topK)

import domain/services/memory_service
import domain/entities/config
import use_cases/insert_memory
import use_cases/search_memories
import std/os

proc main() =
  let dataDir = getCurrentDir() / "data"
  let cfg = defaultEngineConfig()
  var service = initMemoryService(dataDir, cfg)

  let m1 = insertMemory(service, InsertMemoryInput(content: "the quick brown fox"))
  let m2 = insertMemory(service, InsertMemoryInput(content: "lazy dog sleeping in the sun"))
  let m3 = insertMemory(service, InsertMemoryInput(content: "nim programming language"))
  let m4 = insertMemory(service, InsertMemoryInput(content: "foxes are quick and brown"))
  echo "Inserted memories: ", m1.memoryId, ", ", m2.memoryId, ", ", m3.memoryId, ", ", m4.memoryId

  let results = searchMemories(service, SearchMemoriesInput(query: "quick fox", topK: 2))
  echo "Search results for 'quick fox':"
  for mem in results.memories:
    echo "  - [", mem.id, "] ", mem.content

when isMainModule:
  main()

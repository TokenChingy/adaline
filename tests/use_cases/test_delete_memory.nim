# Unit tests for Delete Memory use case.


import ../../use_cases/delete_memory
import ../../domain/services/memory/init
import ../../domain/services/memory/insert
import ../../domain/entities/config
import std/[os, tables]

proc main() =
  let dataDir = "tests/temp_delete_memory"
  removeDir(dataDir)
  createDir(dataDir)

  let cfg = defaultEngineConfig()
  var service = initMemoryService(dataDir, cfg)

  let id = service.insert("hello world")
  assert service.textCache.hasKey(id), "Memory should exist after insert"

  let output = deleteMemory(service, DeleteMemoryInput(memoryId: id))
  assert output.memoryId == id, "Output ID should match input"
  assert not service.textCache.hasKey(id), "Memory should be deleted"

  let output2 = deleteMemory(service, DeleteMemoryInput(memoryId: 999'u64))
  assert output2.memoryId == 999'u64

  removeDir(dataDir)
  echo "test_delete_memory OK"

main()

## Dense-vector delete service.
## Removes a dense-only vector from LSH and frees the slot.
## Guards against deleting text memories or chunks.


import ./types
import ../../entities/fingerprint
import ../../algorithms/fingerprint_lsh
import ../../../infrastructure/mmapped_storage

export types

proc deleteDense*(service: var MemoryService; memoryId: uint64) =
  # Guard: don't delete text memories or chunks via dense path
  if service.textCache.hasKey(memoryId) or service.chunkToParent.hasKey(memoryId):
    return

  var fp: Fingerprint
  service.storage.readFingerprint(memoryId, addr fp)
  removeLsh(service.lsh, addr fp, memoryId)
  service.storage.freeId(memoryId)
  service.storage.syncHeader()

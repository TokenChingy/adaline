## Delete memory service.
## Physically removes a memory and all its chunks from every index.


import ./types
import ../../entities/fingerprint
import ../../algorithms/fingerprint_lsh
import ../../algorithms/lexical_index
import ../../../infrastructure/mmapped_storage
import std/tables

export types

proc deleteMemory*(service: var MemoryService; parentId: uint64) =
  if not service.textCache.hasKey(parentId):
    return

  let content = service.textCache[parentId]

  var chunkIds = newSeq[uint64]()
  for chunkId, pid in service.chunkToParent:
    if pid == parentId:
      chunkIds.add(chunkId)

  if chunkIds.len == 0:
    chunkIds.add(parentId)

  for chunkId in chunkIds:
    var fp: Fingerprint
    service.storage.readFingerprint(chunkId, addr fp)
    removeLsh(service.lsh, addr fp, chunkId)

    let chunkText = if chunkId == parentId: content
                    else: service.textCache.getOrDefault(service.chunkToParent.getOrDefault(chunkId, chunkId), "")
    removeMemory(service.lexical, chunkId, chunkText)

    service.storage.freeId(chunkId)

    service.chunkToParent.del(chunkId)

  service.textCache.del(parentId)
  service.tokenCache.del(parentId)
  service.timestampCache.del(parentId)

  service.storage.syncHeader()

# Dense-vector delete service.
# Removes a dense-only vector from LSH and HNSW, heals graph edges,
# and frees the slot.  Guards against deleting text memories or chunks.


import ./types
import ../../entities/hnsw_node
import ../../algorithms/fingerprint_lsh
import ../../../infrastructure/mmapped_storage
import std/[tables, sequtils]

export types

proc deleteDense*(service: var MemoryService; memoryId: uint64) =
  # Guard: don't delete text memories or chunks via dense path
  if service.textCache.hasKey(memoryId) or service.chunkToParent.hasKey(memoryId):
    return

  let node = service.storage.getHnswNodePtr(memoryId)
  if node.layerCount > 0:
    # Remove this node from all predecessors' neighbor lists
    for lc in 0 ..< int(node.layerCount):
      for nid in node.neighbors(lc):
        if service.hnswReverseIndex.hasKey(nid):
          service.hnswReverseIndex[nid] = service.hnswReverseIndex[nid].filterIt(it != memoryId)

    # Remove all predecessors from this node's neighbor lists
    if service.hnswReverseIndex.hasKey(memoryId):
      for mid in service.hnswReverseIndex[memoryId]:
        let mnode = service.storage.getHnswNodePtr(mid)
        if mnode.layerCount > 0:
          for lc in 0 ..< int(mnode.layerCount):
            discard removeNeighbor(mnode, lc, memoryId)
      service.hnswReverseIndex.del(memoryId)

    # Heal entry point if needed
    if service.hnswEntryPoint == memoryId:
      var newEp: uint64 = 0
      var newMaxLayer = -1
      for i in 1'u64 ..< service.storage.idCount():
        let n = service.storage.getHnswNodePtr(i)
        if n.layerCount > 0 and int(n.entryLayer) > newMaxLayer:
          newMaxLayer = int(n.entryLayer)
          newEp = i
      service.hnswEntryPoint = newEp
      service.maxHnswLayer = newMaxLayer

  let fpPtr = service.storage.getFingerprintPtr(memoryId)
  removeLsh(service.lsh, fpPtr, memoryId)
  service.storage.freeId(memoryId)
  service.storage.syncHeader()

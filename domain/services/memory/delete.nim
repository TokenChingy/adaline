## Delete memory service.
## Physically removes a memory and all its chunks from every index.
## Heals HNSW neighbor lists via an in-memory reverse edge index
## without tombstones or full graph rebuilds.


import ./types
import ../../entities/hnsw_node
import ../../algorithms/fingerprint_lsh
import ../../algorithms/lexical_index
import ../../../infrastructure/mmapped_storage
import std/[tables, sequtils]

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
    let node = service.storage.getHnswNodePtr(chunkId)
    if node.layerCount > 0:
      for lc in 0 ..< int(node.layerCount):
        for nid in node.neighbors(lc):
          if service.hnswReverseIndex.hasKey(nid):
            service.hnswReverseIndex[nid] = service.hnswReverseIndex[nid].filterIt(it != chunkId)

      if service.hnswReverseIndex.hasKey(chunkId):
        for mid in service.hnswReverseIndex[chunkId]:
          let mnode = service.storage.getHnswNodePtr(mid)
          if mnode.layerCount > 0:
            for lc in 0 ..< int(mnode.layerCount):
              discard removeNeighbor(mnode, lc, chunkId)
        service.hnswReverseIndex.del(chunkId)

      if service.hnswEntryPoint == chunkId:
        var newEp: uint64 = 0
        var newMaxLayer = -1
        for i in 1'u64 ..< service.storage.idCount():
          let n = service.storage.getHnswNodePtr(i)
          if n.layerCount > 0 and int(n.entryLayer) > newMaxLayer:
            newMaxLayer = int(n.entryLayer)
            newEp = i
        service.hnswEntryPoint = newEp
        service.maxHnswLayer = newMaxLayer

    let fpPtr = service.storage.getFingerprintPtr(chunkId)
    removeLsh(service.lsh, fpPtr, chunkId)

    let chunkText = if chunkId == parentId: content
                    else: service.textCache.getOrDefault(service.chunkToParent.getOrDefault(chunkId, chunkId), "")
    removeMemory(service.lexical, chunkId, chunkText)

    service.storage.freeId(chunkId)

    service.chunkToParent.del(chunkId)

  service.textCache.del(parentId)
  service.timestampCache.del(parentId)

  service.storage.syncHeader()

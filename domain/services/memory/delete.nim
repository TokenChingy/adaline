import ./types
import ../../entities/hnsw_node
import ../../algorithms/fingerprint_lsh
import ../../algorithms/lexical_index
import ../../../infrastructure/mmapped_storage
import std/[tables, sequtils]

export types

proc deleteMemory*(service: var MemoryService; parentId: uint64) =
  ## Delete a memory and all its chunks. Frees slots for reuse.
  ## Heals HNSW edges via reverse index so no orphaned references remain.
  if not service.textCache.hasKey(parentId):
    return

  let content = service.textCache[parentId]

  # Find all chunk IDs for this parent
  var chunkIds = newSeq[uint64]()
  for chunkId, pid in service.chunkToParent:
    if pid == parentId:
      chunkIds.add(chunkId)

  # If unchunked, the parent is its own chunk
  if chunkIds.len == 0:
    chunkIds.add(parentId)

  for chunkId in chunkIds:
    # Heal HNSW graph: remove chunkId from all neighbor lists that reference it
    let node = service.storage.getHnswNodePtr(chunkId)
    if node.layerCount > 0:
      # Forward edges: for each neighbor N of chunkId, remove chunkId from reverse[N]
      for lc in 0 ..< int(node.layerCount):
        for nid in node.neighbors(lc):
          if service.hnswReverseIndex.hasKey(nid):
            service.hnswReverseIndex[nid] = service.hnswReverseIndex[nid].filterIt(it != chunkId)

      # Backward edges: for each node M that points to chunkId, remove chunkId from M's list
      if service.hnswReverseIndex.hasKey(chunkId):
        for mid in service.hnswReverseIndex[chunkId]:
          let mnode = service.storage.getHnswNodePtr(mid)
          if mnode.layerCount > 0:
            for lc in 0 ..< int(mnode.layerCount):
              discard removeNeighbor(mnode, lc, chunkId)
        service.hnswReverseIndex.del(chunkId)

    # Remove from LSH (needs fingerprint before freeing slot)
    let fpPtr = service.storage.getFingerprintPtr(chunkId)
    removeLsh(service.lsh, fpPtr, chunkId)

    # Remove from lexical index
    let chunkText = if chunkId == parentId: content
                    else: service.textCache.getOrDefault(service.chunkToParent.getOrDefault(chunkId, chunkId), "")
    removeMemory(service.lexical, chunkId, chunkText)

    # Free the storage slot
    service.storage.freeSlot(chunkId)

    # Remove chunk→parent mapping
    service.chunkToParent.del(chunkId)

  # Remove parent from caches
  service.textCache.del(parentId)
  service.timestampCache.del(parentId)

  # Sync header to disk after delete
  service.storage.syncHeader()

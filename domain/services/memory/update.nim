# Update memory service.
# Performs a logical atomic delete-then-insert: heals the old
# chunks out of the graph, then re-inserts updated content and
# maps fresh chunks back to the original parent ID.


import ./types
import ./delete
import ../../algorithms/sdr_encoder
import ../../algorithms/corpus_index
import ../../algorithms/fingerprint_lsh
import ../../algorithms/hnsw_graph
import ../../algorithms/lexical_index
import ../../algorithms/chunker
import ../../../infrastructure/mmapped_storage
import std/[tables, times]

export types

proc updateMemory*(service: var MemoryService; parentId: uint64; content: string) =
  if not service.textCache.hasKey(parentId):
    return

  deleteMemory(service, parentId)

  let timestamp = uint64(getTime().toUnix())
  var storage = service.storage

  discard storage.appendWal(parentId, timestamp, content)
  service.textCache[parentId] = content
  service.timestampCache[parentId] = timestamp

  service.corpus.addMemory(content)

  let chunks = splitIntoChunks(content, service.cfg)

  if chunks.len == 1:
    let chunkId = parentId
    service.chunkToParent[chunkId] = parentId
    discard storage.appendChunkMapping(parentId, chunkId)

    var fp = encodeSdr(content, service.cfg, service.corpus)
    storage.writeFingerprintUnsafe(chunkId, fp)
    insertLsh(service.lsh, addr fp, chunkId)
    addMemory(service.lexical, chunkId, content)
    insertHnsw(storage.graphMem, storage.fpMem, chunkId, addr fp,
               service.cfg, service.maxHnswLayer, service.hnswEntryPoint,
               service.hnswReverseIndex)
  else:
    for chunkText in chunks:
      let chunkId = storage.allocId()
      service.chunkToParent[chunkId] = parentId
      discard storage.appendChunkMapping(parentId, chunkId)

      var fp = encodeSdr(chunkText, service.cfg, service.corpus)
      storage.writeFingerprintUnsafe(chunkId, fp)
      insertLsh(service.lsh, addr fp, chunkId)
      addMemory(service.lexical, chunkId, chunkText)
      insertHnsw(storage.graphMem, storage.fpMem, chunkId, addr fp,
                 service.cfg, service.maxHnswLayer, service.hnswEntryPoint,
                 service.hnswReverseIndex)

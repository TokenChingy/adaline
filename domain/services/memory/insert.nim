## Insert memory service.
## Orchestrates the full insert pipeline: WAL append, chunking,
## SDR encoding, LSH indexing, lexical indexing, and HNSW graph
## insertion with layer assignment and neighbor wiring.


import ./types
import ../../algorithms/sdr_encoder
import ../../algorithms/corpus_index
import ../../algorithms/fingerprint_lsh
import ../../algorithms/hnsw_graph
import ../../algorithms/lexical_index
import ../../algorithms/chunker
import ../../../infrastructure/mmapped_storage
import std/[tables, times]

export types

proc insert*(service: var MemoryService; content: string): uint64 =
  var storage = service.storage
  let parentId = storage.allocId()
  let timestamp = uint64(getTime().toUnix())

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

  result = parentId

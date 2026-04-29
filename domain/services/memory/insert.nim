## Insert memory service.
## Orchestrates the full insert pipeline: WAL append, chunking,
## SDR encoding, LSH indexing, and lexical indexing.


import ./types
import ../../algorithms/sdr_encoder
import ../../algorithms/corpus_index
import ../../algorithms/fingerprint_lsh
import ../../algorithms/lexical_index
import ../../algorithms/chunker
import ../../../infrastructure/mmapped_storage
import std/[tables, sets, times, strutils]

export types

proc insert*(service: var MemoryService; content: string): uint64 =
  var storage = service.storage
  let parentId = storage.allocId()
  let timestamp = uint64(getTime().toUnix())

  discard storage.appendWal(parentId, timestamp, content)
  service.textCache[parentId] = content
  service.lowerTextCache[parentId] = content.toLowerAscii()
  service.tokenCache[parentId] = initHashSet[string]()
  for token in content.toLowerAscii().split(AllChars - Letters - Digits):
    if token.len > 0:
      service.tokenCache[parentId].incl(token)
  service.timestampCache[parentId] = timestamp

  service.corpus.addMemory(content)

  let chunks = splitIntoChunks(content, service.cfg)

  if chunks.len == 1:
    let chunkId = parentId
    service.chunkToParent[chunkId] = parentId
    service.parentToChunks[parentId] = @[chunkId]
    discard storage.appendChunkMapping(parentId, chunkId)

    var fp = encodeSdr(content, service.cfg, service.corpus)
    storage.writeFingerprintUnsafe(chunkId, fp)

    insertLsh(service.lsh, addr fp, chunkId)

    addMemory(service.lexical, chunkId, content)
  else:
    var chunkIds = newSeq[uint64]()
    for chunkText in chunks:
      let chunkId = storage.allocId()
      service.chunkToParent[chunkId] = parentId
      chunkIds.add(chunkId)
      discard storage.appendChunkMapping(parentId, chunkId)

      var fp = encodeSdr(chunkText, service.cfg, service.corpus)
      storage.writeFingerprintUnsafe(chunkId, fp)

      insertLsh(service.lsh, addr fp, chunkId)

      addMemory(service.lexical, chunkId, chunkText)
    service.parentToChunks[parentId] = chunkIds

  result = parentId

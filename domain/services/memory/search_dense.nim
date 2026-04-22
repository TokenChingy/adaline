## Dense-vector search service.
## Encodes a query vector to a fingerprint and searches LSH + HNSW.


import ./types
import ../../algorithms/dense_encoder
import ../../algorithms/fingerprint_lsh
import ../../algorithms/hnsw_graph
import ../../../infrastructure/mmapped_storage

export types

proc searchDense*(service: var MemoryService; vec: seq[float32]; topK: int): seq[tuple[memoryId: uint64, score: float]] =
  let qfp = encodeDense(vec, service.cfg)
  let seeds = queryLsh(service.lsh, addr qfp)
  if service.hnswEntryPoint != 0 or seeds.len > 0:
    result = searchHnsw(service.storage.graphMem, readFingerprintWrapper,
                        addr service.storage, seeds, service.hnswEntryPoint,
                        addr qfp, topK, service.cfg, service.storage.graphRecordSize)

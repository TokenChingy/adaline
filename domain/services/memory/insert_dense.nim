## Dense-vector insert service.
## Bypasses text chunking, SDR encoding, and lexical indexing.
## Encodes a float32 vector directly to a fingerprint and inserts into LSH.


import ./types
import ../../algorithms/dense_encoder
import ../../algorithms/fingerprint_lsh
import ../../../infrastructure/mmapped_storage

export types

proc insertDense*(service: var MemoryService; vec: seq[float32]): uint64 =
  let id = service.storage.allocId()
  let fp = encodeDense(vec, service.cfg)
  service.storage.writeFingerprintUnsafe(id, fp)
  insertLsh(service.lsh, addr fp, id)
  result = id

## Memory service types.
## Defines the MemoryService object that holds references to all
## memory-mapped stores and in-memory indexes.


import ../../entities/config
import ../../algorithms/fingerprint_lsh
import ../../algorithms/lexical_index
import ../../algorithms/corpus_index
import ../../../infrastructure/mmapped_storage
import std/[tables, sets]

type
  MemoryService* = object
    storage*: MmappedStorage
    cfg*: EngineConfig
    lsh*: FingerprintLshIndex
    lexical*: LexicalIndex
    corpus*: CorpusIndex
    textCache*: Table[uint64, string]
    lowerTextCache*: Table[uint64, string]
    tokenCache*: Table[uint64, HashSet[string]]
    timestampCache*: Table[uint64, uint64]
    chunkToParent*: Table[uint64, uint64]
    parentToChunks*: Table[uint64, seq[uint64]]

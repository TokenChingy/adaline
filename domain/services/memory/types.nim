# Memory service types.
# Defines the MemoryService object that holds references to all
# memory-mapped stores and in-memory indexes.


import ../../entities/config
import ../../algorithms/fingerprint_lsh
import ../../algorithms/lexical_index
import ../../algorithms/corpus_index
import ../../../infrastructure/mmapped_storage
import std/tables

type
  MemoryService* = object
    storage*: MmappedStorage
    cfg*: EngineConfig
    lsh*: FingerprintLshIndex
    lexical*: LexicalIndex
    corpus*: CorpusIndex
    maxHnswLayer*: int
    hnswEntryPoint*: uint64
    textCache*: Table[uint64, string]
    timestampCache*: Table[uint64, uint64]
    chunkToParent*: Table[uint64, uint64]
    hnswReverseIndex*: Table[uint64, seq[uint64]]

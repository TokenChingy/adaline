# Chunk mapping entity.
# Links a parent memory ID to its child chunk IDs so that
# multi-chunk memories can be resolved back to a single parent.


type
  ChunkMapping* = object
    parentMemoryId*: uint64
    chunkId*: uint64

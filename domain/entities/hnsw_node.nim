## HNSW node entity.
## Defines the layout of an HNSW graph node: layer count,
## entry layer, and a flat neighbor ID array.
##
## Two APIs are provided:
## - The legacy ``ptr HnswNode`` API uses compile-time maximums (8 layers,
##   32 neighbours) and is retained for unit tests.
## - The new ``HnswNodeView`` API uses runtime-configurable layer and
##   neighbour counts, allowing the storage layer to shrink the on-disk
##   record to match the actual ``EngineConfig``.


const
  HnswMaxLayers* = 8
  HnswMaxNeighbors* = 32
  HnswNeighborSlots* = HnswMaxLayers * HnswMaxNeighbors

type
  HnswNode* = object
    layerCount*: uint8
    entryLayer*: uint8
    reserved*: array[6, uint8]
    neighbors*: array[HnswNeighborSlots, uint32]

  HnswNodeView* = object
    p*: pointer
    maxNeighbors*: int
    maxLayers*: int

static:
  doAssert sizeof(HnswNode) == 1032

proc hnswNodeRecordSize*(maxLayers, maxNeighbors: int): int {.inline.} =
  ## Compute the on-disk bytes needed for a node given the config.
  8 + maxLayers * maxNeighbors * sizeof(uint32)

proc makeView*(p: pointer; maxNeighbors, maxLayers: int): HnswNodeView {.inline.} =
  HnswNodeView(p: p, maxNeighbors: maxNeighbors, maxLayers: maxLayers)

proc layerCount*(v: HnswNodeView): uint8 {.inline.} =
  cast[ptr uint8](v.p)[]

proc entryLayer*(v: HnswNodeView): uint8 {.inline.} =
  cast[ptr uint8](cast[pointer](cast[uint](v.p) + 1))[]

proc `layerCount=`*(v: HnswNodeView; val: uint8) {.inline.} =
  cast[ptr uint8](v.p)[] = val

proc `entryLayer=`*(v: HnswNodeView; val: uint8) {.inline.} =
  cast[ptr uint8](cast[pointer](cast[uint](v.p) + 1))[] = val

iterator neighbors*(v: HnswNodeView; layer: int): uint64 =
  let start = layer * v.maxNeighbors
  let arr = cast[ptr UncheckedArray[uint32]](cast[pointer](cast[uint](v.p) + 8))
  for i in 0 ..< v.maxNeighbors:
    let nid = arr[start + i]
    if nid == 0:
      break
    yield uint64(nid)

proc setNeighbors*(v: HnswNodeView; layer: int; nids: seq[uint64]) =
  let start = layer * v.maxNeighbors
  let arr = cast[ptr UncheckedArray[uint32]](cast[pointer](cast[uint](v.p) + 8))
  for i in 0 ..< v.maxNeighbors:
    if i < nids.len:
      arr[start + i] = uint32(nids[i])
    else:
      arr[start + i] = 0

proc removeNeighbor*(v: HnswNodeView; layer: int; targetId: uint64): bool =
  let start = layer * v.maxNeighbors
  let arr = cast[ptr UncheckedArray[uint32]](cast[pointer](cast[uint](v.p) + 8))
  for i in 0 ..< v.maxNeighbors:
    if uint64(arr[start + i]) == targetId:
      for j in (start + i) ..< (start + v.maxNeighbors - 1):
        arr[j] = arr[j + 1]
      arr[start + v.maxNeighbors - 1] = 0
      return true
  return false

proc clearNode*(v: HnswNodeView) =
  let bytes = hnswNodeRecordSize(v.maxLayers, v.maxNeighbors)
  zeroMem(v.p, bytes)

# ---------------------------------------------------------------------------
# Legacy API (compile-time maximums) – kept for unit-test compatibility.
# ---------------------------------------------------------------------------

iterator neighbors*(node: ptr HnswNode; layer: int): uint64 =
  let start = layer * HnswMaxNeighbors
  for i in 0 ..< HnswMaxNeighbors:
    let nid = node.neighbors[start + i]
    if nid == 0:
      break
    yield uint64(nid)

proc setNeighbors*(node: ptr HnswNode; layer: int; nids: seq[uint64]) =
  let start = layer * HnswMaxNeighbors
  for i in 0 ..< HnswMaxNeighbors:
    if i < nids.len:
      node.neighbors[start + i] = uint32(nids[i])
    else:
      node.neighbors[start + i] = 0

proc removeNeighbor*(node: ptr HnswNode; layer: int; targetId: uint64): bool =
  let start = layer * HnswMaxNeighbors
  for i in 0 ..< HnswMaxNeighbors:
    if uint64(node.neighbors[start + i]) == targetId:
      for j in (start + i) ..< (start + HnswMaxNeighbors - 1):
        node.neighbors[j] = node.neighbors[j + 1]
      node.neighbors[start + HnswMaxNeighbors - 1] = 0
      return true
  return false

proc clearNode*(node: ptr HnswNode) =
  node.layerCount = 0
  node.entryLayer = 0
  for i in 0 ..< HnswNeighborSlots:
    node.neighbors[i] = 0

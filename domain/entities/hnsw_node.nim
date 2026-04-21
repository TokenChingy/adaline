# HNSW node entity.
# Defines the in-memory layout of an HNSW graph node: layer count,
# neighbor count per layer, and a flat neighbor ID array.
# Sized to 1,032 bytes for dense slot addressing.


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

static:
  doAssert sizeof(HnswNode) == 1032

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

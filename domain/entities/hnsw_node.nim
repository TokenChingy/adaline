const
  HnswMaxLayers* = 8
  HnswMaxNeighbors* = 32
  HnswNeighborSlots* = HnswMaxLayers * HnswMaxNeighbors  # 256

type
  HnswNode* = object
    layerCount*: uint8
    entryLayer*: uint8
    reserved*: array[6, uint8]
    neighbors*: array[HnswNeighborSlots, uint64]

static:
  doAssert sizeof(HnswNode) == 2056

iterator neighbors*(node: ptr HnswNode; layer: int): uint64 =
  let start = layer * HnswMaxNeighbors
  for i in 0 ..< HnswMaxNeighbors:
    let nid = node.neighbors[start + i]
    if nid == 0:
      break
    yield nid

proc setNeighbors*(node: ptr HnswNode; layer: int; nids: seq[uint64]) =
  let start = layer * HnswMaxNeighbors
  for i in 0 ..< HnswMaxNeighbors:
    if i < nids.len:
      node.neighbors[start + i] = nids[i]
    else:
      node.neighbors[start + i] = 0

proc clearNode*(node: ptr HnswNode) =
  node.layerCount = 0
  node.entryLayer = 0
  for i in 0 ..< HnswNeighborSlots:
    node.neighbors[i] = 0

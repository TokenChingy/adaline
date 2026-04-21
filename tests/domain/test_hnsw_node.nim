# Unit tests for HNSW node entity.


import unittest
import ../../domain/entities/hnsw_node

suite "HNSW node":
  test "sizeof HnswNode is 2056 bytes":
    check sizeof(HnswNode) == 1032

  test "clearNode zeros all fields":
    var node = HnswNode()
    node.layerCount = 3
    node.entryLayer = 2
    node.neighbors[0] = 100
    node.neighbors[5] = 200
    clearNode(addr node)
    check node.layerCount == 0
    check node.entryLayer == 0
    check node.neighbors[0] == 0
    check node.neighbors[5] == 0

  test "setNeighbors stores nids and pads with zeros":
    var node = HnswNode()
    setNeighbors(addr node, 0, @[1'u64, 2'u64, 3'u64])
    check node.neighbors[0] == 1
    check node.neighbors[1] == 2
    check node.neighbors[2] == 3
    check node.neighbors[3] == 0
    check node.neighbors[31] == 0

  test "setNeighbors on different layers does not overlap":
    var node = HnswNode()
    setNeighbors(addr node, 0, @[10'u64])
    setNeighbors(addr node, 1, @[20'u64])
    check node.neighbors[0] == 10
    check node.neighbors[32] == 20

  test "neighbors iterator yields stored ids":
    var node = HnswNode()
    setNeighbors(addr node, 0, @[100'u64, 200'u64, 300'u64])
    var yielded: seq[uint64]
    for nid in addr(node).neighbors(0):
      yielded.add(nid)
    check yielded == @[100'u64, 200'u64, 300'u64]

  test "neighbors iterator stops at first zero":
    var node = HnswNode()
    node.neighbors[0] = 1
    node.neighbors[1] = 0
    node.neighbors[2] = 3
    var yielded: seq[uint64]
    for nid in addr(node).neighbors(0):
      yielded.add(nid)
    check yielded == @[1'u64]

  test "neighbors iterator yields nothing for empty layer":
    var node = HnswNode()
    var yielded: seq[uint64]
    for nid in addr(node).neighbors(0):
      yielded.add(nid)
    check yielded.len == 0

## Unit tests for EngineConfig entity.


import unittest
import ../../domain/entities/config

suite "EngineConfig":
  test "defaultEngineConfig returns valid dimensions":
    let cfg = defaultEngineConfig()
    check cfg.fingerprintBytes == 1280
    check cfg.fingerprintBits == 10240
    check cfg.tokenBits + cfg.bigramBits + cfg.contextBits == cfg.fingerprintBits

  test "defaultEngineConfig has valid weights":
    let cfg = defaultEngineConfig()
    check cfg.tokenWeight + cfg.bigramWeight + cfg.contextWeight == 1.0

  test "defaultEngineConfig has positive probes":
    let cfg = defaultEngineConfig()
    check cfg.tokenProbes > 0
    check cfg.bigramProbes > 0
    check cfg.contextProbes > 0

  test "defaultEngineConfig has valid LSH coverage":
    let cfg = defaultEngineConfig()
    check cfg.lshBands * cfg.lshRows == 160

  test "defaultEngineConfig has positive HNSW parameters":
    let cfg = defaultEngineConfig()
    check cfg.hnswMaxLayers > 0
    check cfg.hnswMaxNeighbors > 0
    check cfg.hnswEfConstruction > 0
    check cfg.hnswEfSearch > 0

  test "defaultEngineConfig has semantic and lexical search enabled":
    let cfg = defaultEngineConfig()
    check cfg.semanticSearchEnabled == true
    check cfg.lexicalSearchEnabled == true

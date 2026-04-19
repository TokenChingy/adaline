type
  EngineConfig* = object
    # Fingerprint dimensions
    fingerprintBytes*: int
    fingerprintBits*: int

    # Block sizes in bits (must sum to fingerprintBits)
    tokenBits*: int
    bigramBits*: int
    contextBits*: int

    # Regional weights for weighted Jaccard (should sum to 1.0)
    tokenWeight*: float
    bigramWeight*: float
    contextWeight*: float

    # Encoding probes per block
    tokenProbes*: int
    tokenBigramProbes*: int
    bigramProbes*: int
    contextProbes*: int

    # Fingerprint LSH parameters (GoldFinger-style direct banding)
    lshBands*: int
    lshRows*: int

    # HNSW graph parameters
    hnswMaxLayers*: int
    hnswMaxNeighbors*: int
    hnswEfConstruction*: int
    hnswEfSearch*: int

    # Lexical smoothing parameter (Dirichlet mu)
    dirichletMu*: float

    # RRF constant
    rrfK*: int

    # RRF lane weights
    semanticRrfWeight*: float
    lexicalRrfWeight*: float

    # Reranker
    rerankCoverageWeight*: float

    # Chunking threshold (0.0–1.0); chunk when any block exceeds this saturation
    chunkSaturationThreshold*: float

    # Multiplier for query probes (makes query fingerprints denser)
    queryProbeMultiplier*: float

proc defaultEngineConfig*(): EngineConfig =
  result = EngineConfig(
    fingerprintBytes: 1280,
    fingerprintBits: 10240,
    tokenBits: 4096,
    bigramBits: 3072,
    contextBits: 3072,
    tokenWeight: 0.50,
    bigramWeight: 0.25,
    contextWeight: 0.25,
    tokenProbes: 4,
    tokenBigramProbes: 2,
    bigramProbes: 2,
    contextProbes: 2,
    lshBands: 50,
    lshRows: 2,
    hnswMaxLayers: 8,
    hnswMaxNeighbors: 32,
    hnswEfConstruction: 200,
    hnswEfSearch: 64,
    dirichletMu: 2000.0,
    rrfK: 60,
    semanticRrfWeight: 0.5,
    lexicalRrfWeight: 1.0,
    rerankCoverageWeight: 0.5,
    chunkSaturationThreshold: 0.6,
    queryProbeMultiplier: 2.0,
  )

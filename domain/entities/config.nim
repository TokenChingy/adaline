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

    # Reranker
    rerankCoverageWeight*: float

    # Chunking threshold (0.0–1.0); chunk when any block exceeds this saturation
    chunkSaturationThreshold*: float

    # Multiplier for query probes (makes query fingerprints denser)
    queryProbeMultiplier*: float

    # Ablation flags for benchmarking individual lanes
    semanticSearchEnabled*: bool
    lexicalSearchEnabled*: bool
    hnswEnabled*: bool

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
    tokenProbes: 3,
    tokenBigramProbes: 2,
    bigramProbes: 1,
    contextProbes: 1,
    lshBands: 80,
    lshRows: 2,
    hnswMaxLayers: 8,
    hnswMaxNeighbors: 32,
    hnswEfConstruction: 200,
    hnswEfSearch: 64,
    dirichletMu: 2000.0,
    rrfK: 10,
    rerankCoverageWeight: 0.5,
    chunkSaturationThreshold: 0.6,
    queryProbeMultiplier: 2.0,
    semanticSearchEnabled: true,
    lexicalSearchEnabled: true,
    hnswEnabled: true,
  )

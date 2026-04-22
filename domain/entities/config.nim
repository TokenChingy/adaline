## Engine configuration entity.
## Defines all tunable parameters: fingerprint dimensions, SDR probe
## counts, LSH banding, lexical smoothing, RRF constants, chunking
## thresholds, and query-time multipliers.


type
  EngineConfig* = object
    fingerprintBytes*: int
    fingerprintBits*: int

    tokenBits*: int
    bigramBits*: int
    contextBits*: int

    tokenWeight*: float
    bigramWeight*: float
    contextWeight*: float

    tokenProbes*: int
    tokenBigramProbes*: int
    bigramProbes*: int
    contextProbes*: int

    lshBands*: int
    lshRows*: int

    dirichletMu*: float

    rrfK*: int


    rerankCoverageWeight*: float

    chunkSaturationThreshold*: float

    queryProbeMultiplier*: float

    maxTokenFeatures*: int

    semanticSearchEnabled*: bool
    lexicalSearchEnabled*: bool

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
    dirichletMu: 2000.0,
    rrfK: 10,
    rerankCoverageWeight: 0.5,
    chunkSaturationThreshold: 0.6,
    queryProbeMultiplier: 2.0,
    maxTokenFeatures: 12,
    semanticSearchEnabled: true,
    lexicalSearchEnabled: true,
  )

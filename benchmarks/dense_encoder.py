"""
Dense float32 vector → sparse binary fingerprint encoders.

Fingerprint structure mirrors Adaline exactly:
  10,240 bits  =  160 × uint64  =  1,280 bytes

The hashFeature function is a direct Python port of Adaline's
hashFeature() in domain/algorithms/sdr_encoder.nim — the same
deterministic locality-preserving hash used for text features.

Three encoder variants:

  KWTAEncoder     — k-Winners Take All (mirrors Adaline's probe pattern,
                    recommended default)
  SRPEncoder      — Signed Random Projection / SimHash (theoretically
                    grounded for cosine similarity preservation)
  CountSketchEncoder — fastest; one hash per winning dimension
"""
import numpy as np
from abc import ABC, abstractmethod

FINGERPRINT_BITS     = 10240
FINGERPRINT_SEGMENTS = 160   # 160 × uint64 = 1,280 bytes

# Pre-computed popcount for all 256 byte values — avoids np.unpackbits which
# allocates O(P × 10240) bools per query and becomes the bottleneck at scale.
_BYTE_POPCOUNT = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint16)


# ─── Primitives ─────────────────────────────────────────────────────────────

def _hash_feature(feature: str, seed: int = 0) -> int:
    """Direct port of Adaline's hashFeature() (sdr_encoder.nim)."""
    M = 0xFFFF_FFFF_FFFF_FFFF
    h = seed & M
    for c in feature.encode('utf-8'):
        h = (h + c)          & M
        h = (h + (h << 10))  & M
        h = (h ^ (h >>  6))  & M
    h = (h + (h <<  3)) & M
    h = (h ^ (h >> 11)) & M
    h = (h + (h << 15)) & M
    return h


def make_fingerprint() -> np.ndarray:
    return np.zeros(FINGERPRINT_SEGMENTS, dtype=np.uint64)


def _set_bit(fp: np.ndarray, pos: int):
    fp[pos >> 6] |= np.uint64(1) << np.uint64(pos & 63)


def fp_popcount(fp: np.ndarray) -> int:
    return int(_BYTE_POPCOUNT[fp.view(np.uint8)].sum())


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(_BYTE_POPCOUNT[(a & b).view(np.uint8)].sum())
    union = int(_BYTE_POPCOUNT[(a | b).view(np.uint8)].sum())
    return inter / max(union, 1)


def jaccard_query(query: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    """Vectorised: query (160,) vs prototypes (P, 160) → similarities (P,) float32.

    Uses a byte-level popcount lookup table.  At P=5000 this allocates ~6 MB
    instead of the ~50 MB that np.unpackbits would require, and runs ~3× faster.
    """
    q = query[None, :]
    inter = _BYTE_POPCOUNT[(q & prototypes).view(np.uint8)].sum(axis=1)
    union = _BYTE_POPCOUNT[(q | prototypes).view(np.uint8)].sum(axis=1)
    return (inter / np.maximum(union, 1)).astype(np.float32)


# ─── Base class ─────────────────────────────────────────────────────────────

class DenseEncoder(ABC):
    @abstractmethod
    def encode(self, vec: np.ndarray) -> np.ndarray:
        """Encode a single L2-normalised vector to a fingerprint (160 × uint64)."""
        ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def nominal_active_bits(self) -> int:
        """Upper-bound estimate of active bits (actual may be lower due to collisions)."""
        ...

    def encode_batch(self, vecs: np.ndarray) -> np.ndarray:
        return np.stack([self.encode(v) for v in vecs])


# ─── k-WTA ──────────────────────────────────────────────────────────────────

class KWTAEncoder(DenseEncoder):
    """k-Winners Take All encoder.

    Mirrors Adaline's SDR probe pattern: each winning dimension is a
    'feature' string that fires multiple probes into the fingerprint space
    via the same hashFeature hash used for text tokens.  High-magnitude
    winners fire more probes (magnitude-weighted sparsity).
    """

    def __init__(self, k: int = 128, probes: int = 4,
                 fingerprint_bits: int = FINGERPRINT_BITS):
        self.k = k
        self.probes = probes
        self.fingerprint_bits = fingerprint_bits

    @property
    def name(self) -> str:
        return f'kwta_k{self.k}'

    @property
    def nominal_active_bits(self) -> int:
        return self.k * self.probes

    def encode(self, vec: np.ndarray) -> np.ndarray:
        vec = vec.astype(np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm

        top_k = np.argpartition(np.abs(vec), -self.k)[-self.k:]
        fp = make_fingerprint()
        for idx in top_k:
            sign = 'p' if vec[idx] >= 0 else 'n'
            feature = f'{int(idx)}{sign}'
            mag = float(abs(vec[idx]))
            n_probes = max(1, min(self.probes, round(mag * self.probes * 3)))
            for probe in range(n_probes):
                h = _hash_feature(feature, probe)
                _set_bit(fp, h % self.fingerprint_bits)
        return fp


# ─── SRP ────────────────────────────────────────────────────────────────────

class SRPEncoder(DenseEncoder):
    """Signed Random Projection (SimHash-style).

    Projects the input vector onto fingerprint_bits random hyperplanes;
    the top-k projections by magnitude become the active bits.  More
    theoretically grounded for cosine similarity preservation than k-WTA.

    The projection matrix is built once on first use (seed-deterministic).
    For 1280-dim input: 10240×1280 float32 = 52 MB.
    """

    def __init__(self, k: int = 128, fingerprint_bits: int = FINGERPRINT_BITS,
                 seed: int = 42):
        self.k = k
        self.fingerprint_bits = fingerprint_bits
        self.seed = seed
        self._proj: np.ndarray | None = None
        self._proj_dim: int = -1

    @property
    def name(self) -> str:
        return f'srp_k{self.k}'

    @property
    def nominal_active_bits(self) -> int:
        return self.k

    def _get_proj(self, dim: int) -> np.ndarray:
        if self._proj is None or self._proj_dim != dim:
            rng = np.random.default_rng(self.seed)
            P = rng.standard_normal((self.fingerprint_bits, dim)).astype(np.float32)
            P /= np.linalg.norm(P, axis=1, keepdims=True)
            self._proj = P
            self._proj_dim = dim
        return self._proj

    def encode(self, vec: np.ndarray) -> np.ndarray:
        vec = vec.astype(np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        dots = self._get_proj(len(vec)) @ vec
        top_k = np.argpartition(np.abs(dots), -self.k)[-self.k:]
        fp = make_fingerprint()
        for idx in top_k:
            _set_bit(fp, int(idx))
        return fp

    def encode_batch(self, vecs: np.ndarray) -> np.ndarray:
        """Vectorised batch via single matrix multiply."""
        vecs = vecs.astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / np.maximum(norms, 1e-8)
        dots = vecs @ self._get_proj(vecs.shape[1]).T   # (N, F)
        top_k = np.argpartition(np.abs(dots), -self.k, axis=1)[:, -self.k:]
        fps = np.zeros((len(vecs), FINGERPRINT_SEGMENTS), dtype=np.uint64)
        for i in range(len(vecs)):
            for idx in top_k[i]:
                fps[i, int(idx) >> 6] |= np.uint64(1) << np.uint64(int(idx) & 63)
        return fps


# ─── Count Sketch ────────────────────────────────────────────────────────────

class CountSketchEncoder(DenseEncoder):
    """Count Sketch: each winning dimension hashed to one bit position.

    No random matrix needed — fastest encoder.  Top-k selection keeps
    sparsity consistent with the other encoders.
    """

    def __init__(self, k: int = 128, fingerprint_bits: int = FINGERPRINT_BITS,
                 seed: int = 42):
        self.k = k
        self.fingerprint_bits = fingerprint_bits
        self.seed = seed

    @property
    def name(self) -> str:
        return f'sketch_k{self.k}'

    @property
    def nominal_active_bits(self) -> int:
        return self.k

    def encode(self, vec: np.ndarray) -> np.ndarray:
        vec = vec.astype(np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        top_k = np.argpartition(np.abs(vec), -self.k)[-self.k:]
        fp = make_fingerprint()
        for idx in top_k:
            h = _hash_feature(str(int(idx)), self.seed)
            _set_bit(fp, h % self.fingerprint_bits)
        return fp

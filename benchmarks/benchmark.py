#!/usr/bin/env python3
"""
Adaline vision benchmark suite.

Demonstrates Adaline's sparse-fingerprint retrieval on non-NLP inputs
(CIFAR-10 / pretrained CNN features).  Each benchmark includes a dense
cosine-similarity baseline to show how much the sparse encoding preserves.

Requirements:
    pip install torch torchvision numpy

Usage:
    python benchmark.py --dataset cifar10 --benchmark footprint
    python benchmark.py --dataset cifar10 --benchmark classify
    python benchmark.py --dataset cifar10 --benchmark fewshot
    python benchmark.py --dataset cifar10 --benchmark incremental
    python benchmark.py --dataset cifar10 --benchmark openset
    python benchmark.py --dataset cifar10 --benchmark all
    python benchmark.py --dataset cifar10 --benchmark all --backbone resnet50
    python benchmark.py --dataset imagenet --benchmark comparison --backbone resnet50
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from feature_extractor import load_or_extract_cifar10, load_or_extract_imagenet, BACKBONE_DIMS
from dense_encoder import (
    KWTAEncoder, SRPEncoder, CountSketchEncoder,
    jaccard_query, fp_popcount,
    FINGERPRINT_BITS, FINGERPRINT_SEGMENTS,
)

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck',
]

# Published FLOPs (GFLOPs) for one forward pass through the backbone only.
# Source: torchvision model cards / standard benchmarks.
BACKBONE_GFLOPS = {
    'mobilenet_v2':    0.314,
    'efficientnet_b0': 0.394,
    'resnet50':        4.134,
}

# Energy cost per FP32 floating-point operation for reference hardware.
# Source: published chip-level benchmarks (pJ/FLOP at FP32 peak throughput).
_ENERGY_PJ_PER_FLOP = {
    'GPU  (A100 FP32)':    1.3,
    'CPU  (x86 server)':  50.0,
    'Edge (ARM Cortex)': 500.0,
}


# ─── Shared helpers ──────────────────────────────────────────────────────────

def get_class_indices(labels: np.ndarray, n_classes: int = 10) -> dict:
    return {c: np.where(labels == c)[0] for c in range(n_classes)}


def select_prototypes(feats: np.ndarray, class_idx: dict,
                      n_proto: int, rng: np.random.Generator) -> tuple:
    pf, pl = [], []
    for c, indices in class_idx.items():
        chosen = rng.choice(indices, size=min(n_proto, len(indices)), replace=False)
        pf.append(feats[chosen])
        pl.extend([c] * len(chosen))
    return np.vstack(pf), np.array(pl)


def select_queries(feats: np.ndarray, class_idx: dict, n_query: int,
                   rng: np.random.Generator) -> tuple:
    qf, ql = [], []
    for c, indices in class_idx.items():
        chosen = rng.choice(indices, size=min(n_query, len(indices)), replace=False)
        qf.append(feats[chosen])
        ql.extend([c] * len(chosen))
    return np.vstack(qf), np.array(ql)


def accuracy(pred: np.ndarray, truth: np.ndarray) -> float:
    return float((pred == truth).mean())


def aggregate_class_sims(proto_sims: np.ndarray, proto_labels: np.ndarray,
                         n_classes: int) -> np.ndarray:
    """Max-pool per-prototype similarities to per-class. (Q, P) → (Q, C)."""
    class_sims = np.full((len(proto_sims), n_classes), -1.0, dtype=np.float32)
    for c in range(n_classes):
        mask = proto_labels == c
        if mask.any():
            class_sims[:, c] = proto_sims[:, mask].max(axis=1)
    return class_sims


def topk_accuracy(class_sims: np.ndarray, query_labels: np.ndarray, k: int) -> float:
    top_k = np.argpartition(class_sims, -k, axis=1)[:, -k:]
    return float(np.any(top_k == query_labels[:, None], axis=1).mean())


def auroc(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """AUROC via Mann-Whitney U statistic (no sklearn dependency)."""
    pos = np.asarray(pos_scores, dtype=np.float64)
    neg = np.asarray(neg_scores, dtype=np.float64)
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    count = float(np.sum(pos[:, None] > neg[None, :]))
    ties  = float(np.sum(pos[:, None] == neg[None, :]))
    return (count + 0.5 * ties) / (len(pos) * len(neg))


# ─── Retrieval functions ─────────────────────────────────────────────────────

def dense_retrieve(query_feats: np.ndarray,
                   proto_feats: np.ndarray,
                   proto_labels: np.ndarray) -> tuple:
    """Top-1 NN by cosine similarity.  L2-normed → dot product suffices."""
    t0 = time.perf_counter()
    sims = query_feats @ proto_feats.T
    pred = proto_labels[sims.argmax(axis=1)]
    ms_per_q = (time.perf_counter() - t0) * 1000 / len(query_feats)
    return pred, sims, ms_per_q


def sparse_retrieve(query_fps: np.ndarray,
                    proto_fps: np.ndarray,
                    proto_labels: np.ndarray) -> tuple:
    """Top-1 NN by Jaccard similarity over fingerprints."""
    t0 = time.perf_counter()
    sims_list, preds = [], []
    for qfp in query_fps:
        s = jaccard_query(qfp, proto_fps)
        sims_list.append(s)
        preds.append(proto_labels[s.argmax()])
    ms_per_q = (time.perf_counter() - t0) * 1000 / len(query_fps)
    return np.array(preds), np.stack(sims_list), ms_per_q


# ─── Benchmark 1: classify ───────────────────────────────────────────────────

def bench_classify(tr_feats, tr_labels, te_feats, te_labels,
                   encoders, n_query: int = 50, seed: int = 0):
    print('\n=== Benchmark 1: classify (1-shot, 50 queries/class) ===\n')
    rng = np.random.default_rng(seed)
    proto_feats, proto_labels = select_prototypes(
        tr_feats, get_class_indices(tr_labels), 1, rng)
    query_feats, query_labels = select_queries(
        te_feats, get_class_indices(te_labels), n_query, rng)

    print(f"  {'Method':<22} {'Accuracy':>10} {'ms/query':>10}")
    print('  ' + '-' * 46)

    pred, _, ms = dense_retrieve(query_feats, proto_feats, proto_labels)
    print(f"  {'dense (cosine)':<22} {accuracy(pred, query_labels):>9.1%} {ms:>10.3f}")

    for enc in encoders:
        t0 = time.perf_counter()
        proto_fps = enc.encode_batch(proto_feats)
        query_fps = enc.encode_batch(query_feats)
        pred, _, _ = sparse_retrieve(query_fps, proto_fps, proto_labels)
        ms = (time.perf_counter() - t0) * 1000 / len(query_feats)
        print(f"  {enc.name:<22} {accuracy(pred, query_labels):>9.1%} {ms:>10.3f}")


# ─── Benchmark 2: fewshot ────────────────────────────────────────────────────

def bench_fewshot(tr_feats, tr_labels, te_feats, te_labels,
                  encoders, shots=(1, 2, 5, 10, 20),
                  n_query: int = 50, seed: int = 0):
    print('\n=== Benchmark 2: fewshot (accuracy vs prototype count) ===\n')
    rng = np.random.default_rng(seed)
    class_idx_tr = get_class_indices(tr_labels)
    class_idx_te = get_class_indices(te_labels)
    query_feats, query_labels = select_queries(
        te_feats, class_idx_te, n_query, rng)

    methods = ['dense'] + [enc.name for enc in encoders]
    header = f"  {'Shots':<8}" + ''.join(f"{m:>16}" for m in methods)
    print(header)
    print('  ' + '-' * (8 + 16 * len(methods)))

    for n_proto in shots:
        proto_feats, proto_labels = select_prototypes(
            tr_feats, class_idx_tr, n_proto, rng)
        pred, _, _ = dense_retrieve(query_feats, proto_feats, proto_labels)
        row = f"  {n_proto:<8}{accuracy(pred, query_labels):>15.1%}"

        for enc in encoders:
            proto_fps = enc.encode_batch(proto_feats)
            query_fps = enc.encode_batch(query_feats)
            pred, _, _ = sparse_retrieve(query_fps, proto_fps, proto_labels)
            row += f"{accuracy(pred, query_labels):>16.1%}"
        print(row)

    print()
    print('  Adeline showcase: accuracy should improve rapidly with more prototypes')
    print('  and the sparse encoders should stay within a few % of dense.')


# ─── Benchmark 3: incremental ────────────────────────────────────────────────

def bench_incremental(tr_feats, tr_labels, te_feats, te_labels,
                      encoders, n_proto: int = 5, n_query: int = 50,
                      steps: list = None, top_ks: tuple = (1,), seed: int = 0):
    """Incremental class-addition benchmark with configurable steps and top-k metrics.

    steps    — class counts to evaluate, e.g. [10, 50, 100, 500, 1000]
    top_ks   — accuracy metrics to report, e.g. (1, 5); top-5 is meaningful
               when n_classes is large enough that random chance is <20%
    """
    n_total = int(np.max(tr_labels)) + 1
    if steps is None:
        steps = list(range(2, min(n_total + 1, 11), 2))

    print(f'\n=== Benchmark 3: incremental '
          f'({n_proto} prototypes/class, {n_query} queries/class, '
          f'classes {steps[0]}→{steps[-1]}) ===\n')

    rng = np.random.default_rng(seed)
    class_idx_tr = get_class_indices(tr_labels, n_total)
    class_idx_te = get_class_indices(te_labels, n_total)
    methods = ['dense'] + [enc.name for enc in encoders]

    # Collect (class_sims (Q,C), ms_per_q) per method per step
    step_results = {}
    step_ql      = {}

    for n_cls in steps:
        active_tr = {c: class_idx_tr[c] for c in range(n_cls)}
        active_te = {c: class_idx_te[c] for c in range(n_cls)}
        pf, pl = select_prototypes(tr_feats, active_tr, n_proto, rng)
        qf, ql = select_queries(te_feats, active_te, n_query, rng)
        step_ql[n_cls] = ql
        step_results[n_cls] = {}

        # Dense
        t0 = time.perf_counter()
        sims = qf @ pf.T
        cs = aggregate_class_sims(sims, pl, n_cls)
        ms = (time.perf_counter() - t0) * 1000 / len(qf)
        step_results[n_cls]['dense'] = (cs, ms)

        # Sparse encoders
        for enc in encoders:
            t0 = time.perf_counter()
            pfps = enc.encode_batch(pf)
            qfps = enc.encode_batch(qf)
            proto_sims = np.stack([jaccard_query(q, pfps) for q in qfps])
            cs = aggregate_class_sims(proto_sims, pl, n_cls)
            ms = (time.perf_counter() - t0) * 1000 / len(qf)
            step_results[n_cls][enc.name] = (cs, ms)

    # Print one table per top_k
    col_w = 14
    for k in top_ks:
        label = f"Top-{k} Accuracy"
        print(f"  {label}:")
        header = f"  {'Classes':<10}" + ''.join(f"{m:>{col_w}}" for m in methods)
        print(header)
        print('  ' + '-' * len(header))
        for n_cls in steps:
            ql = step_ql[n_cls]
            row = f"  {n_cls:<10}"
            for m in methods:
                cs, _ = step_results[n_cls][m]
                row += f"{topk_accuracy(cs, ql, k):>{col_w}.1%}"
            print(row)
        print()

    # Timing table at the largest class count
    n_last = steps[-1]
    print(f"  Retrieval time at {n_last} classes (ms/query, includes encoding):")
    for m in methods:
        _, ms = step_results[n_last][m]
        print(f"    {m:<22} {ms:.3f} ms")
    print()
    print('  Key insight: accuracy should degrade gracefully as classes grow')
    print('  with no retraining — and sparse encoders should track dense closely.')


# ─── Benchmark 4: openset ────────────────────────────────────────────────────

def bench_openset(tr_feats, tr_labels, te_feats, te_labels,
                  encoders, n_known: int = 6, n_proto: int = 5,
                  n_query: int = 50, seed: int = 0):
    print(f'\n=== Benchmark 4: openset (enrol {n_known} classes, query all 10) ===\n')
    rng = np.random.default_rng(seed)
    class_idx_tr = get_class_indices(tr_labels)
    class_idx_te = get_class_indices(te_labels)

    known = set(range(n_known))
    known_tr = {c: class_idx_tr[c] for c in known}
    proto_feats, proto_labels = select_prototypes(tr_feats, known_tr, n_proto, rng)

    query_feats, query_labels = select_queries(
        te_feats, class_idx_te, n_query, rng)
    is_known = np.array([int(l) in known for l in query_labels])

    print(f"  Known classes: {list(known)}  ({is_known.sum()} known queries, "
          f"{(~is_known).sum()} novel queries)\n")
    print(f"  {'Method':<22} {'AUROC':>7} {'Known acc':>10} {'Known score':>12} {'Novel score':>12}")
    print('  ' + '-' * 67)

    def report(name: str, scores: np.ndarray, preds: np.ndarray):
        auc = auroc(scores[is_known], scores[~is_known])
        kc  = accuracy(preds[is_known], query_labels[is_known])
        mk  = float(scores[is_known].mean())
        mn  = float(scores[~is_known].mean())
        print(f"  {name:<22} {auc:>7.3f} {kc:>9.1%} {mk:>12.4f} {mn:>12.4f}")

    sims = query_feats @ proto_feats.T
    report('dense (cosine)',
           sims.max(axis=1),
           proto_labels[sims.argmax(axis=1)])

    for enc in encoders:
        proto_fps = enc.encode_batch(proto_feats)
        query_fps = enc.encode_batch(query_feats)
        all_scores, all_preds = [], []
        for qfp in query_fps:
            s = jaccard_query(qfp, proto_fps)
            all_scores.append(float(s.max()))
            all_preds.append(int(proto_labels[s.argmax()]))
        report(enc.name, np.array(all_scores), np.array(all_preds))

    print()
    print('  AUROC measures how well the similarity score separates known')
    print('  queries (should score high) from novel queries (should score low).')
    print('  Expected: 0.70–0.85 for dense baseline.')


# ─── Benchmark 5: footprint ──────────────────────────────────────────────────

def bench_footprint(feat_dim: int, encoders, n_samples: int = 50):
    print('\n=== Benchmark 5: footprint (memory comparison) ===\n')
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((n_samples, feat_dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

    dense_bytes = feat_dim * 4
    print(f"  Input dimension : {feat_dim}")
    print(f"  Fingerprint size: {FINGERPRINT_BITS} bits = "
          f"{FINGERPRINT_BITS // 8} bytes = "
          f"{FINGERPRINT_SEGMENTS} × uint64")
    print(f"  Dense storage   : {feat_dim} dims × 4 bytes = {dense_bytes} bytes/vector\n")

    print(f"  {'Encoder':<24} {'Active bits':>12} {'Sparse bytes':>14} {'Ratio':>8}")
    print('  ' + '-' * 62)

    for enc in encoders:
        fps = enc.encode_batch(vecs)
        avg_bits = float(np.mean([fp_popcount(fp) for fp in fps]))
        sparse_bytes = avg_bits * 2  # uint16 position list
        ratio = dense_bytes / max(sparse_bytes, 1)
        print(f"  {enc.name:<24} {avg_bits:>11.1f} {sparse_bytes:>13.0f} {ratio:>7.1f}x")

    print()
    print('  Sparse bytes assumes uint16 active-bit position list.')
    print('  Adaline stores fingerprints as packed bit arrays '
          f'({FINGERPRINT_BITS // 8} bytes each — fixed width).')


# ─── Benchmark 6: ResNet-50 classification vs Adaline retrieval ──────────────

def _load_fc_head(backbone: str) -> tuple:
    """Extract the linear classification head from a pretrained torchvision model.

    Returns (weight, bias) as float32 numpy arrays so we can apply the head
    directly to already-cached backbone features without re-running the full model.

    Shape: weight (n_classes, feat_dim), bias (n_classes,)
    """
    import torchvision.models as models
    if backbone == 'resnet50':
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        return m.fc.weight.detach().numpy(), m.fc.bias.detach().numpy()
    elif backbone == 'mobilenet_v2':
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        return m.classifier[1].weight.detach().numpy(), m.classifier[1].bias.detach().numpy()
    elif backbone == 'efficientnet_b0':
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        return m.classifier[1].weight.detach().numpy(), m.classifier[1].bias.detach().numpy()
    raise ValueError(f"Unknown backbone: {backbone!r}")


def bench_resnet_comparison(tr_feats, tr_labels, te_feats, te_labels,
                             encoders, backbone: str,
                             steps: list = None, n_proto: int = 5,
                             n_query: int = 10, seed: int = 0):
    """Compare ResNet-50's trained classification head against 5-shot Adaline retrieval.

    Both branches start from the SAME backbone features — identical FLOPs for
    feature extraction.  The only difference is what happens after the backbone:

      Classification head  — one matrix-vector product against 1000 trained weights;
                             requires full supervision (1.28 M labelled images).
      Dense retrieval       — dot product against 5 stored prototypes per class;
                             5-shot, no gradient updates needed.
      Sparse retrieval      — Jaccard against 5 fingerprints per class (Adaline);
                             5-shot, 6.4× smaller index than dense.

    Accuracy tables show top-1 and top-5 across incremental class counts.
    Resource table shows memory, FLOPs, latency, and estimated energy per query.
    """
    n_total = int(np.max(tr_labels)) + 1
    if steps is None:
        steps = [10, 50, 100, 200, 500, 1000]

    print(f'\n=== Benchmark 6: classification head vs Adaline retrieval '
          f'({backbone}, {n_proto}-shot) ===\n')
    print('  Supervision:')
    print('    classification head  — pretrained on full ImageNet (1.28M images)')
    print('    dense / sparse       — 5-shot registration, zero gradient updates\n')

    print('  Loading classification head weights...')
    fc_w, fc_b = _load_fc_head(backbone)   # (1000, D), (1000,)
    # Mask template: set inactive class logits to -inf so argpartition ignores them.
    base_mask = np.full(fc_w.shape[0], -1e9, dtype=np.float32)

    rng = np.random.default_rng(seed)
    class_idx_tr = get_class_indices(tr_labels, n_total)
    class_idx_te = get_class_indices(te_labels, n_total)

    methods  = ['cls_head'] + ['dense'] + [enc.name for enc in encoders]
    labels   = ['ResNet cls.head'] + ['dense (5-shot)'] + [enc.name + ' (5-shot)' for enc in encoders]
    col_w    = 17

    # ── Collect per-step results ─────────────────────────────────────────────
    step_results: dict = {}   # step → {method: (class_sims, ms_per_q)}
    step_ql:     dict = {}

    for n_cls in steps:
        active_tr = {c: class_idx_tr[c] for c in range(n_cls)}
        active_te = {c: class_idx_te[c] for c in range(n_cls)}
        pf, pl = select_prototypes(tr_feats, active_tr, n_proto, rng)
        qf, ql = select_queries(te_feats, active_te, n_query, rng)
        step_ql[n_cls]     = ql
        step_results[n_cls] = {}

        # Classification head (masked to n_cls active classes)
        mask = base_mask.copy()
        mask[:n_cls] = 0.0
        t0 = time.perf_counter()
        logits = qf @ fc_w.T + fc_b + mask[None, :]   # (Q, 1000)
        ms = (time.perf_counter() - t0) * 1000 / len(qf)
        step_results[n_cls]['cls_head'] = (logits, ms)

        # Dense retrieval
        t0 = time.perf_counter()
        sims = qf @ pf.T
        cs   = aggregate_class_sims(sims, pl, n_cls)
        ms   = (time.perf_counter() - t0) * 1000 / len(qf)
        step_results[n_cls]['dense'] = (cs, ms)

        # Sparse encoders
        for enc in encoders:
            t0   = time.perf_counter()
            pfps = enc.encode_batch(pf)
            qfps = enc.encode_batch(qf)
            proto_sims = np.stack([jaccard_query(q, pfps) for q in qfps])
            cs   = aggregate_class_sims(proto_sims, pl, n_cls)
            ms   = (time.perf_counter() - t0) * 1000 / len(qf)
            step_results[n_cls][enc.name] = (cs, ms)

    # ── Accuracy tables (top-1 and top-5) ────────────────────────────────────
    for k in (1, 5):
        print(f"  Top-{k} Accuracy:")
        header = f"  {'Classes':<10}" + ''.join(f"{lbl:>{col_w}}" for lbl in labels)
        print(header)
        print('  ' + '-' * len(header))
        for n_cls in steps:
            ql  = step_ql[n_cls]
            row = f"  {n_cls:<10}"
            for m in methods:
                cs, _ = step_results[n_cls][m]
                # cls_head returns (Q, 1000) logits; topk_accuracy handles this
                # because inactive classes are -1e9 and never appear in top-k.
                row += f"{topk_accuracy(cs, ql, k):>{col_w}.1%}"
            print(row)
        print()

    # ── Resource table ────────────────────────────────────────────────────────
    feat_dim       = tr_feats.shape[1]
    n_cls_max      = steps[-1]
    n_proto_total  = n_cls_max * n_proto

    # FLOPs for each "head" (after the backbone, per single query):
    #   cls head:   2 × D × C  multiply-adds  (matrix-vector product)
    #   dense:      2 × D × P  multiply-adds  (dot product against P protos)
    #   sparse:     3 × P × FINGERPRINT_SEGMENTS  bit ops (AND, OR, popcount)
    #               bit ops are ~10-30× cheaper per op than FP32 MACs, so we
    #               list them separately rather than converting to FLOP-equivalents.
    cls_head_mflops   = 2 * feat_dim * n_cls_max     / 1e6
    dense_head_mflops = 2 * feat_dim * n_proto_total / 1e6
    sparse_mbitops    = 3 * n_proto_total * FINGERPRINT_SEGMENTS / 1e6

    # Index memory at maximum class count (does NOT include model weights):
    cls_head_idx_mb   = feat_dim * n_cls_max     * 4 / 1e6   # FC weight matrix
    dense_idx_mb      = feat_dim * n_proto_total * 4 / 1e6
    sparse_idx_mb     = n_proto_total * (FINGERPRINT_BITS // 8) / 1e6

    # Total FLOPs = backbone + head (backbone dominates)
    bb_gflops = BACKBONE_GFLOPS.get(backbone, 0.0)

    # Latency: take timing from largest step
    ms_cls   = step_results[n_cls_max]['cls_head'][1]
    ms_dense = step_results[n_cls_max]['dense'][1]
    ms_enc   = {enc.name: step_results[n_cls_max][enc.name][1] for enc in encoders}

    print(f"  Resource comparison at {n_cls_max} classes, {n_proto} prototypes/class "
          f"({n_proto_total:,} total items):\n")

    cw = 14
    print(f"  {'Method':<22} {'Supervision':<22} {'Index memory':>{cw}} "
          f"{'Head FLOPs':>{cw}} {'ms/query':>{cw}}")
    print('  ' + '-' * (22 + 22 + cw * 3 + 3))

    print(f"  {'cls_head':<22} {'full (1.28M imgs)':<22} "
          f"{cls_head_idx_mb:>{cw-3}.1f} MB   "
          f"{cls_head_mflops:>{cw-4}.1f} MFP   "
          f"{ms_cls:>{cw-3}.3f} ms")
    print(f"  {'dense retrieval':<22} {'5-shot':<22} "
          f"{dense_idx_mb:>{cw-3}.1f} MB   "
          f"{dense_head_mflops:>{cw-4}.1f} MFP   "
          f"{ms_dense:>{cw-3}.3f} ms")
    for enc in encoders:
        print(f"  {enc.name:<22} {'5-shot':<22} "
              f"{sparse_idx_mb:>{cw-3}.1f} MB   "
              f"{sparse_mbitops:>{cw-4}.1f} Mbit  "
              f"{ms_enc[enc.name]:>{cw-3}.3f} ms*")

    print()
    print(f"  * Brute-force Jaccard over all {n_proto_total:,} fingerprints.")
    print(f"    Adaline's HNSW searches O(log N) nodes: estimated < 1 ms/query")
    print(f"    (from Nim benchmarks: ~222 q/s on SciFact corpus).")
    print(f"    MFP = mega floating-point ops.  Mbit = mega bit ops (~10-30× cheaper/op).\n")

    # Index scaling table
    print(f"  Index memory at scale (dense float32 vs Adaline packed fingerprints):\n")
    print(f"  {'Items':>10}  {'Dense (float32)':>18}  {'Sparse (Adaline)':>18}  "
          f"{'Cls. head (fixed)':>20}")
    print('  ' + '-' * 72)
    for n_items in [1_000, 10_000, 100_000, 1_000_000]:
        d_mb  = n_items * feat_dim * 4 / 1e6
        s_mb  = n_items * (FINGERPRINT_BITS // 8) / 1e6
        fixed = f"{cls_head_idx_mb:.1f} MB (no new classes)"
        print(f"  {n_items:>10,}  {d_mb:>16.0f} MB  {s_mb:>16.1f} MB  {fixed:>20}")

    print()

    # Energy table
    print(f"  Estimated inference energy (backbone + head, per query):\n")
    print(f"  {'Platform':<24} {'Backbone':>14} {'+ cls head':>12} "
          f"{'+ dense head':>14} {'+ sparse head':>15}")
    print('  ' + '-' * 81)
    for platform, pj in _ENERGY_PJ_PER_FLOP.items():
        bb_uj       = bb_gflops * 1e9 * pj * 1e-12 * 1e6
        cls_uj      = cls_head_mflops  * 1e6 * pj * 1e-12 * 1e6
        dense_uj    = dense_head_mflops * 1e6 * pj * 1e-12 * 1e6
        # Bit ops use separate (cheaper) execution units; use pJ/op ÷ 20 as estimate
        sparse_uj   = sparse_mbitops * 1e6 * (pj / 20) * 1e-12 * 1e6
        print(f"  {platform:<24} {bb_uj:>12.2f} μJ "
              f"{cls_uj:>+10.3f} μJ "
              f"{dense_uj:>+12.3f} μJ "
              f"{sparse_uj:>+13.3f} μJ")

    print()
    print(f"  Backbone ({backbone}, {bb_gflops} GFLOPs) dominates total energy for all methods.")
    print(f"  The retrieval head is < 1% of total compute in every case.")
    print(f"  Bit-op energy estimated at pJ/op ÷ 20 vs FP32 (conservative; "
          f"hardware popcount is typically faster).")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description='Adaline vision benchmark suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--dataset',   default='cifar10',
                   choices=['cifar10', 'imagenet'])
    p.add_argument('--benchmark', default='all',
                   choices=['classify', 'fewshot', 'incremental',
                            'openset', 'footprint', 'comparison', 'all'])
    p.add_argument('--backbone',  default='mobilenet_v2',
                   choices=list(BACKBONE_DIMS))
    p.add_argument('--data-dir',  default='./data',
                   help='Dataset root (for ImageNet: must contain train/ and val/)')
    p.add_argument('--cache-dir', default='./cache',
                   help='Directory for cached extracted features')
    p.add_argument('--n-train-per-class', type=int, default=20,
                   help='ImageNet only: training images sampled per class for prototypes')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    is_imagenet = args.dataset == 'imagenet'

    # ImageNet default backbone is resnet50
    backbone = args.backbone
    if is_imagenet and backbone == 'mobilenet_v2':
        backbone = 'resnet50'
        print('Note: defaulting to resnet50 for ImageNet (pass --backbone to override)')

    print(f'Adaline Vision Benchmark  —  {args.dataset.upper()} / {backbone}')
    print('=' * 60)

    # SRP needs a (10240 × dim) matrix: skip for ImageNet where dim=2048 (80 MB)
    # and it was the weakest encoder in CIFAR-10 tests anyway.
    if is_imagenet:
        encoders = [KWTAEncoder(k=128, probes=4), CountSketchEncoder(k=128)]
    else:
        encoders = [KWTAEncoder(k=128, probes=4), SRPEncoder(k=128), CountSketchEncoder(k=128)]

    feat_dim = BACKBONE_DIMS[backbone]
    bm = args.benchmark

    if bm in ('footprint', 'all'):
        bench_footprint(feat_dim, encoders)

    if bm == 'comparison' and not is_imagenet:
        print('Error: --benchmark comparison requires --dataset imagenet')
        sys.exit(1)

    if bm not in ('footprint',):
        if is_imagenet:
            (tr_feats, tr_labels), (te_feats, te_labels) = load_or_extract_imagenet(
                args.data_dir, backbone, args.cache_dir, args.n_train_per_class,
            )
        else:
            (tr_feats, tr_labels), (te_feats, te_labels) = load_or_extract_cifar10(
                args.data_dir, backbone, args.cache_dir,
            )
        print(f'\nTrain: {len(tr_feats):,} × {tr_feats.shape[1]}-dim  '
              f'Test: {len(te_feats):,} × {te_feats.shape[1]}-dim')

        # Dataset-specific incremental settings
        if is_imagenet:
            inc_steps  = [10, 50, 100, 200, 500, 1000]
            inc_top_ks = (1, 5)
            inc_nquery = 10   # 10 × 1000 = 10K queries at the final step
        else:
            inc_steps  = None   # defaults to range(2, 11, 2)
            inc_top_ks = (1,)
            inc_nquery = 50

        if bm in ('classify', 'all') and not is_imagenet:
            bench_classify(tr_feats, tr_labels, te_feats, te_labels,
                           encoders, seed=args.seed)
        if bm in ('fewshot', 'all') and not is_imagenet:
            bench_fewshot(tr_feats, tr_labels, te_feats, te_labels,
                          encoders, seed=args.seed)
        if bm in ('incremental', 'all'):
            bench_incremental(tr_feats, tr_labels, te_feats, te_labels,
                              encoders, n_query=inc_nquery,
                              steps=inc_steps, top_ks=inc_top_ks,
                              seed=args.seed)
        if bm in ('openset', 'all') and not is_imagenet:
            bench_openset(tr_feats, tr_labels, te_feats, te_labels,
                          encoders, seed=args.seed)
        if bm == 'comparison':
            bench_resnet_comparison(
                tr_feats, tr_labels, te_feats, te_labels,
                encoders, backbone,
                steps=inc_steps, n_query=inc_nquery, seed=args.seed,
            )

    print('\nDone.')


if __name__ == '__main__':
    main()

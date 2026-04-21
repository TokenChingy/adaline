#!/usr/bin/env python3
"""Dump CIFAR-10 CNN features to a raw binary file for Nim benchmarks.

Usage:
    python dump_features.py [--backbone mobilenet_v2] [--train-per-class 20] [--test-per-class 50]

Output:
    benchmarks/data/cifar10_features.bin

Binary format (little-endian):
    Header (32 bytes):
        magic       [4]  u8   -> b"ADLV"
        version     [1]  u32  -> 1
        n_train     [1]  u32
        n_test      [1]  u32
        feat_dim    [1]  u32
        n_classes   [1]  u32
        reserved    [12] u8
    Data:
        train_features [n_train * feat_dim] f32
        train_labels   [n_train]            u32
        test_features  [n_test * feat_dim]  f32
        test_labels    [n_test]             u32
"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from feature_extractor import load_or_extract_cifar10, BACKBONE_DIMS


def stratified_subset(feats: np.ndarray, labels: np.ndarray, n_per_class: int, rng: np.random.Generator):
    indices = []
    for c in np.unique(labels):
        cls_idx = np.where(labels == c)[0]
        chosen = rng.choice(cls_idx, size=min(n_per_class, len(cls_idx)), replace=False)
        indices.extend(chosen.tolist())
    return feats[indices], labels[indices]


def dump_binary(out_path: Path, tr_feats, tr_labels, te_feats, te_labels):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_train = len(tr_feats)
    n_test = len(te_feats)
    feat_dim = tr_feats.shape[1]
    n_classes = int(np.max(tr_labels)) + 1

    with open(out_path, "wb") as f:
        # Header
        f.write(struct.pack("<4sIIIIIII", b"ADLV", 1, n_train, n_test, feat_dim, n_classes, 0, 0))
        # Data
        f.write(tr_feats.astype(np.float32).tobytes())
        f.write(tr_labels.astype(np.uint32).tobytes())
        f.write(te_feats.astype(np.float32).tobytes())
        f.write(te_labels.astype(np.uint32).tobytes())

    print(f"Wrote {out_path}")
    print(f"  Train: {n_train} samples, {feat_dim} dims, {n_classes} classes")
    print(f"  Test:  {n_test} samples")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="mobilenet_v2", choices=list(BACKBONE_DIMS))
    p.add_argument("--train-per-class", type=int, default=100)
    p.add_argument("--test-per-class", type=int, default=50)
    p.add_argument("--out", default="benchmarks/data/cifar10_features.bin")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    (full_tr_feats, full_tr_labels), (full_te_feats, full_te_labels) = load_or_extract_cifar10(
        data_dir="./data", backbone=args.backbone, cache_dir="./cache"
    )

    rng = np.random.default_rng(args.seed)
    tr_feats, tr_labels = stratified_subset(full_tr_feats, full_tr_labels, args.train_per_class, rng)
    te_feats, te_labels = stratified_subset(full_te_feats, full_te_labels, args.test_per_class, rng)

    dump_binary(Path(args.out), tr_feats, tr_labels, te_feats, te_labels)


if __name__ == "__main__":
    main()

"""
CNN backbone feature extractor for Adaline vision benchmarks.

Supports MobileNetV2 (1280-dim, default), EfficientNet-B0 (1280-dim),
and ResNet-50 (2048-dim). All outputs are L2-normalised float32 vectors.

ImageNet normalisation (mean/std) MUST be applied — the BatchNorm layers
in pretrained models receive wrong-scale inputs without it and produce
near-random features.
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from pathlib import Path

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

BACKBONE_DIMS = {
    'mobilenet_v2':   1280,
    'efficientnet_b0': 1280,
    'resnet50':       2048,
}


def build_transform() -> T.Compose:
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class FeatureExtractor:
    def __init__(self, backbone: str = 'mobilenet_v2'):
        if backbone not in BACKBONE_DIMS:
            raise ValueError(f"Unknown backbone {backbone!r}. Choose from: {list(BACKBONE_DIMS)}")
        self.backbone_name = backbone
        self.dim = BACKBONE_DIMS[backbone]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._build_model(backbone)

    def _build_model(self, backbone: str):
        if backbone == 'mobilenet_v2':
            m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
            self.model = torch.nn.Sequential(m.features, torch.nn.AdaptiveAvgPool2d(1))
        elif backbone == 'efficientnet_b0':
            m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            self.model = torch.nn.Sequential(m.features, torch.nn.AdaptiveAvgPool2d(1))
        elif backbone == 'resnet50':
            m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.model = torch.nn.Sequential(*list(m.children())[:-1])
        self.model.eval().to(self.device)

    @torch.no_grad()
    def extract_batch(self, images: torch.Tensor) -> np.ndarray:
        feats = self.model(images.to(self.device)).squeeze(-1).squeeze(-1)
        feats = F.normalize(feats, p=2, dim=-1)
        return feats.cpu().float().numpy()

    def extract_dataset(self, dataset, batch_size: int = 256) -> tuple:
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=2, pin_memory=True,
        )
        all_feats, all_labels = [], []
        seen = 0
        for images, labels in loader:
            all_feats.append(self.extract_batch(images))
            all_labels.append(labels.numpy())
            seen += len(labels)
            print(f"\r  {seen}/{len(dataset)}", end='', flush=True)
        print()
        return np.vstack(all_feats), np.concatenate(all_labels)


def load_or_extract_cifar10(
    data_dir: str = './data',
    backbone: str = 'mobilenet_v2',
    cache_dir: str = './cache',
) -> tuple:
    """Returns ((train_feats, train_labels), (test_feats, test_labels)).

    Features are L2-normalised float32, shape (N, dim).
    Extracted features are cached to disk so the first run is the only slow one.
    """
    cache = Path(cache_dir)
    cache.mkdir(exist_ok=True)
    tr_path = cache / f'cifar10_{backbone}_train.npz'
    te_path = cache / f'cifar10_{backbone}_test.npz'

    if tr_path.exists() and te_path.exists():
        print(f"Loading cached {backbone} features from {cache_dir}/")
        tr = np.load(tr_path)
        te = np.load(te_path)
        return (tr['feats'], tr['labels']), (te['feats'], te['labels'])

    print(f"Extracting {backbone} features (first run — will be cached)...")
    transform = build_transform()
    extractor = FeatureExtractor(backbone)
    train_ds = CIFAR10(data_dir, train=True,  download=True, transform=transform)
    test_ds  = CIFAR10(data_dir, train=False, download=True, transform=transform)

    print("  Train set:")
    tr_feats, tr_labels = extractor.extract_dataset(train_ds)
    print("  Test set:")
    te_feats, te_labels = extractor.extract_dataset(test_ds)

    np.savez(tr_path, feats=tr_feats, labels=tr_labels)
    np.savez(te_path, feats=te_feats, labels=te_labels)
    print(f"Cached to {cache_dir}/")
    return (tr_feats, tr_labels), (te_feats, te_labels)


# ─── ImageNet ────────────────────────────────────────────────────────────────

_HF_DATASET  = "ILSVRC/imagenet-1k"
_DONE_MARKER = ".download_complete"

# Instructions printed when HuggingFace auth is missing.
_HF_SETUP = """
ImageNet auto-download requires a free HuggingFace account with dataset access:

  1. Create an account at https://huggingface.co
  2. Accept the dataset terms at https://huggingface.co/datasets/ILSVRC/imagenet-1k
  3. Install the library and log in:
       pip install datasets
       huggingface-cli login        (paste your access token when prompted)
  4. Re-run this command — the download will start automatically.

Alternatively, download manually from https://image-net.org/download.php and
place the extracted files as:
    {root}/train/<synset_id>/  and  {root}/val/<synset_id>/
then pass --data-dir {root} to the benchmark.
"""


def _find_imagenet_root(data_dir: str) -> str | None:
    """Return the imagenet root if it looks valid, else None.

    Accepts the root directly (data_dir/val/) or one level up
    (data_dir/imagenet/val/) so both --data-dir /path/to/imagenet and
    --data-dir /path/to/data work without extra flags.
    """
    for candidate in [data_dir, os.path.join(data_dir, 'imagenet')]:
        if os.path.isdir(os.path.join(candidate, 'val')):
            return candidate
    return None


def _load_imagenet_split(imagenet_root: str, split: str,
                         transform) -> torch.utils.data.Dataset:
    """Load an ImageNet split from disk.

    Tries torchvision.datasets.ImageNet first (synset layout), then falls
    back to ImageFolder (works with our zero-padded integer class dirs).
    """
    from torchvision.datasets import ImageNet, ImageFolder
    try:
        return ImageNet(imagenet_root, split=split, transform=transform)
    except Exception:
        pass
    split_dir = os.path.join(imagenet_root, split)
    if os.path.isdir(split_dir):
        return ImageFolder(split_dir, transform=transform)
    raise FileNotFoundError(
        f"Could not load split '{split}' from {imagenet_root!r}.\n"
        f"Expected {imagenet_root}/{split}/<class_dirs>/"
    )


def _make_stratified_subset(dataset, n_per_class: int,
                             seed: int = 42) -> torch.utils.data.Subset:
    """Sample n_per_class images per class, returning a Subset."""
    rng = np.random.default_rng(seed)
    targets = np.array(dataset.targets)
    indices = []
    for c in np.unique(targets):
        cls_idx = np.where(targets == c)[0]
        chosen = rng.choice(cls_idx, size=min(n_per_class, len(cls_idx)), replace=False)
        indices.extend(chosen.tolist())
    return torch.utils.data.Subset(dataset, sorted(indices))


def download_imagenet_hf(target_dir: str, n_train_per_class: int = 20) -> str:
    """Stream ImageNet from HuggingFace into target_dir/.

    Only downloads what the benchmark needs:
      • Training: n_train_per_class images per class  (~3–4 GB for n=20)
      • Validation: all 50,000 images                 (~6.5 GB)
    versus the full training set which is ~155 GB.

    Images are saved as:
      target_dir/train/<NNNN>/<MMMM>.jpg   (zero-padded class and image index)
      target_dir/val/<NNNN>/<MMMM>.jpg

    Zero-padded directory names keep lexicographic == numeric order so that
    torchvision.datasets.ImageFolder assigns labels 0–999 correctly.

    Requires:
        pip install datasets
        huggingface-cli login
        Accept terms at https://huggingface.co/datasets/ILSVRC/imagenet-1k
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "HuggingFace `datasets` library not installed.\n"
            "Run: pip install datasets" + _HF_SETUP.format(root=target_dir)
        )

    out = Path(target_dir)
    done = out / _DONE_MARKER
    if done.exists():
        print(f"ImageNet already downloaded at {out}")
        return str(out)

    out.mkdir(parents=True, exist_ok=True)

    # ── Training subset ──────────────────────────────────────────────────────
    total_needed = 1000 * n_train_per_class
    print(f"Streaming ImageNet train ({n_train_per_class} images/class = "
          f"{total_needed:,} total).  This downloads ~3-4 GB.")

    try:
        ds_train = load_dataset(_HF_DATASET, split="train",
                                streaming=True, trust_remote_code=True)
    except Exception as e:
        raise SystemExit(_HF_SETUP.format(root=target_dir) +
                         f"\nOriginal error: {e}")

    counts: dict[int, int] = {}
    for item in ds_train:
        label: int = item['label']
        if counts.get(label, 0) >= n_train_per_class:
            continue
        class_dir = out / 'train' / f'{label:04d}'
        class_dir.mkdir(parents=True, exist_ok=True)
        img = item['image']
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(class_dir / f'{counts.get(label, 0):04d}.jpg', quality=95)
        counts[label] = counts.get(label, 0) + 1
        saved = sum(counts.values())
        if saved % 1000 == 0:
            print(f"\r  {saved:>6}/{total_needed} images, "
                  f"{len(counts):>4}/1000 classes", end='', flush=True)
        if len(counts) == 1000 and all(v >= n_train_per_class for v in counts.values()):
            break
    print(f"\r  {sum(counts.values()):>6}/{total_needed} training images saved   ")

    # ── Validation set ───────────────────────────────────────────────────────
    print("Streaming ImageNet val (50,000 images).  This downloads ~6.5 GB.")
    ds_val = load_dataset(_HF_DATASET, split="validation",
                          streaming=True, trust_remote_code=True)
    val_counts: dict[int, int] = {}
    for item in ds_val:
        label = item['label']
        class_dir = out / 'val' / f'{label:04d}'
        class_dir.mkdir(parents=True, exist_ok=True)
        img = item['image']
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(class_dir / f'{val_counts.get(label, 0):04d}.jpg', quality=95)
        val_counts[label] = val_counts.get(label, 0) + 1
        total = sum(val_counts.values())
        if total % 5000 == 0:
            print(f"\r  {total:>6}/50000 images", end='', flush=True)
    print(f"\r  {sum(val_counts.values()):>6}/50000 val images saved   ")

    done.touch()
    print(f"Download complete → {out}")
    return str(out)


def load_or_extract_imagenet(
    data_dir: str,
    backbone: str = 'resnet50',
    cache_dir: str = './cache',
    n_train_per_class: int = 20,
) -> tuple:
    """Returns ((train_feats, train_labels), (val_feats, val_labels)).

    On the first call, images are downloaded from HuggingFace if not already
    present, then features are extracted and cached.  Subsequent calls load
    directly from the feature cache.

    data_dir behaviour:
      • If data_dir/val/ exists → treated as the ImageNet root directly.
      • If data_dir/imagenet/val/ exists → uses data_dir/imagenet/.
      • Otherwise → downloads to data_dir/imagenet/ automatically.
    """
    # ── Locate or download images ────────────────────────────────────────────
    imagenet_root = _find_imagenet_root(data_dir)
    if imagenet_root is None:
        target = os.path.join(data_dir, 'imagenet')
        print(f"ImageNet not found under {data_dir!r} — starting auto-download.")
        imagenet_root = download_imagenet_hf(target, n_train_per_class)

    # ── Feature extraction (with disk cache) ─────────────────────────────────
    cache = Path(cache_dir)
    cache.mkdir(exist_ok=True)
    tr_path  = cache / f'imagenet_{backbone}_train_n{n_train_per_class}.npz'
    val_path = cache / f'imagenet_{backbone}_val.npz'

    transform = build_transform()
    extractor = FeatureExtractor(backbone)

    if tr_path.exists():
        print(f"Loading cached ImageNet train features ({n_train_per_class}/class)...")
        tr = np.load(tr_path)
        tr_feats, tr_labels = tr['feats'], tr['labels']
    else:
        print(f"Extracting ImageNet train features ({n_train_per_class}/class)...")
        train_ds = _load_imagenet_split(imagenet_root, 'train', transform)
        subset   = _make_stratified_subset(train_ds, n_train_per_class)
        print(f"  Sampled {len(subset):,} training images:")
        tr_feats, tr_labels = extractor.extract_dataset(subset)
        np.savez(tr_path, feats=tr_feats, labels=tr_labels)
        print(f"  Cached to {tr_path}")

    if val_path.exists():
        print(f"Loading cached ImageNet val features...")
        val = np.load(val_path)
        val_feats, val_labels = val['feats'], val['labels']
    else:
        print(f"Extracting ImageNet val features (50,000 images)...")
        val_ds = _load_imagenet_split(imagenet_root, 'val', transform)
        print(f"  Val set:")
        val_feats, val_labels = extractor.extract_dataset(val_ds)
        np.savez(val_path, feats=val_feats, labels=val_labels)
        print(f"  Cached to {val_path}")

    return (tr_feats, tr_labels), (val_feats, val_labels)

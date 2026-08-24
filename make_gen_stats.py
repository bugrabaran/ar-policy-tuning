import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models import inception_v3, Inception_V3_Weights


class ImageFolderFlat(Dataset):
    def __init__(self, root: str, transform=None, exts=None):
        self.root = Path(root)
        self.transform = transform
        if exts is None:
            exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        self.paths: List[Path] = []
        for p in self.root.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                self.paths.append(p)
        if len(self.paths) == 0:
            raise RuntimeError(f"No image files found in {root}")
        self.paths.sort()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img


def build_inception(weights_name: str, device: str):
    try:
        weights = getattr(Inception_V3_Weights, weights_name)
    except AttributeError:
        raise ValueError(f"Unknown weights '{weights_name}'. Try IMAGENET1K_V1.")

    model = inception_v3(weights=weights, aux_logits=True, transform_input=False)
    model.eval()

    feat_dim = 2048
    captured = {}

    def hook_fn(_, __, output):
        # output: [N, 2048, 1, 1] -> flatten to [N, 2048]
        captured["feat"] = torch.flatten(output, 1)

    handle = model.avgpool.register_forward_hook(hook_fn)

    preprocess = weights.transforms()

    dev = torch.device(device)
    model.to(dev)

    return model, preprocess, handle, feat_dim, dev


@torch.no_grad()
def compute_stats(
    loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    feat_dim: int,
    stats_mode: str = "both",
    use_float64_accum: bool = True,
):
    """
    stats_mode: 'both' | 'full' | 'diagonal'
    """
    assert stats_mode in {"both", "full", "diagonal"}

    dtype = torch.float64 if use_float64_accum else torch.float32

    count = 0
    sum_x = torch.zeros(feat_dim, dtype=dtype)
    sum_xxT = None
    sum_x2 = torch.zeros(feat_dim, dtype=dtype)  # for diagonal/variance

    if stats_mode in {"both", "full"}:
        sum_xxT = torch.zeros((feat_dim, feat_dim), dtype=dtype)

    for imgs in loader:
        imgs = imgs.to(device, non_blocking=True)
        captured = {}
        def local_hook(_, __, output):
            captured["feat"] = torch.flatten(output, 1)

        h = model.avgpool.register_forward_hook(local_hook)
        _ = model(imgs)
        h.remove()

        f = captured["feat"]  # [B, D]
        f64 = f.detach().to("cpu", dtype=dtype)

        b = f64.shape[0]
        count += b

        sum_x += f64.sum(dim=0)
        sum_x2 += (f64 * f64).sum(dim=0)

        if sum_xxT is not None:
            # sum of outer-products: (D,B) @ (B,D) = (D,D)
            sum_xxT += f64.t().mm(f64)

    if count <= 1:
        raise RuntimeError("Need at least 2 images to compute an unbiased covariance.")

    mu = sum_x / count  # [D]

    # Variance (diagonal, unbiased)
    var = (sum_x2 - count * (mu * mu)) / (count - 1)  # [D]

    # Full covariance (unbiased)
    sigma = None
    if sum_xxT is not None:
        # Sigma = (sum xx^T - N * mu mu^T) / (N - 1)
        mu_outer = torch.outer(mu, mu)
        sigma = (sum_xxT - count * mu_outer) / (count - 1)  # [D,D]

    return mu.numpy(), (sigma.numpy() if sigma is not None else None), var.numpy(), count


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", type=str, required=True,
                   help="Directory containing generated images (searched recursively).")
    p.add_argument("--out", type=str, required=True, help="Output .npz path.")
    p.add_argument("--weights", type=str, default="IMAGENET1K_V1",
                   help="Torchvision Inception_V3 weights enum name.")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda",
                   help="'cuda' or 'cpu'.")
    p.add_argument("--stats", type=str, default="both",
                   choices=["both", "full", "diagonal"],
                   help="Which statistics to compute/store.")
    p.add_argument("--pin-memory", action="store_true", help="Pin dataloader memory.")
    args = p.parse_args()

    model, preprocess, hook_handle, feat_dim, dev = build_inception(args.weights, args.device)

    ds = ImageFolderFlat(args.images, transform=preprocess)
    dl = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and (args.device == "cuda"),
        drop_last=False,
    )

    mu, sigma, var, count = compute_stats(
        loader=dl,
        model=model,
        device=dev,
        feat_dim=feat_dim,
        stats_mode=args.stats,
        use_float64_accum=True,
    )

    payload = {
        "mu": mu.astype(np.float64),
        "count": np.array([count], dtype=np.int64),
        "dims": np.array([feat_dim], dtype=np.int64),
        "sigma_diagonal": var.astype(np.float64),
    }
    if sigma is not None:
        payload["sigma"] = sigma.astype(np.float64)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **payload)
    print(f"Wrote: {out_path} (count={count}, dims={feat_dim}, "
          f"keys={list(payload.keys())})")

    hook_handle.remove()


if __name__ == "__main__":
    main()

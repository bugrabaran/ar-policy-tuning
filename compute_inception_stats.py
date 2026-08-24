#!/usr/bin/env python3
"""
Compute Inception-v3 (pool features, 2048-d) statistics (mu, sigma) for FID.

- ImageFolder layout (class subfolders)
- ToTensor() -> [0,1] float32, no normalization
- Bilinear resize to 299x299 in the DataLoader collate_fn (PyTorch tensor path)
- Inception v3 with transform_input=False, fc=Identity()
- Accumulate in float64: sum_x, sum_xxT (stable), then build covariance
- Save NPZ with 'mu' (D,), 'sigma' (D,D), plus a small 'meta' dict

Offline friendly:
- Pass --weights-path /path/to/inception_v3_google-0cc3c7bd.pth
- If not provided, torchvision downloads the official IMAGENET1K_V1 weights.

Usage:
  python compute_inception_stats.py \
    --data-root /path/to/imagenet/train \
    --out /path/to/imagenet_train_incv3_299.npz \
    --weights-path /models/inception_v3_google-0cc3c7bd.pth \
    --batch-size 256 --num-workers 8 --cov-estimator unbiased
"""

import argparse
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm


class Identity(nn.Module):
    def forward(self, x):
        return x


def load_inception_v3(device: torch.device,
                      weights_path: Optional[str],
                      verbose: bool = True) -> nn.Module:
    """
    Load Inception-v3. Prefer local weights and otherwise use torchvision weights.
    We keep transform_input=False and replace fc with Identity().
    """
    model = torchvision.models.inception_v3(weights=None, transform_input=False, init_weights=False)
    ok = False

    if weights_path:
        try:
            if verbose:
                print(f"[Inception] Loading local weights: {weights_path}")
            sd = torch.load(weights_path, map_location="cpu", weights_only=False)
            model.load_state_dict(sd, strict=True)
            ok = True
        except Exception as e:
            print(f"[Inception] Failed to load local weights: {e}")

    if not ok:
        try:
            if verbose:
                print("[Inception] Trying torchvision IMAGENET1K_V1 weights…")
            weights = torchvision.models.Inception_V3_Weights.IMAGENET1K_V1
            model = torchvision.models.inception_v3(weights=weights, transform_input=False)
            ok = True
        except Exception as e:
            raise RuntimeError("Could not load pretrained Inception-v3 weights") from e

    model.fc = Identity()
    model.eval().to(device)
    return model


def collate_resize_bilinear(batch, size: int = 299):
    """
    Collate function that resizes variable-sized tensors to [3,size,size] using
    torch bilinear (align_corners=False) to match training/reward pipeline.
    batch: list of (img_tensor [3,H,W], label)
    """
    imgs, labels = zip(*batch)
    out = []
    for x in imgs:
        if x.dtype != torch.float32:
            x = x.float()
        if x.ndim != 3 or x.shape[0] != 3:
            if x.ndim == 2:
                x = x.unsqueeze(0).repeat(3, 1, 1)
            elif x.shape[0] == 1:
                x = x.repeat(3, 1, 1)
        if x.max() > 1.01:  # if someone fed uint8 by mistake
            x = x / 255.0
        x = F.interpolate(x.unsqueeze(0), size=(size, size),
                          mode="bilinear", align_corners=False).squeeze(0)
        out.append(x)
    return torch.stack(out, 0), torch.as_tensor(labels)


def make_loader(data_root: str,
                batch_size: int,
                num_workers: int,
                shuffle: bool,
                drop_last: bool,
                resize_hw: int) -> DataLoader:
    """
    Dataset: ImageFolder with RGB + ToTensor()
    Resize happens in collate_fn with PyTorch bilinear to ensure parity with reward pipeline.
    """
    tfm = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),  # -> [0,1] float32, keep native HxW
    ])
    ds = datasets.ImageFolder(root=data_root, transform=tfm)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=(num_workers > 0),
        collate_fn=lambda b: collate_resize_bilinear(b, size=resize_hw),
    )
    return loader


@torch.inference_mode()
def extract_batch_features(x_bchw: torch.Tensor,
                           model: nn.Module,
                           resize_hw: int = 299) -> torch.Tensor:
    """
    x_bchw: [B,3,H,W] float32 in [0,1].
    Since we already resize in collate_fn, this guard avoids double-resizing.
    Returns [B,2048] float32 features.
    """
    if x_bchw.dtype != torch.float32:
        x_bchw = x_bchw.float()
        if x_bchw.max() > 1.01:
            x_bchw = x_bchw / 255.0
    if x_bchw.shape[-2:] != (resize_hw, resize_hw):
        x_bchw = F.interpolate(x_bchw, size=(resize_hw, resize_hw),
                               mode="bilinear", align_corners=False)
    feats = model(x_bchw)  # fc=Identity() → [B,2048]
    return feats.float()


def compute_stats(loader: DataLoader,
                  device: torch.device,
                  resize_hw: int = 299,
                  cov_estimator: str = "unbiased",
                  weights_path: Optional[str] = None,
                  pca_d: int = 0, pca_max_samples: int = 50000, pca_seed: int = 0):
    """
    Accumulate mu and covariance in float64 on device, then move to CPU at the end.
    cov_estimator: 'unbiased' => /(N-1); 'mle' => /N
    """
    inc = load_inception_v3(device=device, weights_path=weights_path)
    D = 2048
    sum_x = torch.zeros(D, dtype=torch.float64, device=device)
    sum_xxT = torch.zeros(D, D, dtype=torch.float64, device=device)
    total = 0

    rng = np.random.default_rng(pca_seed)
    pca_buf = None
    pca_count = 0
    if pca_d and pca_d > 0:
        pca_buf = np.empty((pca_max_samples, 2048), dtype=np.float32)



    pbar = tqdm(loader, desc="Extracting features", dynamic_ncols=True)
    for images, _ in pbar:
        images = images.to(device, non_blocking=True)     # [B,3,299,299] float32 in [0,1]
        feats = extract_batch_features(images, inc, resize_hw=resize_hw)  # [B,2048]
        f64 = feats.to(torch.float64)
        if pca_buf is not None:
            feats_cpu = feats.detach().cpu().numpy().astype(np.float32)  # [B,2048]
            for row in feats_cpu:
                if pca_count < pca_max_samples:
                    pca_buf[pca_count] = row
                else:
                    j = rng.integers(0, pca_count + 1)
                    if j < pca_max_samples:
                        pca_buf[j] = row
                pca_count += 1

        sum_x += f64.sum(dim=0)                 # [D]
        sum_xxT += f64.t().matmul(f64)          # [D,D]
        total += feats.size(0)

        if total % 4096 == 0:
            pbar.set_postfix(samples=total)

    if total == 0:
        raise RuntimeError("No images found. Check --data-root path and contents.")

    mu = (sum_x / total).cpu().numpy()          # [D], float64
    ExxT = (sum_xxT / total).cpu().numpy()      # [D,D], float64
    sigma = ExxT - np.outer(mu, mu)             # population covariance (/N)

    if cov_estimator == "unbiased":
        sigma *= (total / max(total - 1, 1))    # sample covariance (/N-1)
    elif cov_estimator == "mle":
        pass
    else:
        raise ValueError("--cov-estimator must be 'unbiased' or 'mle'.")

    return mu, sigma, total, (pca_buf[:min(pca_count, pca_max_samples)] if pca_buf is not None else None)



def main():
    ap = argparse.ArgumentParser("Compute Inception-v3 feature stats (mu, sigma) for FID")
    ap.add_argument("--pca-d", type=int, default=0,
                help="If >0, also compute PCA projection and PCA-space real covariance.")
    ap.add_argument("--pca-max-samples", type=int, default=50000,
                    help="Max number of real features to keep for PCA fitting (reservoir).")
    ap.add_argument("--pca-seed", type=int, default=0)
    ap.add_argument("--out-pca", type=str, default="",
                    help="Optional separate NPZ path for PCA stats. If empty, appends _pca{d}.npz")

    ap.add_argument("--data-root", type=str, required=True,
                    help="Folder with class subfolders (ImageFolder). e.g., /imagenet/train")
    ap.add_argument("--out", type=str, required=True, help="Output NPZ path")
    ap.add_argument("--weights-path", type=str, default="",
                    help="Local inception_v3 .pth to avoid downloads "
                         "(e.g., inception_v3_google-0cc3c7bd.pth)")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--resize", type=int, default=299)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--cov-estimator", type=str, default="unbiased",
                    choices=["unbiased", "mle"])
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    device = torch.device(args.device)
    loader = make_loader(args.data_root, args.batch_size, args.num_workers,
                         shuffle=args.shuffle, drop_last=False, resize_hw=args.resize)

    print(f"Using device: {device}")
    print(f"Dataset root: {args.data_root}")
    print(f"Batches: {len(loader)} (batch_size={args.batch_size})")
    print(f"Resize: {args.resize} | Cov estimator: {args.cov_estimator}")
    if args.weights_path:
        print(f"Local weights: {args.weights_path}")

    mu, sigma, total, Xpca = compute_stats(loader, device, resize_hw=args.resize,
                                     cov_estimator=args.cov_estimator,
                                     weights_path=(args.weights_path or None),
                                     pca_d=args.pca_d, pca_max_samples=args.pca_max_samples, pca_seed=args.pca_seed)

    np.savez(
        args.out,
        mu=mu.astype(np.float64),
        sigma=sigma.astype(np.float64),
        sigma_diagonal=np.diag(sigma).astype(np.float64),
        count=np.array([total], dtype=np.int64),
        dims=np.array([mu.shape[0]], dtype=np.int64),
        meta=dict(
            model="torchvision.inception_v3",
            weights="IMAGENET1K_V1",
            transform_input=False,
            input_range="[0,1]",
            normalize=None,
            resize=int(args.resize),
            resize_mode="bilinear_align_corners=False",
            cov_estimator=args.cov_estimator,
        ),
    )
    print(f"Saved Inception statistics: {args.out}")

    if args.pca_d <= 0:
        return

    assert Xpca is not None and Xpca.shape[0] >= args.pca_d + 1, "Not enough PCA samples."

    X = Xpca.astype(np.float64) - mu[None, :]

    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    W = Vt[:args.pca_d].T

    Z = X @ W
    mu_pca = Z.mean(axis=0)

    Zc = Z - mu_pca[None, :]
    if args.cov_estimator == "unbiased":
        denom = max(Z.shape[0] - 1, 1)
    else:
        denom = Z.shape[0]
    sigma_pca = (Zc.T @ Zc) / denom


    out_pca = args.out_pca
    if not out_pca:
        base, ext = os.path.splitext(args.out)
        out_pca = f"{base}_pca{args.pca_d}{ext}"

    np.savez(out_pca,
            mu_real=mu.astype(np.float64),
            W=W.astype(np.float64),
            mu_real_pca=mu_pca.astype(np.float64),
            sigma_real_pca=sigma_pca.astype(np.float64),
            meta=dict(
                pca_d=int(args.pca_d),
                pca_max_samples=int(min(Xpca.shape[0], args.pca_max_samples)),
                pca_seed=int(args.pca_seed),
                cov_estimator=args.cov_estimator,
                base_stats_npz=args.out,
                note="PCA fit on real Inception pool3 features (centered by mu_real)."
            ))
    print(f"Saved PCA stats: {out_pca}")



if __name__ == "__main__":
    main()

import torch
import torch.nn.functional as F
import numpy as np
import torchvision
import torch.nn as nn
from typing import Optional, Tuple
import torch.distributed as dist

try:
    import open_clip
    _HAS_CLIP = True
except Exception:
    _HAS_CLIP = False

try:
    import hpsv2
    _HAS_HPSV2 = True
    from PIL import Image
    from collections import defaultdict
except Exception:
    _HAS_HPSV2 = False

class CLIPReward:
    def __init__(self, device, model_name="ViT-L-14", pretrained="openai"):
        self.device = device
        if not _HAS_CLIP:
            raise RuntimeError("OpenCLIP is required when --w-clip is non-zero. Install open_clip_torch.")
        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
        self.model = model.eval()
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1,3,1,1)
        std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1,3,1,1)
        self.register_mean, self.register_std = mean, std
        self.target_hw = 224

    @torch.no_grad()
    def __call__(self, images_uint8, prompts_or_tok) -> torch.Tensor:
        if self.model is None:
            return torch.zeros(images_uint8.size(0), device=images_uint8.device, dtype=torch.float32)

        x = images_uint8.permute(0,3,1,2).to(self.device, dtype=torch.float32) / 255.0
        x = F.interpolate(x, size=(self.target_hw, self.target_hw), mode="bilinear", align_corners=False)
        x = (x - self.register_mean) / self.register_std

        if isinstance(prompts_or_tok, torch.Tensor):
            tok = prompts_or_tok.to(self.device, non_blocking=True)
        else:
            tok = open_clip.tokenize(prompts_or_tok).to(self.device, non_blocking=True)

        img_feat = self.model.encode_image(x)
        txt_feat = self.model.encode_text(tok)
        img_feat = img_feat / (img_feat.norm(dim=-1, keepdim=True) + 1e-6)
        txt_feat = txt_feat / (txt_feat.norm(dim=-1, keepdim=True) + 1e-6)
        sim = (img_feat * txt_feat).sum(dim=-1).float()
        return torch.nan_to_num(sim, nan=0.0, posinf=0.0, neginf=0.0)





class HPSv2Reward:
    def __init__(self, device, version="v2.1"):
        self.device = device
        self.version = version
        if not _HAS_HPSV2:
            raise RuntimeError("HPSv2 is required when --w-hps is non-zero. Install hpsv2.")
        # hpsv2 1.2.0 wheels can omit the vendored CLIP BPE vocabulary.
        # Its tokenizer API is compatible with open_clip_torch, which bundles
        # the same vocabulary, so route the vendored import there when needed.
        from pathlib import Path
        hps_bpe = Path(hpsv2.__file__).resolve().parent / "src" / "open_clip" / "bpe_simple_vocab_16e6.txt.gz"
        if not hps_bpe.is_file():
            if not _HAS_CLIP:
                raise RuntimeError(
                    "The installed hpsv2 package is missing its BPE vocabulary; "
                    "install open_clip_torch to provide the compatible tokenizer fallback."
                )
            import sys
            import open_clip.tokenizer as open_clip_tokenizer
            sys.modules.setdefault("hpsv2.src.open_clip.tokenizer", open_clip_tokenizer)
            print("[warn] hpsv2 BPE vocabulary missing; using the open_clip tokenizer")
        self.enabled = True

    @torch.no_grad()
    def __call__(self, images_uint8, prompts: list[str]) -> torch.Tensor:
        if not self.enabled:
            return torch.zeros(images_uint8.size(0), device=images_uint8.device, dtype=torch.float32)

        B = images_uint8.size(0)
        assert len(prompts) == B, f"len(prompts) ({len(prompts)}) != batch size ({B})"

        groups = defaultdict(list)  # prompt -> list[(idx, np_img)]
        for idx, (img, p) in enumerate(zip(images_uint8, prompts)):
            groups[p].append((idx, img.detach().cpu().numpy()))

        out = torch.empty(B, dtype=torch.float32, device=images_uint8.device)

        for ptxt, items in groups.items():
            pil_list = [Image.fromarray(arr) for _, arr in items]
            arg = pil_list[0] if len(pil_list) == 1 else pil_list
            raw = hpsv2.score(arg, ptxt, self.version)
            scores = _to_float_list(raw)

            if len(scores) == 1 and len(items) > 1:
                scores = scores * len(items)
            if len(scores) != len(items):
                raise ValueError(f"hpsv2 returned {len(scores)} scores for {len(items)} images (prompt={ptxt!r})")

            for (idx, _), val in zip(items, scores):
                v = float(val)
                if not np.isfinite(v):
                    v = 0.0
                out[idx] = v

        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

class CompositeReward:
    def __init__(self, device, w_clip=1.0, w_hps=0.0):
        self.wc, self.wh= float(w_clip), float(w_hps)
        self.clip = CLIPReward(device) if self.wc != 0.0 else None
        self.hps = HPSv2Reward(device) if self.wh != 0.0 else None

    @torch.no_grad()
    def __call__(self, images_uint8, prompts):
        out = torch.zeros(images_uint8.size(0), device=images_uint8.device, dtype=torch.float32)
        if self.wc != 0.0:
            assert self.clip is not None
            r = self.clip(images_uint8, prompts)
            out = out + self.wc * torch.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
        if self.wh != 0.0:
            assert self.hps is not None
            r = self.hps(images_uint8, prompts)
            out = out + self.wh * torch.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _to_float_list(s):
    """
    Normalize hpsv2.score(...) outputs to a flat python list[float].
    Handles: Python scalars, numpy scalars/arrays, torch tensors, lists/tuples of any of these.
    """
    if isinstance(s, (int, float)):
        return [float(s)]
    if isinstance(s, np.generic):  # numpy scalar (e.g., np.float16/32/64)
        return [float(s)]
    if torch.is_tensor(s):
        return [float(x) for x in s.reshape(-1).detach().cpu().numpy().tolist()]
    if isinstance(s, np.ndarray):
        return [float(x) for x in s.reshape(-1).tolist()]
    if isinstance(s, (list, tuple)):
        out = []
        for x in s:
            if isinstance(x, (int, float)):
                out.append(float(x))
            elif isinstance(x, np.generic):
                out.append(float(x))
            elif torch.is_tensor(x):
                out.append(float(x.item()))
            elif isinstance(x, np.ndarray):
                out.extend([float(y) for y in x.reshape(-1).tolist()])
            else:
                raise TypeError(f"Unexpected element type in scores list: {type(x)}")
        return out
    raise TypeError(f"Unexpected return type from hpsv2.score: {type(s)}")

class InceptionPool3(nn.Module):
    def __init__(self, device: torch.device, weights_name: Optional[str] = "IMAGENET1K_V1", resize_hw: int = 299):
        super().__init__()
        self.resize_hw = int(resize_hw)  # <— store once

        weights = None
        if weights_name is not None:
            try:
                weights = getattr(torchvision.models.Inception_V3_Weights, weights_name)
            except Exception:
                try:
                    weights = torchvision.models.Inception_V3_Weights.DEFAULT
                except Exception:
                    weights = None

        inc = torchvision.models.inception_v3(
            weights=weights,
            transform_input=False,
        ).to(device)
        inc.eval()
        for p in inc.parameters():
            p.requires_grad = False

        self._feat = None
        def _hook(_m, _i, o):
            self._feat = o
        inc.avgpool.register_forward_hook(_hook)
        self.inc = inc

        self.normalize_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
        self.normalize_std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] != self.resize_hw or x.shape[-1] != self.resize_hw:
            x = F.interpolate(x, size=(self.resize_hw, self.resize_hw), mode="bilinear",
                              align_corners=False, antialias=True)
        x = (x - self.normalize_mean) / self.normalize_std
        _ = self.inc(x)
        feats = torch.flatten(self._feat, 1)  # [N,2048]
        return feats



@torch.no_grad()
def diag_fid_mle(mu_r: torch.Tensor, std_r: torch.Tensor,
                 mu_g: torch.Tensor, std_g: torch.Tensor) -> torch.Tensor:
    """
    Diagonal FID (MLE) = ||mu_r - mu_g||^2 + ||std_r - std_g||^2

    If inputs are [D], returns a scalar.
    If mu_g/std_g are [N,D] (batched), returns [N] with per-row distances.
    """
    diff_mu = (mu_r - mu_g)
    diff_sd = (std_r - std_g)
    sq = diff_mu * diff_mu + diff_sd * diff_sd
    return sq.sum(dim=-1) if sq.ndim > 1 else sq.sum()



@torch.no_grad()
def diag_fid_batch(mu_r: torch.Tensor, std_r: torch.Tensor,
                   feats: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    FID_diag on a batch's MLE moments.
    feats: [N,D]
    Returns scalar tensor.
    """
    mu = feats.mean(0)
    var = feats.var(0, unbiased=False)
    std = torch.sqrt(torch.clamp(var, min=0.0) + eps)
    return diag_fid_mle(mu_r, std_r, mu, std)


@torch.no_grad()
def diag_fid_loo_contrib(mu_r: torch.Tensor, std_r: torch.Tensor,
                         feats: torch.Tensor, eps: float = 1e-12) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized LOO contributions for diagonal FID with MLE variance.
    Returns:
      fid_full : scalar (batch FID)
      contribs : [N] where contribs[i] = FID(X_r, X_g \ {x_i}) - FID(X_r, X_g)
                 Positive => x_i is helpful (removing it worsens FID).
    """
    assert feats.ndim == 2
    N, D = feats.shape
    device = feats.device

    if N < 2:
        mu = feats.mean(0)
        std = torch.sqrt(torch.clamp(feats.var(0, unbiased=False), min=0.0) + eps)
        fid = diag_fid_mle(mu_r, std_r, mu, std)
        return fid, torch.zeros(N, device=device)

    # Full-batch FID
    fid_full = diag_fid_batch(mu_r, std_r, feats, eps=eps)

    # Raw sums
    S1 = feats.sum(0)           # [D]
    S2 = (feats * feats).sum(0) # [D]
    denom = N - 1

    # Leave-one-out moments
    S1_minus = S1.unsqueeze(0) - feats           # [N,D]
    S2_minus = S2.unsqueeze(0) - feats * feats   # [N,D]

    mu_loo = S1_minus / denom                    # [N,D]
    m2_loo = S2_minus / denom                    # [N,D]
    var_loo = torch.clamp(m2_loo - mu_loo * mu_loo, min=0.0)
    std_loo = torch.sqrt(var_loo + eps)

    # FID_-i for each i
    diff_mu = mu_r.unsqueeze(0) - mu_loo         # [N,D]
    diff_sd = std_r.unsqueeze(0) - std_loo       # [N,D]
    fid_minus = (diff_mu * diff_mu).sum(1) + (diff_sd * diff_sd).sum(1)  # [N]

    contribs = fid_minus - fid_full
    return fid_full, contribs

class EMADiagMoments:
    """
    Tracks EMA of first (mean) and second raw moments (mean of squares) per feature-dimension.
    Debiasing follows the standard Adam-like correction: m_hat = m / (1 - beta^t).
    """
    def __init__(
        self,
        dim: int,
        beta: float = 0.99,
        debias: bool = True,
        eps: float = 1e-12,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
    ):
        self.dim = int(dim)
        self.beta = float(beta)
        self.debias = bool(debias)
        self.eps = float(eps)
        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype

        self._m1 = torch.zeros(self.dim, device=self.device, dtype=self.dtype)  # EMA of mean
        self._m2 = torch.zeros(self.dim, device=self.device, dtype=self.dtype)  # EMA of mean-of-squares
        self._t = 0  # number of updates applied

    @torch.no_grad()
    def _debias_factor(self, t_plus: int) -> float:
        if not self.debias:
            return 1.0
        # t_plus = prospective step number (e.g., t+1 when peeking next update)
        return 1.0 - (self.beta ** t_plus)

    @torch.no_grad()
    def bootstrap_from_mu_std(self, mu: torch.Tensor, std: torch.Tensor, debiased_target: bool = True):
        """
        Initialize EMA buffers from unbiased target moments (mu, std).
        If debiased_target=True, sets internal state so current_mu_std() returns (mu, std).
        """
        mu = mu.to(device=self.device, dtype=self.dtype)
        std = std.to(device=self.device, dtype=self.dtype)
        m2 = std * std + mu * mu  # E[x^2] = Var + (E[x])^2

        if debiased_target:
            one_minus = 1.0 - self.beta
            self._m1 = one_minus * mu
            self._m2 = one_minus * m2
            self._t = 1  # so debiasing divides by (1 - beta^t) = (1 - beta)
        else:
            self._m1 = mu
            self._m2 = m2
            self._t = 0


    @torch.no_grad()
    def current_mu_std(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (mu_ema, std_ema) as debiased estimates derived from current state.
        Shapes: both [D], float64 on configured device.
        """
        if self.debias and self._t > 0:
            denom = 1.0 - (self.beta ** self._t)
            m1_hat = self._m1 / denom
            m2_hat = self._m2 / denom
        else:
            m1_hat = self._m1
            m2_hat = self._m2

        var_hat = torch.clamp(m2_hat - m1_hat * m1_hat, min=0.0)
        std_hat = torch.sqrt(var_hat + self.eps)
        return m1_hat, std_hat

    @torch.no_grad()
    def update_from_moments(self, mu_batch: torch.Tensor, m2_batch: torch.Tensor) -> None:
        """
        Updates EMA using batch moments (mean and mean-of-squares).
        Both tensors must be [D] and float-convertible.
        """
        mu_batch = mu_batch.to(device=self.device, dtype=self.dtype)
        m2_batch = m2_batch.to(device=self.device, dtype=self.dtype)

        beta = self.beta
        one_minus = 1.0 - beta

        self._m1 = beta * self._m1 + one_minus * mu_batch
        self._m2 = beta * self._m2 + one_minus * m2_batch
        self._t += 1

    @torch.no_grad()
    def peek_update_from_moments(
        self,
        mu_batch: torch.Tensor,  # [D] or [B,D]
        m2_batch: torch.Tensor,  # [D] or [B,D]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the *debiased* (mu_ema_next, std_ema_next) that would result
        *if* we applied an update with the provided batch moments, WITHOUT
        mutating internal state. Supports vectorized [B,D] inputs.

        Output shapes: if input [D] -> ([D],[D]); if [B,D] -> ([B,D],[B,D])
        """
        mu_batch = mu_batch.to(device=self.device, dtype=self.dtype)
        m2_batch = m2_batch.to(device=self.device, dtype=self.dtype)

        beta = self.beta
        one_minus = 1.0 - beta

        # Broadcast EMA buffers to match batch shape if needed
        # Target shape: either [D] or [B,D]
        target_shape = mu_batch.shape
        m1 = self._m1
        m2 = self._m2
        if mu_batch.dim() == 2:  # [B,D]
            B, D = mu_batch.shape
            assert D == self.dim, "Dimension mismatch in peek_update_from_moments"
            m1 = m1.unsqueeze(0).expand(B, D)
            m2 = m2.unsqueeze(0).expand(B, D)
        else:
            assert mu_batch.numel() == self.dim, "Dimension mismatch in peek_update_from_moments"

        m1_next = beta * m1 + one_minus * mu_batch
        m2_next = beta * m2 + one_minus * m2_batch

        if self.debias:
            denom = self._debias_factor(self._t + 1)
            m1_hat = m1_next / denom
            m2_hat = m2_next / denom
        else:
            m1_hat = m1_next
            m2_hat = m2_next

        var_hat = torch.clamp(m2_hat - m1_hat * m1_hat, min=0.0)
        std_hat = torch.sqrt(var_hat + self.eps)

        return m1_hat.reshape(target_shape), std_hat.reshape(target_shape)

class InceptionDiagFIDEMAReward:
    """
    Returns **per-sample** rewards via:
      - "loo_batch": vectorized LOO on the current batch vs real stats (diag FID MLE).
      - "ema_loo":   per-sample LOO contributions to the FID measured *after* the EMA update.
      - "ema_scalar": broadcast scalar = (EMA FID before) - (EMA FID after).

    Positive reward = better (sample reduces FID).

    reward(images_uint8 [N,H,W,3]) -> tensor [N]
    """

    def __init__(self,
                 device: torch.device,
                 stats_npz: str,
                 beta: float = 0.99,
                 debias_ema: bool = True,
                 eps: float = 1e-12,
                 resize_hw: int = 299,
                 weights_name: Optional[str] = "IMAGENET1K_V1",
                 reward_mode: str = "loo_batch",
                 init_gen_stats_npz: Optional[str] = None):
        """
        stats_npz: npz with {'mu': [D], 'sigma': [D,D]} or {'mu': [D], 'sigma_diagonal': [D]}.
                   We'll only use diag (std_r).
        """
        import numpy as np

        self.device = device
        self.eps = float(eps)
        self.resize_hw = int(resize_hw)
        self.reward_mode = reward_mode
        if self.reward_mode not in {"loo_batch", "ema_loo", "ema_scalar"}:
            raise ValueError(f"Unsupported FID reward mode: {self.reward_mode}")

        # Load real distribution stats
        data = np.load(stats_npz)
        mu_r_np = data["mu"].astype("float64")
        if "sigma_diagonal" in data:
            var_r_np = data["sigma_diagonal"].astype("float64")
        else:
            var_r_np = np.diag(data["sigma"].astype("float64")).copy()

        self.mu_r = torch.from_numpy(mu_r_np).to(device=self.device, dtype=torch.float64)
        self.std_r = torch.sqrt(torch.clamp(torch.from_numpy(var_r_np), min=0.0)).to(self.device, dtype=torch.float64)

        D = self.mu_r.numel()

        # EMA tracker (on float64 for stability)
        self.ema = EMADiagMoments(dim=D, beta=beta, debias=debias_ema, eps=eps,
                                  device=device, dtype=torch.float64)

        if init_gen_stats_npz:
            try:
                g = np.load(init_gen_stats_npz)
                mu_g_np = g["mu"].astype("float64")
                if "sigma_diagonal" in g:
                    var_g_np = g["sigma_diagonal"].astype("float64")
                else:
                    var_g_np = np.diag(g["sigma"].astype("float64")).copy()
                mu_g = torch.from_numpy(mu_g_np).to(self.device, dtype=torch.float64)
                std_g = torch.sqrt(torch.clamp(torch.from_numpy(var_g_np), min=0.0)).to(self.device, dtype=torch.float64)
                self.ema.bootstrap_from_mu_std(mu_g, std_g, debiased_target=True)
            except Exception as e:
                print(f"[warn] init_gen_stats_npz ignored ({e})")

        self.inception = InceptionPool3(device=self.device, weights_name=weights_name, resize_hw=resize_hw)

        self.last_fid_diag_batch: Optional[float] = None
        self.last_fid_diag_ema: Optional[float] = None
        self.last_t_fid_ms = {}


    @torch.no_grad()
    def _preprocess(self, images_uint8: torch.Tensor) -> torch.Tensor:
        """
        images_uint8: [N,H,W,3] uint8 on CPU or CUDA.
        Returns float tensor [N,3,H,W] in [0,1] on self.device.
        """
        if images_uint8.dtype != torch.uint8:
            raise ValueError("images_uint8 must be uint8 [N,H,W,3]")
        x = images_uint8.to(self.device, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0
        return x

    @torch.no_grad()
    def __call__(self, images_uint8: torch.Tensor, prompts_ignored=None) -> torch.Tensor:
        import torch.distributed as dist
        
        # 1) features on this rank
        x = self._preprocess(images_uint8)                                # [n_local,3,H,W]
        feats = self.inception(x).to(torch.float32)                       # [n_local,D]
        n_local, D = feats.shape
        device = feats.device

        # 2) local raw sums (float64 for stability)
        S1_local = feats.sum(0, dtype=torch.float64)                      # [D]
        S2_local = (feats.double().pow(2)).sum(0)                         # [D]
        N_local  = torch.tensor([n_local], dtype=torch.long, device=device)
        # 3) global reduce (across ranks)
        if dist.is_available() and dist.is_initialized():
            S1 = S1_local.clone();  dist.all_reduce(S1, op=dist.ReduceOp.SUM)
            S2 = S2_local.clone();  dist.all_reduce(S2, op=dist.ReduceOp.SUM)
            N  = N_local.clone();   dist.all_reduce(N,  op=dist.ReduceOp.SUM)
        else:
            S1, S2, N = S1_local, S2_local, N_local
        N_total = int(N.item())

        # 4) global batch moments & batch FID (diagonal MLE)
        eps = self.eps
        mu_g  = S1 / max(N_total, 1)                                      # [D]
        m2_g  = S2 / max(N_total, 1)                                      # [D]
        var_g = torch.clamp(m2_g - mu_g * mu_g, min=0.0)                  # [D]
        std_g = torch.sqrt(var_g + eps)                                   # [D]
        fid_batch_full = diag_fid_mle(self.mu_r, self.std_r, mu_g, std_g) # scalar float64
        self.last_fid_diag_batch = float(fid_batch_full.item())

        # 5) snapshot EMA *before* applying update (for scalar or diagnostics)
        mu_ema_prev, std_ema_prev = self.ema.current_mu_std()

        # 6) vectorized LOO for local samples using global sums
        if n_local == 0:
            # even if no local samples, apply EMA update and return empty
            self.ema.update_from_moments(mu_g, m2_g)
            # record EMA FID after the applied update
            mu_ema_now, std_ema_now = self.ema.current_mu_std()
            self.last_fid_diag_ema = float(diag_fid_mle(self.mu_r, self.std_r, mu_ema_now, std_ema_now).item())
            return torch.zeros(0, device=device, dtype=torch.float32)

        # If N_total < 2, LOO is undefined; we'll still produce zeros and proceed with EMA update
        denom = max(N_total - 1, 1)
        f64 = feats.double()                                              # [n_local,D]
        S1m = (S1.unsqueeze(0) - f64) / denom                             # mu_loo_i
        S2m = (S2.unsqueeze(0) - f64*f64) / denom                         # m2_loo_i

        var_loo = torch.clamp(S2m - S1m*S1m, min=0.0)
        std_loo = torch.sqrt(var_loo + eps)                               # [n_local,D]
        if self.reward_mode == "loo_batch" or N_total < 2:
            # FID_-i for each local i vs real stats, all in float64
            fid_minus = diag_fid_mle(
                self.mu_r.unsqueeze(0), self.std_r.unsqueeze(0),
                S1m, std_loo
            )  # [n_local]
            # Positive = better: if removing i makes FID worse, reward > 0
            contribs = (fid_minus - fid_batch_full).to(dtype=torch.float32)
            rewards = torch.nan_to_num(contribs, nan=0.0, posinf=0.0, neginf=0.0)

            # Apply *real* EMA update once per call (with full batch), record EMA FID
            self.ema.update_from_moments(mu_g, m2_g)
            mu_ema_now, std_ema_now = self.ema.current_mu_std()
            self.last_fid_diag_ema = float(diag_fid_mle(self.mu_r, self.std_r, mu_ema_now, std_ema_now).item())
            return rewards

        elif self.reward_mode == "ema_loo":
            # Peek EMA if we update with the full batch
            mu_ema_full, std_ema_full = self.ema.peek_update_from_moments(mu_g, m2_g)      # [D],[D]

            # Peek EMA for LOO variants (vectorized over local samples)
            mu_ema_loo, std_ema_loo = self.ema.peek_update_from_moments(S1m, S2m)          # [n_local,D],[n_local,D]

            # FID after full update (scalar) and after LOO updates ([n_local])
            fid_after_full = diag_fid_mle(self.mu_r, self.std_r, mu_ema_full, std_ema_full)       # scalar
            fid_after_loo  = diag_fid_mle(self.mu_r.unsqueeze(0), self.std_r.unsqueeze(0),
                                          mu_ema_loo, std_ema_loo)                                  # [n_local]

            # Positive = better: if removing i makes EMA FID worse, reward > 0
            rewards = (fid_after_loo - fid_after_full).to(dtype=torch.float32)
            rewards = torch.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0)

            # Apply the *real* EMA update and record diagnostics
            self.ema.update_from_moments(mu_g, m2_g)
            mu_ema_now, std_ema_now = self.ema.current_mu_std()
            self.last_fid_diag_ema = float(diag_fid_mle(self.mu_r, self.std_r, mu_ema_now, std_ema_now).item())
            return rewards

        else:
            # Improvement in EMA FID caused by this batch (broadcast per sample)
            mu_ema_full, std_ema_full = self.ema.peek_update_from_moments(mu_g, m2_g)
            fid_prev = diag_fid_mle(self.mu_r, self.std_r, mu_ema_prev, std_ema_prev)      # scalar
            fid_next = diag_fid_mle(self.mu_r, self.std_r, mu_ema_full, std_ema_full)      # scalar
            reward_scalar = (fid_prev - fid_next).to(dtype=torch.float32)  # positive if FID decreased

            rewards = torch.full((n_local,), reward_scalar.item(), device=device, dtype=torch.float32)

            # Apply the real EMA update and record diagnostics
            self.ema.update_from_moments(mu_g, m2_g)
            mu_ema_now, std_ema_now = self.ema.current_mu_std()
            self.last_fid_diag_ema = float(diag_fid_mle(self.mu_r, self.std_r, mu_ema_now, std_ema_now).item())
            return rewards

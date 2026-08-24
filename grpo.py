import os, math, argparse
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP

from tokenizer.tokenizer_image.vq_model import VQ_models
from autoregressive.models.gpt import GPT_models
from autoregressive.models.generate import generate as generate_llamagen
from pathlib import Path
from utils.reward_utils import CompositeReward, InceptionDiagFIDEMAReward
from utils.data_helpers import load_id2name_from_txt, canonicalize_alias
from utils.checkpoint_utils import save_training_state, try_load_resume, EMAWrapper, get_state_dict_for_saving
from utils.logger import PhaseTimer


os.environ["TOKENIZERS_PARALLELISM"] = "false"

def selective_log_softmax(logits, target_ids):
    # logits: [B,T,V], target_ids: [B,T]
    logp = F.log_softmax(logits, dim=-1)
    return logp.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)  # [B,T]

def lengths_mask_from_eos(seq, eos_id, T):
    if eos_id < 0:
        return torch.ones_like(seq, dtype=seq.dtype).float()
    is_eos = (seq == eos_id)
    first_eos_idx = torch.where(
        is_eos.any(dim=1),
        is_eos.float().argmax(dim=1),
        torch.full((seq.size(0),), T - 1, device=seq.device, dtype=torch.long),
    )
    arange = torch.arange(T, device=seq.device).unsqueeze(0).expand(seq.size(0), -1)
    mask = (arange <= first_eos_idx.unsqueeze(1)).to(seq.dtype).float()
    return mask

@torch.no_grad()
def decode_tokens_to_images(vq_model, index_sample, codebook_embed_dim, H_tokens, W_tokens):
    B, T = index_sample.shape
    idx = index_sample.long()

    qzshape = [B, codebook_embed_dim, H_tokens, W_tokens]
    imgs = vq_model.decode_code(idx, qzshape)

    # imgs is [-1,1] -> [0,1] -> uint8 HWC
    imgs = torch.clamp(0.5 * imgs + 0.5, 0.0, 1.0)
    imgs = (imgs * 255.0).permute(0, 2, 3, 1).to(torch.uint8)
    return imgs


def _ensure_kv_and_pos_on_device(mdl):
    """Move newly (re)created KV-cache buffers and position caches to the model's device/dtype."""
    p = next(mdl.parameters())
    dev, dtype = p.device, p.dtype

    # KV buffers
    for blk in mdl.layers:
        kv = getattr(getattr(blk, "attention", None), "kv_cache", None)
        if kv is not None:
            if hasattr(kv, "k_cache") and kv.k_cache.device != dev:
                kv.k_cache = kv.k_cache.to(device=dev, dtype=dtype)
            if hasattr(kv, "v_cache") and kv.v_cache.device != dev:
                kv.v_cache = kv.v_cache.to(device=dev, dtype=dtype)

    # position/attention buffers
    if hasattr(mdl, "causal_mask") and mdl.causal_mask.device != dev:
        mdl.causal_mask = mdl.causal_mask.to(dev)
    if hasattr(mdl, "freqs_cis"):
        try:
            if mdl.freqs_cis.device != dev:
                mdl.freqs_cis = mdl.freqs_cis.to(dev)
        except AttributeError:
            # in case it's an iterable of tensors
            mdl.freqs_cis = type(mdl.freqs_cis)(x.to(dev) for x in mdl.freqs_cis)

def _patch_kv_update_to_be_device_safe(mdl):
    import types, torch
    for mod in mdl.modules():
        kv = getattr(mod, "kv_cache", None)
        if kv is None or not hasattr(kv, "update"):
            continue
        orig_update = kv.update

        def wrapped_update(input_pos, k_val, v_val, _orig=orig_update, _kv=kv):
            dev = k_val.device
            # move registered buffers to the same device/dtype as k_val
            if hasattr(_kv, "k_cache") and torch.is_tensor(_kv.k_cache) and _kv.k_cache.device != dev:
                _kv.k_cache = _kv.k_cache.to(device=dev, dtype=k_val.dtype)
            if hasattr(_kv, "v_cache") and torch.is_tensor(_kv.v_cache) and _kv.v_cache.device != dev:
                _kv.v_cache = _kv.v_cache.to(device=dev, dtype=v_val.dtype)

            if not torch.is_tensor(input_pos):
                input_pos = torch.as_tensor(input_pos, device=dev, dtype=torch.long)
            elif input_pos.device != dev:
                input_pos = input_pos.to(dev)

            return _orig(input_pos, k_val, v_val)

        kv.update = types.MethodType(wrapped_update, kv)

def ddp_mean_scalar(x: torch.Tensor) -> float:
    """All-reduce mean of a 0-dim or 1-dim scalar tensor across ranks. Returns float on rank-any."""
    if x.ndim > 0:
        x = x.mean()
    x = x.detach().to(torch.float32).clone()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        x /= dist.get_world_size()
    return float(x.cpu())

def ddp_mean_dict(d: dict) -> dict[str, float]:
    """All-reduce mean on CUDA tensors. Cast non-tensors to CUDA first (NCCL requires CUDA)."""
    out = {}
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda", torch.cuda.current_device()) if use_cuda else torch.device("cpu")
    world = dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1

    for k, v in d.items():
        if not torch.is_tensor(v):
            v = torch.as_tensor(v, device=device, dtype=torch.float32)
        else:
            v = v.detach().to(device=device, dtype=torch.float32)

        if world > 1:
            dist.all_reduce(v, op=dist.ReduceOp.SUM)
            v /= world
        out[k] = float(v.item())
    return out

def _save_sample_images(images_uint8: torch.Tensor,
                        prompts: list[str] | None,
                        out_dir: str,
                        step: int,
                        rank: int,
                        g_idx: int,
                        max_images: int = 8):
    """
    Save up to max_images from a [B,H,W,3] uint8 tensor. Also write a .txt with the prompt.
    """
    try:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        B = int(images_uint8.size(0))
        k = min(max_images, B)
        imgs_cpu = images_uint8[:k].detach().cpu().numpy()  # [k,H,W,3]
        for i in range(k):
            fname_png = os.path.join(out_dir, f"step{step:06d}_rank{rank}_g{g_idx}_b{i}.png")
            Image.fromarray(imgs_cpu[i]).save(fname_png)
            if prompts is not None and i < len(prompts):
                fname_txt = os.path.join(out_dir, f"step{step:06d}_rank{rank}_g{g_idx}_b{i}.txt")
                with open(fname_txt, "w", encoding="utf-8") as f:
                    f.write(str(prompts[i]))
    except Exception as e:
        print(f"[warn] saving samples failed: {e}")



def forward_logits_parallel(model, c_indices, index_sample, requires_grad=True):
    """
    Returns logits [B, T, V] aligned so that gather(logits, seq) gives logp(token_t).
    """
    mdl = model.module if hasattr(model, "module") else model
    B, T = index_sample.shape
    p = next(mdl.parameters())
    dev = p.device

    ctx = torch.enable_grad() if requires_grad else torch.no_grad()
    with ctx:
        logits, _ = mdl(idx=index_sample.long().to(dev), cond_idx=c_indices.to(dev), input_pos=None, mode="train")
        if logits.size(1) != T:
            logits = logits[:, :T, :]
        return logits

def grpo_train_step(
    gpt_model, vq_model,
    c_indices, prompts,
    latent_tokens_T, eos_id, codebook_embed_dim,
    G=2, K=2, temperature=1.0, top_k=1000, top_p=1.0,
    eps=0.2, beta=0.0, ema_ref: EMAWrapper | None = None,
    optimizer: torch.optim.Optimizer = None,
    reward_fn=None,
    fid_reward=None,
    cfg_scale=7.5,
    w_fid: float = 0.0,
    save_images: bool = False,
    samples_dir: str | None = None,
    step_for_saving: int = 0,
    rank_for_saving: int = 0,
    max_images_to_save: int = 8,
    ent_coef_base: float = 2.2e-3,
    ent_min: float = 7e-5,
    ent_max: float = 4e-3,
    ent_target_frac: float = 0.78,
    ent_deadband: float = 0.015,
    ent_k: float = 3.0,
    warmup_frac: float = 0.05,
    decay_start_frac: float = 0.85,
    global_step: int | None = None,
    total_steps: int | None = None
):
    import math, contextlib
    device = c_indices.device
    B = c_indices.size(0)
    T = latent_tokens_T

    mdl_nograd = gpt_model.module if hasattr(gpt_model, "module") else gpt_model
    if reward_fn is None:
        reward_fn = CompositeReward(device=device, w_clip=1.0, w_hps=0.0)

    timer = PhaseTimer()

    timer.tic("generate")
    with torch.no_grad():
        mdl_nograd.eval()
        seqs = []
        for _ in range(G):
            seq = generate_llamagen(
                mdl_nograd, c_indices, T,
                cfg_scale=cfg_scale,
                temperature=temperature, top_k=top_k, top_p=top_p, sample_logits=True,
            )
            seqs.append(seq)
        seqs = torch.stack(seqs, dim=1)  # [B, G, T]
    mdl_nograd.train()
    timer.toc("generate")

    timer.tic("reward")
    H_tokens = W_tokens = int(math.sqrt(T))

    imgs_per_g = []
    with torch.inference_mode():
        for g in range(G):
            imgs_g = decode_tokens_to_images(vq_model, seqs[:, g], codebook_embed_dim, H_tokens, W_tokens)
            imgs_per_g.append(imgs_g)

    if save_images and samples_dir:
        for g, imgs_g in enumerate(imgs_per_g):
            _save_sample_images(
                images_uint8=imgs_g, prompts=prompts, out_dir=samples_dir,
                step=step_for_saving, rank=rank_for_saving, g_idx=g, max_images=max_images_to_save,
            )

    imgs_cat = torch.cat(imgs_per_g, dim=0)  # [G*B, H, W, 3]
    prompts_cat = prompts * G

    tok = None
    if reward_fn.clip is not None:
        import open_clip
        tok = open_clip.tokenize(prompts_cat)

    r_clip_cat = (
        reward_fn.clip(images_uint8=imgs_cat, prompts_or_tok=tok)
        if reward_fn.clip is not None
        else torch.zeros(imgs_cat.size(0), device=device, dtype=torch.float32)
    )
    r_hps_cat = (
        reward_fn.hps(images_uint8=imgs_cat, prompts=prompts_cat)
        if reward_fn.hps is not None
        else torch.zeros(imgs_cat.size(0), device=device, dtype=torch.float32)
    )

    r_fid_cat = None
    if fid_reward is not None and w_fid != 0.0:
        r_fid_cat = fid_reward(images_uint8=imgs_cat, prompts_ignored=None)  # [G*B]


    r_clip = r_clip_cat.view(G, B).transpose(0, 1).contiguous()
    r_hps  = r_hps_cat.view(G, B).transpose(0, 1).contiguous()
    r_fid  = torch.zeros_like(r_clip) if r_fid_cat is None else r_fid_cat.view(G, B).transpose(0, 1).contiguous()

    rewards_comp = reward_fn.wc * r_clip + reward_fn.wh * r_hps

    rewards_fid = w_fid * r_fid

    clip_mean   = r_clip.mean()
    hps_mean    = r_hps.mean()

    timer.toc("reward")

    mean_c = rewards_comp.mean(dim=1, keepdim=True)
    std_c  = rewards_comp.std (dim=1, keepdim=True)
    adv_comp = (rewards_comp - mean_c) / (std_c + 1e-4)  # [B,G]

    if (w_fid != 0.0) and (fid_reward is not None):
        mean_f = rewards_fid.mean(dim=1, keepdim=True)
        std_f  = rewards_fid.std (dim=1, keepdim=True)
        adv_fid = (rewards_fid - mean_f) / (std_f + 1e-4)  # [B,G]
    else:
        adv_fid = None

    diag_stats = {
        "R/mean": (rewards_comp + rewards_fid).mean(),
        "R/std":  (rewards_comp + rewards_fid).std(),
        "A/mean": adv_comp.mean(),
        "A/std":  adv_comp.std(),
    }
    if fid_reward is not None:
        if fid_reward.last_fid_diag_batch is not None:
            diag_stats["FID/diag_batch"] = torch.tensor(fid_reward.last_fid_diag_batch, device=device)
        if fid_reward.last_fid_diag_ema is not None:
            diag_stats["FID/diag_ema"] = torch.tensor(fid_reward.last_fid_diag_ema, device=device)

    timer.tic("old_logp")
    with torch.no_grad():
        logp_old_seq = []
        masks = []
        mask_util = []
        for g in range(G):
            seq = seqs[:, g]  # [B, T]
            logits_old = forward_logits_parallel(gpt_model, c_indices, seq, requires_grad=False)  # [B,T,V]
            logp_old = selective_log_softmax(logits_old, seq)  # [B,T]
            mask = lengths_mask_from_eos(seq, eos_id=eos_id, T=seq.size(1))  # [B,T]
            lp = (logp_old * mask).sum(1) / mask.sum(1).clamp(min=1.0)       # [B]
            logp_old_seq.append(lp)
            masks.append(mask)
            mask_util.append(mask.float().mean())  # fraction of tokens counted
        logp_old_seq = torch.stack(logp_old_seq, dim=1)   # [B, G]
        masks = torch.stack(masks, dim=1)                 # [B, G, T]
        mask_util = torch.stack(mask_util).mean()         # scalar
    timer.toc("old_logp")

    total_loss = torch.zeros((), device=device)
    clip_low = torch.zeros((), device=device)
    clip_high = torch.zeros((), device=device)
    clip_region = torch.zeros((), device=device)
    ratios_all, kls_all = [], []

    is_ddp = isinstance(gpt_model, torch.nn.parallel.DistributedDataParallel)
    nullctx = contextlib.nullcontext

    timer.tic("train")
    for epoch in range(K):
        optimizer.zero_grad(set_to_none=True)
        epoch_loss_total = torch.zeros((), device=device)

        with (gpt_model.no_sync() if is_ddp else nullctx()):
            epoch_loss_comp = torch.zeros((), device=device)
            for g in range(G):
                seq = seqs[:, g]  # [B,T]
                logits_now = forward_logits_parallel(gpt_model, c_indices, seq, requires_grad=True)
                logp_full = F.log_softmax(logits_now, dim=-1)      # [B,T,V]
                p_full = logp_full.exp()
                ent = (-p_full * logp_full).sum(dim=-1)            # [B,T]
                ent_masked = (ent * masks[:, g]).sum(1) / masks[:, g].sum(1).clamp(min=1.0)  # [B]
                ent_mean = ent_masked.mean()

                V = logits_now.size(-1)
                H_max = math.log(V)
                H_norm = float((ent_mean.detach() / H_max).clamp(0, 1))
                diag_stats["H/token_mean"] = ent_mean.detach()
                diag_stats["H/token_norm"] = torch.as_tensor(H_norm, device=logits_now.device)

                progress = None
                if (global_step is not None) and (total_steps is not None) and total_steps > 0:
                    progress = max(0.0, min(1.0, global_step / total_steps))

                if progress is None:
                    sched_coef = ent_coef_base
                else:
                    if progress < warmup_frac:
                        sched_coef = ent_coef_base * (progress / warmup_frac)
                    elif progress < decay_start_frac:
                        sched_coef = ent_coef_base
                    else:
                        t = (progress - decay_start_frac) / max(1e-8, (1.0 - decay_start_frac))
                        sched_coef = ent_coef_base * 0.5 * (1.0 + math.cos(math.pi * t))

                err = ent_target_frac - H_norm
                if abs(err) > ent_deadband:
                    sched_coef *= math.exp(ent_k * err)
                sched_coef = float(min(max(sched_coef, ent_min), ent_max))
                diag_stats["ent_coef_eff"] = torch.as_tensor(sched_coef, device=logits_now.device)

                logp_now = selective_log_softmax(logits_now, seq)      # [B,T]
                logp_now_seq = (logp_now * masks[:, g]).sum(1) / masks[:, g].sum(1).clamp(min=1.0)  # [B]

                ratio = torch.exp(logp_now_seq - logp_old_seq[:, g])   # [B]
                clipped = torch.clamp(ratio, 1 - eps, 1 + eps)         # [B]
                A = adv_comp[:, g]                                     # [B]
                loss_g = torch.min(-ratio * A, -clipped * A).mean()    # scalar
                loss_g = loss_g - sched_coef * ent_mean

                if beta > 0 and ema_ref is not None:
                    with torch.no_grad():
                        ema_ref.model.eval()
                        logits_ref = forward_logits_parallel(ema_ref.model, c_indices, seq, requires_grad=False)
                        logp_ref = selective_log_softmax(logits_ref, seq)
                        logp_ref_seq = (logp_ref * masks[:, g]).sum(1) / masks[:, g].sum(1).clamp(min=1.0)
                    delta = logp_ref_seq - logp_now_seq
                    kl = torch.exp(delta) - delta - 1
                    loss_g = loss_g + beta * kl.mean()
                    kls_all.append(kl.mean().detach())

                epoch_loss_comp = epoch_loss_comp + loss_g

                low = ((ratio < (1 - eps)) & (A < 0)).float().mean()
                high = ((ratio > (1 + eps)) & (A > 0)).float().mean()
                region = ((ratio < (1 - eps)) | (ratio > (1 + eps))).float().mean()
                clip_low += low / (K * G)
                clip_high += high / (K * G)
                clip_region += region / (K * G)
                ratios_all.append(ratio.mean().detach())

            epoch_loss_comp.backward()
            epoch_loss_total = epoch_loss_total + epoch_loss_comp.detach()

        if adv_fid is not None:
            epoch_loss_fid = torch.zeros((), device=device)
            for g in range(G):
                seq = seqs[:, g]
                logits_now = forward_logits_parallel(gpt_model, c_indices, seq, requires_grad=True)
                logp_now = selective_log_softmax(logits_now, seq)                              # [B,T]
                logp_now_seq = (logp_now * masks[:, g]).sum(1) / masks[:, g].sum(1).clamp(min=1.0)  # [B]

                ratio = torch.exp(logp_now_seq - logp_old_seq[:, g])   # [B]
                clipped = torch.clamp(ratio, 1 - eps, 1 + eps)
                A = adv_fid[:, g]
                loss_g = 2 * torch.min(-ratio * A, -clipped * A).mean()   
                epoch_loss_fid = epoch_loss_fid + loss_g

            epoch_loss_fid.backward()
            epoch_loss_total = epoch_loss_total + epoch_loss_fid.detach()

        torch.nn.utils.clip_grad_norm_(gpt_model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss = total_loss + epoch_loss_total

    timer.toc("train")

    time_stats = timer.get()
    tokens_total = float(B * T * G)
    toks_gen = tokens_total
    toks_reward = tokens_total
    toks_old = tokens_total
    passes = 1 + (1 if (adv_fid is not None) else 0)
    toks_train = tokens_total * K * passes

    local_stats = {
        "loss": (total_loss / K),
        "reward_mean": (rewards_comp + rewards_fid).mean(),
        "R/clip": clip_mean,
        "R/hps": hps_mean,
        "clip_region": clip_region,
        "clip_low": clip_low,
        "clip_high": clip_high,
        "ratio/mean": torch.tensor(0.0, device=device) if not ratios_all else torch.stack([r.to(device) if not isinstance(r, torch.Tensor) else r for r in ratios_all]).mean(),
        "kl/mean": torch.tensor(0.0, device=device) if not kls_all else torch.stack([k.to(device) if not isinstance(k, torch.Tensor) else k for k in kls_all]).mean(),
        "mask_util": mask_util,
        "toks_per_s/generate": torch.tensor(toks_gen / max(time_stats.get("generate", 1e-9), 1e-9), device=device),
        "toks_per_s/reward":   torch.tensor(toks_reward / max(time_stats.get("reward",   1e-9), 1e-9), device=device),
        "toks_per_s/oldlp":    torch.tensor(toks_old    / max(time_stats.get("old_logp", 1e-9), 1e-9), device=device),
        "toks_per_s/train":    torch.tensor(toks_train  / max(time_stats.get("train",    1e-9), 1e-9), device=device),
        **{k: (v if isinstance(v, torch.Tensor) else torch.as_tensor(v, device=device))
           for k, v in diag_stats.items()}
    }

    stats = ddp_mean_dict(local_stats)
    return stats



def main(args):
    if args.ent_coef_base < 0 or args.ent_min < 0 or args.ent_max < args.ent_min:
        raise ValueError("Entropy coefficients must satisfy 0 <= ent_min <= ent_max and ent_coef_base >= 0")
    if not 0.0 <= args.ent_target_frac <= 1.0:
        raise ValueError("--ent-target-frac must be in [0, 1]")
    if args.ent_deadband < 0 or args.ent_k < 0:
        raise ValueError("--ent-deadband and --ent-k must be non-negative")
    if not 0.0 <= args.warmup_frac <= args.decay_start_frac <= 1.0:
        raise ValueError("Entropy schedule must satisfy 0 <= warmup_frac <= decay_start_frac <= 1")
    if args.G < 2:
        raise ValueError("--G must be at least 2 to compute within-prompt advantages")
    if args.K < 1 or args.per_proc_batch_size < 1 or args.max_iterations < 1:
        raise ValueError("--K, --per-proc-batch-size, and --max-iterations must be positive")
    if args.save_every < 1 or args.save_samples_every < 0:
        raise ValueError("--save-every must be positive and --save-samples-every must be non-negative")

    assert torch.cuda.is_available(), "Training requires at least one GPU."
    torch.set_grad_enabled(True)

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * world + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    rng_cuda = torch.Generator(device=f"cuda:{device}")
    rng_cuda.manual_seed(seed)
    print(f"Starting rank={rank}, seed={seed}, world_size={world}.")

    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim
    ).to(device)
    vq_model.eval()
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu", weights_only=False)
    vq_model.load_state_dict(checkpoint["model"])
    del checkpoint
    print("image tokenizer loaded")

    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    latent_size = args.image_size // args.downsample_size
    latent_tokens_T = latent_size ** 2
    max_seq_len = args.cls_token_num + latent_tokens_T

    gpt_model = GPT_models[args.gpt_model](
        block_size=latent_tokens_T,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
    ).to(device=device, dtype=precision)

    checkpoint = torch.load(args.gpt_ckpt, map_location="cpu", weights_only=False)
    if "model" in checkpoint:
        model_weight = checkpoint["model"]
    elif "module" in checkpoint:
        model_weight = checkpoint["module"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise Exception("please check model weight")
    gpt_model.load_state_dict(model_weight, strict=False)
    del checkpoint
    print("gpt model loaded (llamagen)")

    if args.compile:
        gpt_model = torch.compile(gpt_model, mode="reduce-overhead", fullgraph=True)

    _patch_kv_update_to_be_device_safe(gpt_model)
    gpt_model.setup_caches(
        max_batch_size=args.per_proc_batch_size,
        max_seq_length=max_seq_len,
        dtype=precision,
    )
    _ensure_kv_and_pos_on_device(gpt_model)
    gpt_model.to(device)

    gpt_model = DDP(gpt_model, device_ids=[device], output_device=device, find_unused_parameters=False)

    ema_model = GPT_models[args.gpt_model](
        block_size=latent_size ** 2,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
    ).to(device=device, dtype=precision)
    ema_model.setup_caches(args.per_proc_batch_size, max_seq_len, precision)
    _ensure_kv_and_pos_on_device(ema_model)
    ema_model.to(device)

    src_state = get_state_dict_for_saving(gpt_model)
    src_state = {k: v for k, v in src_state.items() if ".kv_cache." not in k}
    ema_model.load_state_dict(src_state, strict=False)

    ema_ref = EMAWrapper(ema_model, beta=args.ema_beta) if args.beta > 0.0 else None


    if dist.get_rank() == 0:
        os.makedirs(args.output_dir, exist_ok=True)

    optimizer = torch.optim.AdamW(gpt_model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)

    global_step, start_iter = try_load_resume(args.resume, gpt_model, optimizer, ema_ref)
    torch.cuda.manual_seed_all(args.global_seed * world + rank + global_step*100003)

    id2name = None
    if args.gpt_type == "c2i":
        assert os.path.exists(args.id2name_txt), f"--id2name-txt not found: {args.id2name_txt}"
        id2name = load_id2name_from_txt(args.id2name_txt)
        if args.num_classes > len(id2name):
            raise ValueError(f"--num_classes ({args.num_classes}) exceeds mapping size ({len(id2name)})")

    gpt_model.train()

    n = args.per_proc_batch_size
    global_batch_size = n * world
    num_train_samples = args.num_train_samples
    total_samples = int(math.ceil(num_train_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total samples to train on: {total_samples}")
    assert total_samples % world == 0
    samples_needed_this_gpu = total_samples // world
    assert samples_needed_this_gpu % n == 0
    iterations = args.max_iterations

    latent_tokens_T = latent_size ** 2
    eos_id = args.eos_id
    codebook_embed_dim = args.codebook_embed_dim

    reward_fn = CompositeReward(device=device, w_clip=args.w_clip, w_hps=args.w_hps)
    fid_reward = None
    if args.w_fid != 0.0:
        if not args.fid_stats_npz:
            raise ValueError("--fid-stats-npz is required when --w-fid is non-zero")
        fid_reward = InceptionDiagFIDEMAReward(
            device=device,
            stats_npz=args.fid_stats_npz,
            beta=getattr(args, "fid_beta", 0.99),
            debias_ema=(not getattr(args, "fid_no_debias", False)),
            eps=1e-12,
            resize_hw=299,
            reward_mode=args.reward_mode,
            init_gen_stats_npz=(args.fid_init_gen_npz if getattr(args, "fid_init_gen_npz", "") else None)
        )
    iter_range = range(start_iter, iterations)
    if rank == 0:
        iter_range = tqdm(iter_range)

    for it in iter_range:

        class_ids = torch.randint(0, args.num_classes, (n,), device=device, dtype=torch.long, generator=rng_cuda)
        names = [canonicalize_alias(id2name[int(cid)], policy=args.alias_policy) for cid in class_ids.tolist()]
        prompt_batch = [args.prompt_template.format(name=nm) for nm in names]
        c_indices = class_ids                   # [B]

        stats = grpo_train_step(
            gpt_model, vq_model,
            c_indices, prompts=prompt_batch,
            latent_tokens_T=latent_tokens_T, eos_id=eos_id, codebook_embed_dim=codebook_embed_dim,
            G=args.G, K=args.K, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
            eps=args.eps, beta=args.beta, ema_ref=ema_ref,
            reward_fn=reward_fn,
            fid_reward=fid_reward,
            w_fid=args.w_fid,
            optimizer=optimizer, cfg_scale=args.cfg_scale,
            save_images=(dist.get_rank() == 0 and args.save_samples_every > 0 and (global_step % args.save_samples_every == 0)),
            samples_dir=args.samples_dir,
            step_for_saving=global_step,
            rank_for_saving=dist.get_rank(),
            max_images_to_save=args.num_sample_images,
            ent_coef_base=args.ent_coef_base,
            ent_min=args.ent_min,
            ent_max=args.ent_max,
            ent_target_frac=args.ent_target_frac,
            ent_deadband=args.ent_deadband,
            ent_k=args.ent_k,
            warmup_frac=args.warmup_frac,
            decay_start_frac=args.decay_start_frac,
            global_step=global_step,
            total_steps=iterations,
        )

        global_step += 1

        if (global_step % args.save_every == 0):
            if rank == 0:
                save_training_state(
                    out_dir=args.output_dir,
                    model_ddp=gpt_model,
                    optimizer=optimizer,
                    ema_ref=ema_ref,
                    step=global_step,
                    it=it + 1,
                    args=args,
                    extra_stats=stats,
                )

            dist.barrier()
        if rank == 0:
            iter_range.set_description(f"loss {stats['loss']:.4f} | R {stats['reward_mean']:.3f} | clip {stats['clip_region']:.3f}")
            fid = stats.get('FID/diag_batch', float('nan'))
            print(
                f"[diag] ratio_mean={stats['ratio/mean']:.4f}  "
                f"kl_mean={stats['kl/mean']:.4e}  "
                f"mask_util={stats['mask_util']:.3f}  "
                f"A(mean,std)=({stats['A/mean']:.4f},{stats['A/std']:.4f})  "
                f"H(token_mean,norm)=({stats['H/token_mean']:.4f},{stats['H/token_norm']:.4f})  "
                f"toks/s gen={stats['toks_per_s/generate']:.0f}  "
                f"toks/s reward={stats['toks_per_s/reward']:.0f}  "
                f"oldlp={stats['toks_per_s/oldlp']:.0f}  "
                f"train={stats['toks_per_s/train']:.0f}  "
                f"ent_coeff={stats['ent_coef_eff']:.4f}  "
                f"R/clip={stats['R/clip']:.3f}  R/hps={stats['R/hps']:.3f}  "
                f"FID={fid:.3f}"
            )

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id2name-txt", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "imagenet_id2name.txt"),
                        help="Text file containing a Python dict: {0:'tench, Tinca tinca', 1:'goldfish, ...', ...}")
    parser.add_argument("--prompt-template", type=str, default="a photo of a {name}",
                        help="Template for CLIP/HPS prompts; {name} will be replaced with class name/alias")
    parser.add_argument("--alias-policy", type=str, choices=["first", "full"], default="first",
                        help="How to turn the alias string into a name: 'first' = take the first alias before a comma; 'full' = keep the entire alias string")


    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-L")
    parser.add_argument("--gpt-ckpt", type=str, required=True)
    parser.add_argument("--gpt-type", type=str, choices=['c2i'], default="c2i")
    parser.add_argument("--num-classes", type=int, default=1000, help="number of classes for c2i")
    parser.add_argument("--cls-token-num", type=int, default=1)
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"])
    parser.add_argument("--compile", action='store_true', default=False)

    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, required=True)
    parser.add_argument("--codebook-size", type=int, default=16384)
    parser.add_argument("--codebook-embed-dim", type=int, default=8)

    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=256)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)

    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--per-proc-batch-size", type=int, default=8)
    parser.add_argument("--num-train-samples", type=int, default=1024**2)
    parser.add_argument("--max-iterations", type=int, default=600,
                        help="Number of policy-training iterations; use a small value for smoke tests")

    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)

    parser.add_argument("--G", type=int, default=2, help="completions per prompt")
    parser.add_argument("--K", type=int, default=2, help="inner optimization epochs with cached old log-probs")
    parser.add_argument("--eps", type=float, default=0.2, help="clipping epsilon")
    parser.add_argument("--beta", type=float, default=0.02, help="KL weight to EMA ref")
    parser.add_argument("--ema-beta", type=float, default=0.999, help="EMA decay for reference")
    parser.add_argument("--eos-id", type=int, default=-1, help="-1 means no EOS; otherwise token id used for masking")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--reward-mode", type=str, choices=["loo_batch", "ema_loo", "ema_scalar"], default="ema_loo")
    parser.add_argument("--ent-coef-base", type=float, default=2.2e-3)
    parser.add_argument("--ent-min", type=float, default=7e-5)
    parser.add_argument("--ent-max", type=float, default=4e-3)
    parser.add_argument("--ent-target-frac", type=float, default=0.78)
    parser.add_argument("--ent-deadband", type=float, default=0.015)
    parser.add_argument("--ent-k", type=float, default=3.0)
    parser.add_argument("--warmup-frac", type=float, default=0.05)
    parser.add_argument("--decay-start-frac", type=float, default=0.85)
    parser.add_argument("--w-clip", type=float, default=1.0)
    parser.add_argument("--w-hps", type=float, default=0.0)

    parser.add_argument("--output-dir", type=str, default="grpo_runs/run1")
    parser.add_argument("--save-every", type=int, default=200, help="save every N iterations")
    parser.add_argument("--resume", type=str, default="", help="directory with checkpoint-last.pth")

    parser.add_argument("--save-samples-every", type=int, default=100, help="Save a batch of generated samples every N steps on rank0")
    parser.add_argument("--samples-dir", type=str, default="grpo_runs/samples", help="Directory to write sample PNGs")
    parser.add_argument("--num-sample-images", type=int, default=8, help="How many images from the batch to save per (g)")

    parser.add_argument("--w-fid", type=float, default=0.0, help="Weight for EMA diag FID LOO reward")
    parser.add_argument("--fid-stats-npz", type=str, default="", help="NPZ with Inception mu/sigma (e.g. ImageNet clean-fid stats)")
    parser.add_argument("--fid-beta", type=float, default=0.99, help="EMA beta for generated feature stats")
    parser.add_argument("--fid-no-debias", action="store_true", help="Disable Adam-style EMA debias")
    parser.add_argument("--fid-init-gen-npz", type=str, default="",
                        help="Optional NPZ with *generated* Inception stats (mu/sigma or sigma_diagonal) to bootstrap EMA.")


    args = parser.parse_args()
    main(args)

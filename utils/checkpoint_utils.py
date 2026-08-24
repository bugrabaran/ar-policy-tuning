import torch
import torch.nn as nn
import os
import random
import numpy as np
import time

def safe_save(obj, path, retries=3, delay=3):
    tmp = path + ".tmp"
    for i in range(retries):
        try:
            torch.save(obj, tmp)
            os.replace(tmp, path)       # atomic on POSIX
            return
        except Exception as e:
            print(f"[WARN] save failed {i+1}/{retries}: {e}")
            time.sleep(delay)
    raise RuntimeError(f"Failed to save {path}")

def get_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state

def set_rng_state(state):
    if "python" in state: 
        random.setstate(state["python"])
    if "numpy"  in state: 
        np.random.set_state(state["numpy"])
    if "torch_cpu" in state: 
        torch.set_rng_state(state["torch_cpu"])

def save_training_state(
    out_dir, model_ddp, optimizer, ema_ref, step, it, args, extra_stats=None
):
    os.makedirs(out_dir, exist_ok=True)
    last_ckpt = os.path.join(out_dir, "checkpoint-last.pth")
    prev_ckpt = os.path.join(out_dir, "checkpoint-prev.pth")

    if os.path.exists(last_ckpt):
        os.replace(last_ckpt, prev_ckpt)

    payload = {
        "model": get_state_dict_for_saving(model_ddp),  # unwraps .module internally
        "optimizer": optimizer.state_dict(),
        "ema_model": (ema_ref.model.state_dict() if ema_ref is not None else None),
        "ema_beta": (ema_ref.beta if ema_ref is not None else None),
        "args": vars(args),
        "global_step": step,
        "iteration": it,
        "rng_state": get_rng_state(),
        "stats": extra_stats or {},
    }
    safe_save(payload, last_ckpt)

def try_load_resume(resume_dir, model_ddp, optimizer, ema_ref):
    if not resume_dir:
        return 0, 0  # step, iteration

    last = os.path.join(resume_dir, "checkpoint-last.pth")
    prev = os.path.join(resume_dir, "checkpoint-prev.pth")
    ckpt = None
    if os.path.isfile(last):
        ckpt = torch.load(last, map_location="cpu", weights_only=False)
        print(f"[resume] loaded {last}")
    elif os.path.isfile(prev):
        ckpt = torch.load(prev, map_location="cpu", weights_only=False)
        print(f"[resume] loaded {prev}")
    else:
        print(f"[resume] no checkpoint found under {resume_dir}")
        return 0, 0

    state = ckpt["model"]
    state = {k: v for k, v in state.items() if ".kv_cache." not in k}
    model_ddp.module.load_state_dict(state, strict=False)  # unwrapped module
    if "optimizer" in ckpt and ckpt["optimizer"] is not None:
        optimizer.load_state_dict(ckpt["optimizer"])

    if ema_ref is not None and ckpt.get("ema_model") is not None:
        state_ema = ckpt["ema_model"]
        state_ema = {k: v for k, v in state_ema.items() if ".kv_cache." not in k}
        ema_ref.model.load_state_dict(state_ema, strict=False)
        if "ema_beta" in ckpt and ckpt["ema_beta"] is not None:
            ema_ref.beta = ckpt["ema_beta"]

    step = int(ckpt.get("global_step", 0))
    it = int(ckpt.get("iteration", 0))

    if "rng_state" in ckpt:
        set_rng_state(ckpt["rng_state"])

    return step, it

class EMAWrapper(nn.Module):
    def __init__(self, model, beta=0.997):
        super().__init__()
        self.model = model.eval()     # assume caller passes a separate instance
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.beta = beta

    @torch.no_grad()
    def update(self, src):
        for p_tgt, p_src in zip(self.model.parameters(), src.parameters()):
            p_tgt.data.mul_(self.beta).add_(p_src.data, alpha=1 - self.beta)


def get_state_dict_for_saving(model):
    m = model
    if hasattr(m, "module"):
        m = m.module
    if hasattr(m, "engine"):              # deepspeed
        if hasattr(m.engine, "module"):
            m = m.engine.module
    return m.state_dict()

def save_checkpoint(path, model, optimizer=None, ema_ref: EMAWrapper | None = None, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "model": get_state_dict_for_saving(model),
        "extra": extra or {},
    }
    if optimizer is not None:
        try:
            payload["optimizer"] = optimizer.state_dict()
        except Exception:
            pass
    if ema_ref is not None:
        try:
            payload["ema_model"] = ema_ref.model.state_dict()
            payload["ema_beta"] = ema_ref.beta
        except Exception:
            pass
    torch.save(payload, path)

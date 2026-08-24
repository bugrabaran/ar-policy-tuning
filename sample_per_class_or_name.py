import os, argparse, ast, difflib
from typing import Dict, List, Tuple
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

from tokenizer.tokenizer_image.vq_model import VQ_models
from autoregressive.models.gpt import GPT_models
from autoregressive.models.generate import generate

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def load_id2name(path: str) -> Dict[int, str]:
    """Load an ImageNet-style id->name dict from a .txt containing a Python dict literal."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    d = ast.literal_eval(text)
    return {int(k): str(v) for k, v in d.items()}

def build_alias_lookup(id2name: Dict[int, str]) -> Dict[str, int]:
    """
    Build a case-insensitive alias->id map.
    Each value string may contain comma-separated aliases; we split and strip.
    """
    alias2id = {}
    for idx, name in id2name.items():
        for alias in [a.strip() for a in name.split(",")]:
            if not alias:
                continue
            alias2id[alias.lower()] = idx
    return alias2id

def parse_inline_names(s: str) -> List[str]:
    return [t.strip() for t in s.split(",") if t.strip()]

def parse_names_file(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

def names_to_ids(
    names: List[str],
    alias2id: Dict[str, int],
    all_aliases_sorted: List[str],
    max_suggestions: int = 5
) -> Tuple[List[int], List[str]]:
    """
    Map user-provided names to class ids. Returns (ids, warnings).
    Uses case-insensitive exact match on any alias; provides fuzzy suggestions if not found.
    """
    ids = []
    warnings = []
    seen = set()
    for name in names:
        key = name.lower()
        if key in alias2id:
            cid = alias2id[key]
            if cid not in seen:
                ids.append(cid)
                seen.add(cid)
        else:
            close = difflib.get_close_matches(key, all_aliases_sorted, n=max_suggestions, cutoff=0.6)
            if close:
                suggestions = ", ".join(close)
                warnings.append(f'Unknown class "{name}". Did you mean: {suggestions}?')
            else:
                warnings.append(f'Unknown class "{name}".')
    return ids, warnings

def save_samples(samples_uint8, out_dir, class_id, class_name_tag, start_idx):
    os.makedirs(out_dir, exist_ok=True)
    for i, sample in enumerate(samples_uint8):
        idx = start_idx + i
        fname = f"class_{class_id:04d}_{class_name_tag}_{idx:06d}.png"
        Image.fromarray(sample).save(os.path.join(out_dir, fname))

def main(args):
    torch.set_grad_enabled(False)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    id2name = load_id2name(args.id2name_file)
    alias2id = build_alias_lookup(id2name)
    all_aliases_sorted = sorted(alias2id.keys())

    requested_ids = []
    warnings = []

    if args.class_ids:
        requested_ids.extend([int(x.strip()) for x in args.class_ids.split(",") if x.strip()])

    if args.class_names:
        names = parse_inline_names(args.class_names)
        ids_from_names, w = names_to_ids(names, alias2id, all_aliases_sorted)
        requested_ids.extend(ids_from_names)
        warnings.extend(w)

    if args.class_list_file:
        names = parse_names_file(args.class_list_file)
        ids_from_names, w = names_to_ids(names, alias2id, all_aliases_sorted)
        requested_ids.extend(ids_from_names)
        warnings.extend(w)

    requested_ids = list(dict.fromkeys(requested_ids))
    if warnings:
        print("\n".join(["[warn] " + w for w in warnings]))
    if not requested_ids:
        raise SystemExit("No valid classes resolved. Provide --class-ids, --class-names, or --class-list-file.")

    def short_tag(cid: int) -> str:
        tag = id2name[cid].split(",")[0].strip().lower().replace(" ", "_")
        return "".join(ch for ch in tag if ch.isalnum() or ch in ("_", "-"))

    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim
    ).to(device)
    vq_model.eval()
    vq_ckpt = torch.load(args.vq_ckpt, map_location="cpu", weights_only=False)
    vq_model.load_state_dict(vq_ckpt["model"])
    del vq_ckpt

    latent_size = args.image_size // args.downsample_size
    gpt_model = GPT_models[args.gpt_model](
        vocab_size=args.codebook_size,
        block_size=latent_size ** 2,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
    ).to(device=device, dtype=precision)
    checkpoint = torch.load(args.gpt_ckpt, map_location="cpu", weights_only=False)

    if args.from_fsdp:
        model_weight = checkpoint
    elif "model" in checkpoint:
        model_weight = checkpoint["model"]
    elif "module" in checkpoint:
        model_weight = checkpoint["module"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise RuntimeError("Unrecognized checkpoint format; try --from-fsdp.")

    state = {k: v for k, v in model_weight.items() if ".kv_cache." not in k}
    gpt_model.load_state_dict(state, strict=False)
    del checkpoint
    gpt_model.eval()

    if args.compile:
        print("Compiling GPT with torch.compile(...)")
        gpt_model = torch.compile(gpt_model, mode="reduce-overhead", fullgraph=True)
    else:
        print("No model compile")

    model_string_name = args.gpt_model.replace("/", "-")
    if args.from_fsdp:
        ckpt_string_name = args.gpt_ckpt.split('/')[-2]
    else:
        ckpt_string_name = os.path.basename(args.gpt_ckpt).replace(".pth", "").replace(".pt", "")

    folder_name = (
        f"{model_string_name}-{ckpt_string_name}"
        f"-img{args.image_size}_eval{args.image_size_eval}-{args.vq_model}"
        f"-topk{args.top_k}-topp{args.top_p}-temp{args.temperature}"
        f"-cfg{args.cfg_scale}-seed{args.seed}"
        f"-perclass{args.per_class}-classes{len(requested_ids)}"
    )
    out_dir = os.path.join(args.sample_dir, folder_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Saving samples in: {out_dir}")

    B = args.batch_size
    qz_hw = latent_size
    total_requested = len(requested_ids) * args.per_class
    print(f"Requested total samples: {total_requested} "
          f"({args.per_class} per class across {len(requested_ids)} classes)")

    for cid in requested_ids:
        tag = short_tag(cid)
        generated = 0
        save_index = 0
        print(f"\n[Class {cid} | {id2name[cid]}] Generating {args.per_class} samples...")
        while generated < args.per_class:
            bs = min(B, args.per_class - generated)
            c_indices = torch.full((bs,), int(cid), device=device, dtype=torch.long)
            qzshape = [bs, args.codebook_embed_dim, qz_hw, qz_hw]

            use_amp = (precision != torch.float32) and (use_cuda or precision == torch.bfloat16)
            dtype_for_amp = precision if precision != torch.float32 else (torch.bfloat16 if use_cuda else torch.float32)
            with torch.autocast(device_type="cuda" if use_cuda else "cpu",
                                dtype=dtype_for_amp,
                                enabled=use_amp):
                index_sample = generate(
                    gpt_model, c_indices, qz_hw * qz_hw,
                    cfg_scale=args.cfg_scale, cfg_interval=args.cfg_interval,
                    temperature=args.temperature, top_k=args.top_k,
                    top_p=args.top_p, sample_logits=True,
                )

                samples = vq_model.decode_code(index_sample, qzshape)  # [-1, 1]
                if args.image_size_eval != args.image_size:
                    samples = F.interpolate(samples, size=(args.image_size_eval, args.image_size_eval), mode='bicubic')

            samples = torch.clamp(127.5 * samples + 128.0, 0, 255)
            samples = samples.permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

            save_samples(samples, out_dir, cid, tag, save_index)
            generated += bs
            save_index += bs

    print("\nDone.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-L")
    parser.add_argument("--gpt-ckpt", type=str, required=True)
    parser.add_argument("--gpt-type", type=str, choices=['c2i'], default="c2i")
    parser.add_argument("--from-fsdp", action="store_true")
    parser.add_argument("--cls-token-num", type=int, default=1)

    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, required=True)
    parser.add_argument("--codebook-size", type=int, default=16384)
    parser.add_argument("--codebook-embed-dim", type=int, default=8)

    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=256)
    parser.add_argument("--image-size-eval", type=int, choices=[256, 384, 512], default=256)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)

    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--cfg-interval", type=float, default=-1)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)

    parser.add_argument("--precision", type=str, default="bf16", choices=["none", "fp16", "bf16"])
    parser.add_argument("--compile", action="store_true", default=False)
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--id2name-file", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "imagenet_id2name.txt"),
                        help="Python dict literal mapping {id: 'alias1, alias2, ...'}")
    parser.add_argument("--class-ids", type=str, default="",
                        help="Optional numeric ids, comma-separated (e.g., '3,7,21').")
    parser.add_argument("--class-names", type=str, default="",
                        help="Comma-separated class names/aliases (e.g., 'golden retriever, tabby').")
    parser.add_argument("--class-list-file", type=str, default="",
                        help="Text file with one class name per line.")
    parser.add_argument("--per-class", type=int, default=5, help="How many samples to generate per class.")

    args = parser.parse_args()
    main(args)

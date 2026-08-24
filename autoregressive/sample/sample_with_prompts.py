import argparse
import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn.functional as F
import torch.distributed as dist

from tqdm import tqdm
from PIL import Image
import numpy as np

from tokenizer.tokenizer_image.vq_model import VQ_models
from autoregressive.models.gpt import GPT_models
from autoregressive.models.generate import generate

from utils.data_helpers import load_id2name_from_txt, canonicalize_alias


def create_npz_from_sample_folder(sample_dir, num=50_000):
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def main(args):
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU."
    torch.set_grad_enabled(False)

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * world + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={world}.")

    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim)
    vq_model.to(device).eval()
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu", weights_only=False)
    vq_model.load_state_dict(checkpoint["model"])
    del checkpoint

    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
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
        raise Exception("please check model weight, maybe add --from-fsdp to run command")
    state = {k: v for k, v in model_weight.items() if ".kv_cache." not in k}
    gpt_model.load_state_dict(state, strict=False)
    gpt_model.eval()
    del checkpoint

    if args.compile:
        print("compiling the model...")
        gpt_model = torch.compile(gpt_model, mode="reduce-overhead", fullgraph=True)
    else:
        print("no model compile")

    model_string_name = args.gpt_model.replace("/", "-")
    if args.from_fsdp:
        ckpt_string_name = args.gpt_ckpt.split('/')[-2]
    else:
        ckpt_string_name = os.path.basename(args.gpt_ckpt).replace(".pth", "").replace(".pt", "")
    folder_name = (
        f"{model_string_name}-{ckpt_string_name}-size-{args.image_size}-size-{args.image_size_eval}-"
        f"{args.vq_model}-topk-{args.top_k}-topp-{args.top_p}-temperature-{args.temperature}-"
        f"cfg-{args.cfg_scale}-seed-{args.global_seed}"
    )
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    assert os.path.exists(args.id2name_txt), f"--id2name-txt not found: {args.id2name_txt}"
    id2name = load_id2name_from_txt(args.id2name_txt)
    if args.num_classes > len(id2name):
        raise ValueError(f"--num_classes ({args.num_classes}) exceeds mapping size ({len(id2name)})")

    cap_tmp = Path(sample_folder_dir) / f"captions.rank{rank}.tsv"
    cap_f = open(cap_tmp, "w", encoding="utf-8")

    n = args.per_proc_batch_size
    global_batch_size = n * world
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % world == 0
    samples_needed_this_gpu = total_samples // world
    assert samples_needed_this_gpu % n == 0
    iterations = samples_needed_this_gpu // n
    pbar = tqdm(range(iterations)) if rank == 0 else range(iterations)
    total = 0

    for _ in pbar:
        c_indices = torch.randint(0, args.num_classes, (n,), device=device)

        qzshape = [len(c_indices), args.codebook_embed_dim, latent_size, latent_size]
        index_sample = generate(
            gpt_model, c_indices, latent_size ** 2,
            cfg_scale=args.cfg_scale, cfg_interval=args.cfg_interval,
            temperature=args.temperature, top_k=args.top_k,
            top_p=args.top_p, sample_logits=True,
        )

        samples = vq_model.decode_code(index_sample, qzshape)  # [-1, 1]
        if args.image_size_eval != args.image_size:
            samples = F.interpolate(samples, size=(args.image_size_eval, args.image_size_eval), mode='bicubic')
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        names = [canonicalize_alias(id2name[int(cid)], policy=args.alias_policy) for cid in c_indices.tolist()]
        prompts = [args.prompt_template.format(name=nm) for nm in names]

        for i, sample in enumerate(samples):
            index = i * world + rank + total
            fn = f"{index:06d}.png"
            Image.fromarray(sample).save(f"{sample_folder_dir}/{fn}")
            cap_f.write(f"{fn}\t{prompts[i]}\n")

        total += global_batch_size

    cap_f.close()

    dist.barrier()

    if rank == 0:
        out_tsv = Path(sample_folder_dir) / "captions.tsv"
        with open(out_tsv, "w", encoding="utf-8") as fout:
            for r in range(world):
                part = Path(sample_folder_dir) / f"captions.rank{r}.tsv"
                with open(part, "r", encoding="utf-8") as fin:
                    for line in fin:
                        fout.write(line)
                os.remove(part)
        print(f"Wrote prompts file: {out_tsv}")

    dist.barrier()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-L")
    parser.add_argument("--gpt-ckpt", type=str, required=True)
    parser.add_argument("--gpt-type", type=str, choices=['c2i'], default="c2i")
    parser.add_argument("--from-fsdp", action='store_true')
    parser.add_argument("--cls-token-num", type=int, default=1)
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"])
    parser.add_argument("--compile", action='store_true', default=False)

    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, required=True)
    parser.add_argument("--codebook-size", type=int, default=16384)
    parser.add_argument("--codebook-embed-dim", type=int, default=8)

    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=256)
    parser.add_argument("--image-size-eval", type=int, choices=[256, 384, 512], default=256)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)

    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale",  type=float, default=1.5)
    parser.add_argument("--cfg-interval", type=float, default=-1)
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50000)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)

    parser.add_argument(
        "--id2name-txt",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "dataset" / "imagenet_id2name.txt"),
    )
    parser.add_argument("--prompt-template", type=str, default="a photo of a {name}")
    parser.add_argument("--alias-policy", type=str, choices=["first", "full"], default="first")

    args = parser.parse_args()
    main(args)

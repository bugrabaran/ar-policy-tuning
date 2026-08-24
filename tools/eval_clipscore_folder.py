import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from utils.reward_utils import CLIPReward

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

def read_captions_tsv(path: Path) -> dict[str, str]:
    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            # expected: "filename<TAB>prompt"
            fname, cap = line.split("\t", 1)
            mapping[fname] = cap
    return mapping

def list_images(folder: Path):
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS])

def pil_to_uint8_tensor(p: Path) -> torch.Tensor:
    im = Image.open(p).convert("RGB")
    arr = np.asarray(im, dtype=np.uint8)  # [H,W,3]
    return torch.from_numpy(arr)

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", type=str, required=True, help="Folder with images and captions.tsv")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--model", type=str, default="ViT-L-14")
    ap.add_argument("--pretrained", type=str, default="openai")
    ap.add_argument("--save-csv", type=str, default="", help="Optional: path to write per-image scores CSV")
    args = ap.parse_args()

    folder = Path(args.folder)
    tsv = folder / "captions.tsv"
    assert tsv.exists(), f"captions.tsv not found in {folder}"

    caps = read_captions_tsv(tsv)
    paths = [p for p in list_images(folder) if p.name in caps]
    assert paths, f"No images matching captions.tsv found in {folder}"

    reward = CLIPReward(device=args.device, model_name=args.model, pretrained=args.pretrained)
    target_hw = getattr(reward, "target_hw", 224)

    import open_clip
    all_scores = []
    names = []
    B = args.batch_size

    for i in tqdm(range(0, len(paths), B), desc="CLIP scoring"):
        batch_paths = paths[i:i+B]
        imgs = [torch.from_numpy(np.array(Image.open(p).convert("RGB").resize((target_hw, target_hw), Image.BICUBIC), dtype=np.uint8)) for p in batch_paths]
        batch = torch.stack(imgs, dim=0)  # [N,H,W,3] uint8

        prompts = [caps[p.name] for p in batch_paths]
        tok = open_clip.tokenize(prompts)

        scores = reward(batch, tok).detach().cpu().tolist()

        all_scores.extend(scores)
        names.extend([p.name for p in batch_paths])

    mean = float(np.mean(all_scores))
    std = float(np.std(all_scores))

    print(f"Images scored: {len(all_scores)}")
    print(f"Average CLIPScore: {mean:.6f}  (std: {std:.6f})")

    if args.save_csv:
        out = Path(args.save_csv)
        import csv
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["filename", "prompt", "clip_score"])
            for nm, sc in zip(names, all_scores):
                w.writerow([nm, caps[nm], f"{sc:.6f}"])
        print(f"Wrote per-image scores to {out}")

if __name__ == "__main__":
    main()

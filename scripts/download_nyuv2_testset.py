import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image


def download_nyuv2(root="data/nyuv2", max_images=None, force=False, val_split=0.1, seed=42):
    root = Path(root)
    image_dir = root / "image"
    depth_dir = root / "depth"

    if not force and (image_dir.exists() and len(list(image_dir.glob("*.jpg"))) > 0):
        existing = len(list(image_dir.glob("*.jpg")))
        print(f"NYUv2 already exists at {root} ({existing} images). Use --force to re-download.")
        return

    try:
        from datasets import load_dataset
    except ImportError:
        print("=" * 60)
        print("Need 'datasets' library. Install with:")
        print("  pip install datasets")
        print("=" * 60)
        return

    image_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading NYUv2 from Hugging Face (jagennath-hari/nyuv2)…")
    ds = load_dataset("jagennath-hari/nyuv2", split="train", streaming=True)

    if max_images is not None:
        ds = ds.take(max_images)

    all_fnames = []
    for i, item in enumerate(ds):
        img = item["rgb"]
        depth = np.array(item["depth"])

        fname = f"nyu_{i:04d}.jpg"
        img.save(str(image_dir / fname), quality=95)

        Image.fromarray(depth).save(str(depth_dir / fname.replace(".jpg", ".png")))
        all_fnames.append(fname)

        if (i + 1) % 25 == 0:
            print(f"  [{i+1}] downloaded")

    if not all_fnames:
        print("No images downloaded.")
        return

    rng = random.Random(seed)
    rng.shuffle(all_fnames)

    split_idx = int(len(all_fnames) * (1 - val_split))
    train_files = all_fnames[:split_idx]
    val_files = all_fnames[split_idx:]

    with open(root / "nyuv2_train.txt", "w") as f:
        for name in train_files:
            f.write(name + "\n")
    with open(root / "nyuv2_val.txt", "w") as f:
        for name in val_files:
            f.write(name + "\n")

    print(f"Done! {len(all_fnames)} images -> {root}")
    print(f"  Images:     {image_dir}/")
    print(f"  Depth maps: {depth_dir}/")
    print(f"  Train:      nyuv2_train.txt ({len(train_files)})")
    print(f"  Val:        nyuv2_val.txt ({len(val_files)})")
    print()
    print("Train with:")
    print(f"  python train.py --data-root {root} --backbone swin_tiny_patch4_window7_224 --epochs 30")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="data/nyuv2")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Limit to N images for quick test")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Fraction of images for validation (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    download_nyuv2(args.root, args.max_images, args.force, args.val_split, args.seed)

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def download_nyuv2_testset(root="data/nyuv2", max_images=None, force=False):
    root = Path(root)
    image_dir = root / "image"
    depth_dir = root / "depth"
    split_file = root / "nyuv2_test.txt"

    if split_file.exists() and not force:
        existing = len(list(image_dir.glob("*.jpg")))
        print(f"NYUv2 test set already exists at {root} ({existing} images). Use --force to re-download.")
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

    print("Downloading NYUv2 from Hugging Face (jagennath-hari/nyuv2)...")
    ds = load_dataset("jagennath-hari/nyuv2", split="train", streaming=True)

    if max_images is not None:
        ds = ds.take(max_images)

    rel_paths = []
    for i, item in enumerate(ds):
        img = item["rgb"]
        depth = np.array(item["depth"])

        img_path = f"nyu_{i:04d}.jpg"
        depth_path = f"nyu_{i:04d}.png"

        img.save(str(image_dir / img_path), quality=95)

        Image.fromarray(depth).save(str(depth_dir / depth_path))

        rel_paths.append(img_path)

        if (i + 1) % 25 == 0:
            print(f"  [{i+1}] downloaded")

    with open(split_file, "w") as f:
        for p in rel_paths:
            f.write(p + "\n")

    print(f"Done! {len(rel_paths)} images -> {root}")
    print(f"  Images: {image_dir}/")
    print(f"  Depth:  {depth_dir}/")
    print(f"  Split:  {split_file}")
    print(f"\nTrain with:")
    print(f"  python train.py --data-root {root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="data/nyuv2")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Limit to N images for quick test")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    download_nyuv2_testset(args.root, args.max_images, args.force)

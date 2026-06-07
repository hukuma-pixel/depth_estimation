import argparse
from pathlib import Path

import numpy as np
from PIL import Image

SPLIT_NAMES = {
    'train': 'train',
    'val':   'val',
    'test':  'test',
}


def download_split(root, split_name, prefix, force):
    import datasets

    root = Path(root)
    image_dir = root / 'image'
    depth_dir = root / 'depth'
    image_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading NYUv2 split='{split_name}' -> prefix '{prefix}' ...")
    ds = datasets.load_dataset("jagennath-hari/nyuv2", split=split_name, streaming=True)

    fnames = []
    for i, item in enumerate(ds):
        fname = f"{prefix}_{i:04d}.jpg"
        img_path = image_dir / fname
        depth_path = depth_dir / fname.replace('.jpg', '.png')

        if not force and img_path.exists():
            fnames.append(fname)
            if (i + 1) % 100 == 0:
                print(f"  [{i+1}] exists, skip")
            continue

        item["rgb"].save(str(img_path), quality=95)
        Image.fromarray(np.array(item["depth"])).save(str(depth_path))
        fnames.append(fname)

        if (i + 1) % 25 == 0:
            print(f"  [{i+1}] downloaded")

    if not fnames:
        print(f"  No images for split='{split_name}'.")
        return []

    return fnames


def download_nyuv2(root="data/nyuv2", splits=None, force=False):
    """
    Download specific splits from HF nyuv2.

    Args:
        splits: list of strings — 'train', 'val', 'test' (default: all three)
        force: re-download even if file exists
    """
    if splits is None:
        splits = ['train', 'val', 'test']

    all_fnames = {}

    for split_name in splits:
        prefix = f"nyu_{split_name}"
        fnames = download_split(root, split_name, prefix, force)
        all_fnames[split_name] = fnames

    root_path = Path(root)

    for split_name in splits:
        fnames = all_fnames.get(split_name, [])
        if not fnames:
            continue
        txt_path = root_path / f'nyuv2_{split_name}.txt'
        with open(txt_path, 'w') as f:
            for name in fnames:
                f.write(name + '\n')
        print(f"  {txt_path.name}: {len(fnames)} images")

    print(f"\nDone! Files in {root_path / 'image/'}")
    print("Train with:")
    print(f"  python train.py --data-root {root}")
    print("Evaluate with:")
    print(f"  python evaluate.py --checkpoint <path> --split test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="data/nyuv2")
    parser.add_argument("--force", action="store_true",
                        help="Re-download existing files")
    parser.add_argument("--all", dest="splits", action="store_const",
                        const=['train', 'val', 'test'],
                        help="Download all three splits (train + val + test)")
    parser.set_defaults(splits=['train', 'val', 'test'])
    args = parser.parse_args()
    download_nyuv2(args.root, args.splits, args.force)

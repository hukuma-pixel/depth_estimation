import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.nyuv2 import NYUv2Dataset
from models.depth_net import MonocularDepthNet
from models.metrics import evaluate_model


BACKBONE_MAP = {
    96: 'swin_tiny_patch4_window7_224',
    128: 'swin_base_patch4_window7_224',
    192: 'swin_large_patch4_window7_224',
}


def _unwrap_state(state):
    if 'model_state_dict' in state:
        return state['model_state_dict']
    return state


def detect_backbone(checkpoint_path):
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    state = _unwrap_state(state)
    pe_weight = state.get('encoder.patch_embed.proj.weight')
    if pe_weight is not None:
        embed_dim = pe_weight.shape[0]
        return BACKBONE_MAP.get(embed_dim, None)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data-root', type=str, default='data/nyuv2')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--input-size', type=int, default=192)
    parser.add_argument('--max-depth', type=float, default=10.0)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    backbone = detect_backbone(args.checkpoint)
    if backbone is None:
        print("Could not detect backbone, falling back to swin_tiny")
        backbone = 'swin_tiny_patch4_window7_224'
    print(f"Detected backbone: {backbone}")

    model = MonocularDepthNet(swin_version=backbone, pretrained=False).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(_unwrap_state(state))
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    dataset = NYUv2Dataset(args.data_root, args.split,
                            input_size=(args.input_size, args.input_size),
                            max_depth=args.max_depth)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=0, pin_memory=False)
    print(f"Split: {args.split} ({len(dataset)} images)")

    metrics = evaluate_model(model, loader, device, max_depth=args.max_depth)

    print()
    print(f"{'Metric':>10} | {'Value':>8}")
    print("-" * 22)
    for key in ['abs_rel', 'rmse', 'silog', 'd1', 'd2', 'd3']:
        print(f"{key:>10} | {metrics[key]:>8.4f}")


if __name__ == '__main__':
    main()

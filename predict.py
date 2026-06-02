import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models.depth_net import MonocularDepthNet


def load_image(path, input_size=480):
    img = Image.open(path).convert('RGB')
    orig_size = img.size
    transform = transforms.Compose([
        transforms.Resize(input_size, interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(img).unsqueeze(0)
    return tensor, orig_size


def save_depth_map(depth_tensor, output_path, colormap=True):
    depth = depth_tensor.squeeze().cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth_img = Image.fromarray((depth * 255).astype('uint8'), mode='L')
    if colormap:
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            cm = plt.get_cmap('inferno')
            colored = (cm(depth)[:, :, :3] * 255).astype('uint8')
            depth_img = Image.fromarray(colored, mode='RGB')
        except ImportError:
            pass
    depth_img.save(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str, help='Path to input image')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model weights')
    parser.add_argument('--backbone', type=str, default='swin_tiny_patch4_window7_224')
    parser.add_argument('--input-size', type=int, default=480)
    parser.add_argument('--output', type=str, default=None, help='Path to save depth map')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    output_path = Path(args.output or f'depth_{Path(args.image_path).stem}.png')

    model = MonocularDepthNet(swin_version=args.backbone, pretrained=(args.checkpoint is None)).to(device)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state)
    model.eval()

    img_tensor, orig_size = load_image(args.image_path, args.input_size)
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        _, _, d1 = model(img_tensor)
        d1 = F.interpolate(d1, size=(orig_size[1], orig_size[0]), mode='bilinear', align_corners=False)

    save_depth_map(d1, output_path)
    print(f"Depth map saved to {output_path}")


if __name__ == '__main__':
    main()

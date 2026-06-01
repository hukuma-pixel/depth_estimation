import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from models.depth_net import MonocularDepthNet
from losses.depth_loss import scale_invariant_log_loss, normal_loss


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for imgs, depths in loader:
        imgs, depths = imgs.to(device), depths.to(device)
        optimizer.zero_grad()

        d3, d2, d1 = model(imgs)

        gt1 = F.interpolate(depths, size=d1.shape[-2:], mode='bilinear', align_corners=False)
        gt2 = F.interpolate(depths, size=d2.shape[-2:], mode='bilinear', align_corners=False)
        gt3 = F.interpolate(depths, size=d3.shape[-2:], mode='bilinear', align_corners=False)

        l_depth = scale_invariant_log_loss(d1, gt1) + \
                  0.5 * scale_invariant_log_loss(d2, gt2) + \
                  0.25 * scale_invariant_log_loss(d3, gt3)
        l_norm = normal_loss(d1.squeeze(1), gt1.squeeze(1))
        loss = l_depth + l_norm

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    input_size = 224
    dummy_imgs = torch.randn(4, 3, input_size, input_size).to(device)
    dummy_depths = (torch.rand(4, 1, input_size, input_size) * 8.0 + 1.0).to(device)
    loader = DataLoader(TensorDataset(dummy_imgs, dummy_depths), batch_size=2, shuffle=True)

    model = MonocularDepthNet(pretrained=False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4)

    for epoch in range(3):
        loss = train_epoch(model, loader, optimizer, device)
        print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")

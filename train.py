import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models.hafm import HAFM
from models.dprm import DPRM
from losses.depth_loss import scale_invariant_log_loss, normal_loss


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for imgs, depths in loader:
        imgs, depths = imgs.to(device), depths.to(device)
        optimizer.zero_grad()

        # Forward
        d3, d2, d1 = model(imgs)


        # Loss (используем только финальную карту d1 для старта)
        l_depth = scale_invariant_log_loss(d1, depths)
        l_norm = normal_loss(d1.squeeze(1), depths.squeeze(1))
        loss = l_depth + l_norm

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Заглушка данных (замените на DataLoader NYU-V2/KITTI)
    dummy_imgs = torch.randn(4, 3, 256, 256).to(device)
    dummy_depths = torch.rand(4, 1, 256, 256).to(device) * 10.0
    loader = DataLoader(TensorDataset(dummy_imgs, dummy_depths), batch_size=2, shuffle=True)

    # Модель (упрощённая инициализация для проверки pipeline)
    # В реальности используйте MonocularDepthNet с features_only=True
    model = nn.Module()  # Замените на реальную архитектуру
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4)

    for epoch in range(5):
        loss = train_epoch(model, loader, optimizer, device)
        print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")
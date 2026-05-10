import torch
import torch.nn as nn
from timm.models import swin_transformer
from .hafm import HAFM
from .dprm import DPRM


class MonocularDepthNet(nn.Module):
    def __init__(self, swin_version='swin_tiny_patch4_window7_224', C=192):
        super().__init__()
        # Encoder
        self.encoder = swin_transformer(swin_version, pretrained=True, num_classes=0)
        # Удаляем head, оставляем только backbone
        self.encoder.head = nn.Identity()

        # Decoder: HAFM для 3 стадий (1/16, 1/8, 1/4)
        # Swin выдаёт каналы: [C, 2C, 4C, 8C]. Используем 4C, 2C, C для фузии
        self.hafm3 = HAFM(4 * C)
        self.hafm2 = HAFM(2 * C)
        self.hafm1 = HAFM(C)

        self.dprm = DPRM(channels=[4 * C, 2 * C, C])

    def forward(self, x):
        # Извлекаем промежуточные признаки Swin
        features = self.encoder.forward_features(x)
        # Swin в timm возвращает dict или tuple. Адаптируем под 4 стадии:
        f_enc1, f_enc2, f_enc3, f_enc4 = features[0], features[1], features[2], features[3]
        # Приводим к нужным размерностям (B, C, H, W)
        # В timm Swin возвращает BxNxC, нужно reshape
        b, _, _ = f_enc4.shape
        f_enc4 = f_enc4.permute(0, 2, 1).reshape(b, -1, f_enc4.shape[-1])  # упрощённо

        # Для точного соответствия лучше использовать feature extraction hooks.
        # Ниже упрощённый псевдо-код для наглядности структуры:
        # f_enc4 -> downsampled 1/32
        # f_enc3 -> 1/16
        # f_enc2 -> 1/8
        # f_enc1 -> 1/4

        # Примерный pipeline (требует адаптации под реальные shape Swin):
        # f_up4 = ... (init)
        # f_dec3 = self.hafm3(f_enc3, f_up4)
        # f_dec2 = self.hafm2(f_enc2, f_up3)
        # f_dec1 = self.hafm1(f_enc1, f_up2)
        # d3, d2, d1 = self.dprm(f_dec3, f_dec2, f_dec1)
        # return d3, d2, d1
        pass
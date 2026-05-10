import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y_avg = self.mlp(self.avg_pool(x).view(b, c))
        y_max = self.mlp(self.max_pool(x).view(b, c))
        return (y_avg + y_max).view(b, c, 1, 1)


class LRFB(nn.Module):
    """Local Residual Fusion Block (CNN-based)"""

    def __init__(self, channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(channels * 2, channels, 1)
        self.residual = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )
        self.ca = ChannelAttention(channels)

    def forward(self, f_up, f_enc):
        x = self.conv1x1(torch.cat([f_up, f_enc], dim=1))
        res = self.residual(x)
        return x + res * self.ca(res)


class GAFB(nn.Module):
    """Global Attention Fusion Block (Transformer-based)
    Примечание: В статье используется BiFormer (BRCA/BRSA).
    Для старта используется стандартное MultiheadAttention.
    Замените на BiFormer при необходимости точного воспроизведения.
    """

    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm_enc = nn.LayerNorm(channels)
        self.norm_up = nn.LayerNorm(channels)
        self.cross_attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.self_attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels))

    def forward(self, f_enc, f_up):
        b, c, h, w = f_enc.shape
        f_enc_flat = f_enc.permute(0, 2, 3, 1).reshape(b, h * w, c)
        f_up_flat = f_up.permute(0, 2, 3, 1).reshape(b, h * w, c)

        q = self.norm_up(f_up_flat)
        kv = self.norm_enc(f_enc_flat)
        cross_out, _ = self.cross_attn(q, kv, kv)

        self_out, _ = self.self_attn(cross_out + q, cross_out + q, cross_out + q)
        f_out = self.norm_up(self_out) + self.ffn(self_out)
        return f_out.permute(0, 2, 1).reshape(b, c, h, w) + f_up


class SFB(nn.Module):
    """Selective Fusion Block"""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 2, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, f_local, f_global):
        sa = self.conv(torch.cat([f_local, f_global], dim=1))
        return sa[:, 0:1] * f_local + sa[:, 1:2] * f_global


class HAFM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.lrfb = LRFB(channels)
        self.gafb = GAFB(channels)
        self.sfb = SFB(channels)

    def forward(self, f_enc, f_up):
        return self.sfb(self.lrfb(f_up, f_enc), self.gafb(f_enc, f_up))
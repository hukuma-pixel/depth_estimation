import torch
import torch.nn as nn
from .biformer import BiLevelRoutingAttention


class ChannelAttention(nn.Module):
    """CBAM-style channel attention. MLP без Sigmoid — Sigmoid после суммы avg+max."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y_avg = self.mlp(self.avg_pool(x).view(b, c))
        y_max = self.mlp(self.max_pool(x).view(b, c))
        return torch.sigmoid(y_avg + y_max).view(b, c, 1, 1)


class LRFB(nn.Module):
    """Local Residual Fusion Block (CNN-based)"""

    def __init__(self, channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(channels * 2, channels, 1)
        self.residual = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.ca = ChannelAttention(channels)

    def forward(self, f_up, f_enc):
        x = self.conv1x1(torch.cat([f_up, f_enc], dim=1))
        res = self.residual(x)
        return x + res * self.ca(res)


class GAFB(nn.Module):
    """Global Attention Fusion Block (Transformer-based)

    Реализован как pre-norm Transformer:
      cross-attn (f_up -> q, f_enc -> kv) -> self-attn -> FFN,
    с LayerNorm перед каждым блоком и residual-соединениями.

    При use_biformer=True использует BiFormer (BRCA/BRSA) вместо
    стандартного MultiheadAttention.
    """

    def __init__(self, channels, num_heads=4, use_biformer=False, sr_ratio=2, topk=4):
        super().__init__()
        self.use_biformer = use_biformer
        self.norm_cross_q = nn.LayerNorm(channels)
        self.norm_cross_kv = nn.LayerNorm(channels)

        attn_cls = BiLevelRoutingAttention if use_biformer else nn.MultiheadAttention
        attn_kw = dict(dim=channels, num_heads=num_heads, sr_ratio=sr_ratio, topk=topk) if use_biformer else dict(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.cross_attn = attn_cls(**attn_kw)
        self.self_attn = attn_cls(**attn_kw)

        self.norm_self = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )
        self.norm_ffn = nn.LayerNorm(channels)

    def _run_attn(self, attn, q, k, v):
        if self.use_biformer:
            return attn(q, k, v)
        out, _ = attn(q, k, v)
        return out

    def forward(self, f_enc, f_up):
        B, C, H, W = f_enc.shape
        f_enc_flat = f_enc.permute(0, 2, 3, 1).reshape(B, H * W, C)
        f_up_flat = f_up.permute(0, 2, 3, 1).reshape(B, H * W, C)

        q = self.norm_cross_q(f_up_flat)
        kv = self.norm_cross_kv(f_enc_flat)
        cross_out = self._run_attn(self.cross_attn, q, kv, kv)
        x = cross_out + f_up_flat

        x_norm = self.norm_self(x)
        self_out = self._run_attn(self.self_attn, x_norm, x_norm, x_norm)
        x = self_out + x

        x = self.ffn(self.norm_ffn(x)) + x

        return x.permute(0, 2, 1).reshape(B, C, H, W) + f_up



class HAFM(nn.Module):
    """Hybrid Attention Fusion Module.

    Объединяет локальную (LRFB) и глобальную (GAFB) ветви через
    селективный гейт (SFB). f_enc — признак энкодера, f_up —
    апсемпленный признак с предыдущего уровня декодера.
    """

    def __init__(self, channels, use_biformer=False):
        super().__init__()
        self.lrfb = LRFB(channels)
        self.gafb = GAFB(channels, use_biformer=use_biformer)
        self.sfb = SFB(channels)

    def forward(self, f_enc, f_up):
        return self.sfb(self.lrfb(f_up, f_enc), self.gafb(f_enc, f_up))


class SFB(nn.Module):
    """Selective Fusion Block — пространственный гейт для
    взвешенного слияния локального (LRFB) и глобального (GAFB) путей."""

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


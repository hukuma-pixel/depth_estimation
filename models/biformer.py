import torch
import torch.nn as nn


class BiLevelRoutingAttention(nn.Module):
    """Bi-Level Routing Attention (BRCA/BRSA).

    Complexity: O(S⁴ + N·topk·(HW/S²)) instead of O(N²).
    S = sr_ratio (number of regions per spatial dim).
    """

    def __init__(self, dim, num_heads=4, sr_ratio=2, topk=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.sr_ratio = sr_ratio
        self.topk = topk

        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, dim * 2)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, k=None, v=None):
        B, N, C = x.shape
        H = W = int(N ** 0.5)

        if k is None:
            k = x
        if v is None:
            v = k

        S = self.sr_ratio
        K = self.topk

        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv_proj(k).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        N_r = (H // S) * (W // S)
        N_w = S * S

        q_r = q.reshape(B, self.num_heads, N_r, N_w, self.head_dim).mean(dim=3)
        k_r = k.reshape(B, self.num_heads, N_r, N_w, self.head_dim).mean(dim=3)

        region_affinity = torch.einsum('bhid,bhjd->bhij', q_r, k_r) * self.scale
        _, topk_idx = torch.topk(region_affinity, K, dim=-1)

        k_rgn = k.reshape(B, self.num_heads, N_r, N_w, self.head_dim)
        v_rgn = v.reshape(B, self.num_heads, N_r, N_w, self.head_dim)

        k_rgn_exp = k_rgn.unsqueeze(2).expand(-1, -1, N_r, -1, -1, -1)
        v_rgn_exp = v_rgn.unsqueeze(2).expand(-1, -1, N_r, -1, -1, -1)

        idx = topk_idx.unsqueeze(-1).unsqueeze(-1)
        idx = idx.expand(-1, -1, -1, -1, N_w, self.head_dim)

        k_sel = k_rgn_exp.gather(3, idx)
        v_sel = v_rgn_exp.gather(3, idx)

        k_attn = k_sel.reshape(B, self.num_heads, N_r, K * N_w, self.head_dim)
        v_attn = v_sel.reshape(B, self.num_heads, N_r, K * N_w, self.head_dim)

        q_rgn = q.reshape(B, self.num_heads, N_r, N_w, self.head_dim)

        attn = torch.einsum('bhnwd,bhnsd->bhwns', q_rgn, k_attn) * self.scale
        attn = attn.softmax(dim=-1)

        out = torch.einsum('bhwns,bhnsd->bhnwd', attn, v_attn)
        out = out.reshape(B, self.num_heads, N, self.head_dim)
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)
        out = self.out_proj(out)
        return out

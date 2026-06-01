import pytest
import torch

from models.depth_net import MonocularDepthNet, DecoderUpsample


def test_decoder_upsample():
    B, C_in, H, W = 2, 768, 14, 14
    C_out = 384
    x = torch.randn(B, C_in, H, W)
    up = DecoderUpsample(C_in, C_out)
    out = up(x)
    assert out.shape == (B, C_out, H * 2, W * 2), f"Expected {(B, C_out, H*2, W*2)}, got {out.shape}"


@pytest.mark.slow
def test_depth_net_forward():
    model = MonocularDepthNet(swin_version='swin_large_patch4_window7_224', C=192, pretrained=False)
    model.eval()
    B, H, W = 2, 224, 224
    x = torch.randn(B, 3, H, W)
    with torch.no_grad():
        d3, d2, d1 = model(x)
    assert d3.shape == (B, 1, H // 8, W // 8), f"d3: {d3.shape}"
    assert d2.shape == (B, 1, H // 4, W // 4), f"d2: {d2.shape}"
    assert d1.shape == (B, 1, H // 2, W // 2), f"d1: {d1.shape}"


@pytest.mark.slow
def test_depth_net_gradients():
    model = MonocularDepthNet(swin_version='swin_large_patch4_window7_224', C=192, pretrained=False)
    model.train()
    B, H, W = 2, 128, 128
    x = torch.randn(B, 3, H, W)
    d3, d2, d1 = model(x)
    loss = d1.mean() + d2.mean() + d3.mean()
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient None for {name}"

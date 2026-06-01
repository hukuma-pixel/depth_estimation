import pytest
import torch

from models.dprm import DPRM, UpsamplingBlock, VerticalBlock, PredictionHead


def test_upsampling_block_shape():
    B, C_in, H, W = 2, 768, 16, 16
    C_out = 384
    x = torch.randn(B, C_in, H, W)
    block = UpsamplingBlock(C_in, C_out)
    out = block(x)
    assert out.shape == (B, C_out, H * 2, W * 2), f"Expected {(B, C_out, H*2, W*2)}, got {out.shape}"


def test_upsampling_block_runtime_shapes():
    pairs = [(768, 384), (384, 192), (192, 192)]
    for C_in, C_out in pairs:
        block = UpsamplingBlock(C_in, C_out)
        x = torch.randn(1, C_in, 14, 14)
        out = block(x)
        assert out.shape == (1, C_out, 28, 28), f"{C_in}->{C_out}: {out.shape}"


def test_prediction_head_shape():
    B, C, H, W = 2, 384, 32, 32
    x = torch.randn(B, C, H, W)
    head = PredictionHead(C)
    out = head(x)
    assert out.shape == (B, 1, H, W)


def test_vertical_block_shape():
    B, H, W = 2, 32, 32
    out_ch = 192
    x = torch.randn(B, 1, H, W)
    vblock = VerticalBlock(out_ch)
    out = vblock(x)
    assert out.shape == (B, out_ch, H * 2, W * 2), f"Expected {(B, out_ch, H*2, W*2)}, got {out.shape}"


def test_dprm_forward_shapes():
    C = 192
    B = 2
    H, W = 256, 256
    f3 = torch.randn(B, 4 * C, H // 16, W // 16)
    f2 = torch.randn(B, 2 * C, H // 8, W // 8)
    f1 = torch.randn(B, C, H // 4, W // 4)

    model = DPRM(channels=[4 * C, 2 * C, C])
    d3, d2, d1 = model(f3, f2, f1)

    assert d3.shape == (B, 1, H // 8, W // 8), f"d3: {d3.shape}"
    assert d2.shape == (B, 1, H // 4, W // 4), f"d2: {d2.shape}"
    assert d1.shape == (B, 1, H // 2, W // 2), f"d1: {d1.shape}"


def test_dprm_backward():
    C = 192
    B = 2
    H, W = 128, 128
    f3 = torch.randn(B, 4 * C, H // 16, W // 16)
    f2 = torch.randn(B, 2 * C, H // 8, W // 8)
    f1 = torch.randn(B, C, H // 4, W // 4)

    model = DPRM(channels=[4 * C, 2 * C, C])
    d3, d2, d1 = model(f3, f2, f1)
    loss = d1.mean() + d2.mean() + d3.mean()
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradient is None for {name}"
        assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"


def test_dprm_monotonic_resolution():
    C = 192
    B = 1
    H, W = 256, 256
    f3 = torch.randn(B, 4 * C, H // 16, W // 16)
    f2 = torch.randn(B, 2 * C, H // 8, W // 8)
    f1 = torch.randn(B, C, H // 4, W // 4)

    model = DPRM(channels=[4 * C, 2 * C, C])
    d3, d2, d1 = model(f3, f2, f1)

    assert d3.shape[-1] < d2.shape[-1] < d1.shape[-1], \
        f"Resolutions should be monotonic: d3={d3.shape[-1]}, d2={d2.shape[-1]}, d1={d1.shape[-1]}"


@pytest.mark.parametrize("C", [96, 128, 192])
def test_dprm_various_backbones(C):
    B = 2
    H, W = 256, 256
    f3 = torch.randn(B, 4 * C, H // 16, W // 16)
    f2 = torch.randn(B, 2 * C, H // 8, W // 8)
    f1 = torch.randn(B, C, H // 4, W // 4)

    model = DPRM(channels=[4 * C, 2 * C, C])
    d3, d2, d1 = model(f3, f2, f1)

    assert d3.shape == (B, 1, H // 8, W // 8), f"C={C} d3: {d3.shape}"
    assert d2.shape == (B, 1, H // 4, W // 4), f"C={C} d2: {d2.shape}"
    assert d1.shape == (B, 1, H // 2, W // 2), f"C={C} d1: {d1.shape}"

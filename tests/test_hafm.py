import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.hafm import ChannelAttention, LRFB, GAFB, SFB, HAFM
from models.depth_net import MonocularDepthNet


B, C, H, W = 2, 96, 56, 56


def test_channel_attention():
    m = ChannelAttention(C)
    x = torch.randn(B, C, H, W)
    out = m(x)
    assert out.shape == (B, C, 1, 1), f'CA shape: {out.shape}'
    assert (out >= 0).all() and (out <= 1).all(), 'CA must output values in [0, 1]'


def test_lrfb():
    m = LRFB(C)
    f_up = torch.randn(B, C, H, W)
    f_enc = torch.randn(B, C, H, W)
    out = m(f_up, f_enc)
    assert out.shape == (B, C, H, W), f'LRFB shape: {out.shape}'


def test_gafb():
    m = GAFB(C, num_heads=4)
    f_enc = torch.randn(B, C, H, W)
    f_up = torch.randn(B, C, H, W)
    out = m(f_enc, f_up)
    assert out.shape == (B, C, H, W), f'GAFB shape: {out.shape}'


def test_gafb_different_resolutions():
    m = GAFB(192, num_heads=4)
    f_enc = torch.randn(B, 192, 28, 28)
    f_up = torch.randn(B, 192, 28, 28)
    out = m(f_enc, f_up)
    assert out.shape == (B, 192, 28, 28), f'GAFB diff res shape: {out.shape}'


def test_sfb():
    m = SFB(C)
    f_local = torch.randn(B, C, H, W)
    f_global = torch.randn(B, C, H, W)
    out = m(f_local, f_global)
    assert out.shape == (B, C, H, W), f'SFB shape: {out.shape}'


def test_hafm():
    m = HAFM(C)
    f_enc = torch.randn(B, C, H, W)
    f_up = torch.randn(B, C, H, W)
    out = m(f_enc, f_up)
    assert out.shape == (B, C, H, W), f'HAFM shape: {out.shape}'


def test_hafm_multi_scale():
    configs = [
        (96, 56, 56),
        (192, 28, 28),
        (384, 14, 14),
    ]
    for c, h, w in configs:
        m = HAFM(c)
        f_enc = torch.randn(B, c, h, w)
        f_up = torch.randn(B, c, h, w)
        out = m(f_enc, f_up)
        assert out.shape == (B, c, h, w), f'HAFM ({c},{h},{w}) shape: {out.shape}'


def test_depth_net_default():
    m = MonocularDepthNet(pretrained=False)
    x = torch.randn(1, 3, 224, 224)
    d3, d2, d1 = m(x)
    assert d3.shape[-2:] == (28, 28), f'd3 spatial: {d3.shape}'
    assert d2.shape[-2:] == (56, 56), f'd2 spatial: {d2.shape}'
    assert d1.shape[-2:] == (112, 112), f'd1 spatial: {d1.shape}'
    assert d3.shape[1] == 1, f'd3 channels: {d3.shape}'
    assert d2.shape[1] == 1, f'd2 channels: {d2.shape}'
    assert d1.shape[1] == 1, f'd1 channels: {d1.shape}'


if __name__ == '__main__':
    test_channel_attention()
    print('[PASS] test_channel_attention')
    test_lrfb()
    print('[PASS] test_lrfb')
    test_gafb()
    print('[PASS] test_gafb')
    test_gafb_different_resolutions()
    print('[PASS] test_gafb_different_resolutions')
    test_sfb()
    print('[PASS] test_sfb')
    test_hafm()
    print('[PASS] test_hafm')
    test_hafm_multi_scale()
    print('[PASS] test_hafm_multi_scale')
    test_depth_net_default()
    print('[PASS] test_depth_net_default')
    print('All tests passed!')

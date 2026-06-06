import math
import torch.nn as nn
import torch.nn.functional as F
import timm
from .hafm import HAFM
from .dprm import DPRM


class DecoderUpsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return self.conv(x)


class MonocularDepthNet(nn.Module):
    def __init__(self, swin_version='swin_tiny_patch4_window7_224', pretrained=True):
        super().__init__()
        self.encoder = timm.create_model(swin_version, pretrained=pretrained, features_only=True)
        self.encoder.patch_embed.strict_img_size = False
        self.swin_stride = 224

        chs = self.encoder.feature_info.channels()
        _, C1, C2, C3 = chs

        self.up4 = DecoderUpsample(C3, C2)
        self.up3 = DecoderUpsample(C2, C1)
        self.up2 = DecoderUpsample(C1, chs[0])

        self.hafm3 = HAFM(C2)
        self.hafm2 = HAFM(C1)
        self.hafm1 = HAFM(chs[0])

        self.dprm = DPRM(channels=[C2, C1, chs[0]])

    def _pad_to_swin(self, x):
        H, W = x.shape[2], x.shape[3]
        stride = self.swin_stride
        pH = math.ceil(H / stride) * stride
        pW = math.ceil(W / stride) * stride
        if pH == H and pW == W:
            return x, H, W
        pad_h = pH - H
        pad_w = pW - W
        return F.pad(x, (0, pad_w, 0, pad_h)), H, W

    def enable_grad_checkpointing(self):
        for module in self.encoder.modules():
            if hasattr(module, 'grad_checkpointing'):
                module.grad_checkpointing = True

    def forward(self, x):
        x, H, W = self._pad_to_swin(x)

        features = self.encoder(x)
        features = [f.permute(0, 3, 1, 2).contiguous() for f in features]
        f1, f2, f3, f4 = features

        f_up4 = self.up4(f4)
        f_dec3 = self.hafm3(f3, f_up4)

        f_up3 = self.up3(f_dec3)
        f_dec2 = self.hafm2(f2, f_up3)

        f_up2 = self.up2(f_dec2)
        f_dec1 = self.hafm1(f1, f_up2)

        d3, d2, d1 = self.dprm(f_dec3, f_dec2, f_dec1)

        d3 = d3[:, :, :H // 8, :W // 8]
        d2 = d2[:, :, :H // 4, :W // 4]
        d1 = d1[:, :, :H // 2, :W // 2]
        return d3, d2, d1

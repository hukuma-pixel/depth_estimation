import torch.nn as nn
import torch.nn.functional as F


class UpsamplingBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * 4, 3, padding=1)
        self.shuffle = nn.PixelShuffle(2)
        self.conv1x1 = nn.Conv2d(out_ch, out_ch, 1)

    def forward(self, x):
        return self.conv1x1(self.shuffle(self.conv(x)))


class PredictionHead(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, 1, 3, padding=1),
            nn.Softplus()
        )

    def forward(self, x):
        return self.head(x)


class VerticalBlock(nn.Module):
    def __init__(self, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(1, out_ch, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, depth_map):
        depth_up = F.interpolate(depth_map, scale_factor=2, mode='bilinear', align_corners=False)
        return self.act(self.bn(self.conv(depth_up)))


class DPRM(nn.Module):
    def __init__(self, channels=[768, 384, 192]):
        super().__init__()
        # Stage 3 (coarsest): 1/16 -> 1/8
        self.ub3 = UpsamplingBlock(channels[0], channels[1])
        self.ph3 = PredictionHead(channels[1])
        self.vb3 = VerticalBlock(channels[2])

        # Stage 2 (medium): 1/8 -> 1/4
        self.ub2 = UpsamplingBlock(channels[1], channels[2])
        self.ph2 = PredictionHead(channels[2])
        self.vb2 = VerticalBlock(channels[2])

        # Stage 1 (finest): 1/4 -> 1/2
        self.ub1 = UpsamplingBlock(channels[2], channels[2])
        self.ph1 = PredictionHead(channels[2])

    def forward(self, f3, f2, f1):
        ub3_out = self.ub3(f3)
        d3 = self.ph3(ub3_out)

        f2t = ub3_out + f2
        d2 = self.ph2(self.ub2(f2t) + self.vb3(d3))

        f1t = self.ub2(f2) + f1
        d1 = self.ph1(self.ub1(f1t) + self.vb2(d2))

        return d3, d2, d1

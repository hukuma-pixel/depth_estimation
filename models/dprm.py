import torch.nn as nn

class UpsamplingBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.shuffle = nn.Sequential(
            nn.PixelShuffle(2),
            nn.Conv2d(out_ch // 4, out_ch, 1)
        )
    def forward(self, x):
        return self.shuffle(self.conv(x))

class PredictionHead(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.head = nn.Conv2d(in_ch, 1, 3, padding=1)
    def forward(self, x):
        return self.head(x)

class VerticalBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(1, out_ch, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, depth_map):
        return self.act(self.bn(self.conv(depth_map)))

class DPRM(nn.Module):
    def __init__(self, channels=[192, 96, 48]):
        super().__init__()
        self.ub3 = UpsamplingBlock(channels[0], channels[0]//4)
        self.ph3 = PredictionHead(channels[0]//4)
        self.vb3 = VerticalBlock(1, channels[1])

        self.ub2 = UpsamplingBlock(channels[1], channels[1]//4)
        self.ph2 = PredictionHead(channels[1]//4)
        self.vb2 = VerticalBlock(1, channels[2])

        self.ub1 = UpsamplingBlock(channels[2], channels[2]//4)
        self.ph1 = PredictionHead(channels[2]//4)

    def forward(self, f3, f2, f1):
        # Eq. 7
        d3 = self.ph3(self.ub3(f3))
        # Eq. 8-9
        f2t = self.ub3(f3) + f2
        d2 = self.ph2(self.ub2(f2t) + self.vb3(d3))
        # Eq. 10-11
        f1t = self.ub2(f2) + f1
        d1 = self.ph1(self.ub1(f1t) + self.vb2(d2))
        return d3, d2, d1
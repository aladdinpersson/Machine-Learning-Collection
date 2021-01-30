import torch
import torch.nn as nn
import torch.nn.functional as F


class Inception(nn.Module):
    def __init__(
        self, in_channels, out1x1, out3x3reduced, out3x3, out5x5reduced, out5x5, outpool
    ):
        super().__init__()

        self.branch_1 = BasicConv2d(in_channels, out1x1, kernel_size=1, stride=1)

        self.branch_2 = nn.Sequential(
            BasicConv2d(in_channels, out3x3reduced, kernel_size=1),
            BasicConv2d(out3x3reduced, out3x3, kernel_size=3, padding=1),
        )

        # Is in the original googLeNet paper 5x5 conv but in Inception_v2 it has shown to be
        # more efficient if you instead do two 3x3 convs which is what I am doing here!
        self.branch_3 = nn.Sequential(
            BasicConv2d(in_channels, out5x5reduced, kernel_size=1),
            BasicConv2d(out5x5reduced, out5x5, kernel_size=3, padding=1),
            BasicConv2d(out5x5, out5x5, kernel_size=3, padding=1),
        )

        self.branch_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, outpool, kernel_size=1),
        )

    def forward(self, x):
        y1 = self.branch_1(x)
        y2 = self.branch_2(x)
        y3 = self.branch_3(x)
        y4 = self.branch_4(x)

        return torch.cat([y1, y2, y3, y4], 1)


class GoogLeNet(nn.Module):
    def __init__(self, img_channel):
        super().__init__()

        self.first_layers = nn.Sequential(
            BasicConv2d(img_channel, 192, kernel_size=3, padding=1)
        )

        self._3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self._3b = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self._4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self._4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self._4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self._4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self._4e = Inception(528, 256, 160, 320, 32, 128, 128)

        self._5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self._5b = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.linear = nn.Linear(1024, 10)

    def forward(self, x):
        out = self.first_layers(x)

        out = self._3a(out)
        out = self._3b(out)
        out = self.maxpool(out)

        out = self._4a(out)
        out = self._4b(out)
        out = self._4c(out)
        out = self._4d(out)
        out = self._4e(out)
        out = self.maxpool(out)

        out = self._5a(out)
        out = self._5b(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


def test():
    net = GoogLeNet(1)
    x = torch.randn(3, 1, 32, 32)
    y = net(x)
    print(y.size())


# test()

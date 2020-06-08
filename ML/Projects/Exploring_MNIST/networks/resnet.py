import torch
import torch.nn as nn


class residual_template(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, identity_downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(
            out_channels, out_channels * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.identity_downsample is not None:
            residual = self.identity_downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, residual_template, layers, image_channel, num_classes=10):
        self.in_channels = 64
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=image_channel,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(
            residual_template, layers[0], channels=64, stride=1
        )
        self.layer2 = self._make_layer(
            residual_template, layers[1], channels=128, stride=2
        )
        self.layer3 = self._make_layer(
            residual_template, layers[2], channels=256, stride=2
        )
        self.layer4 = self._make_layer(
            residual_template, layers[3], channels=512, stride=2
        )
        self.avgpool = nn.AvgPool2d(kernel_size=4, stride=1)
        self.fc = nn.Linear(512 * residual_template.expansion, num_classes)

        # initialize weights for conv layers, batch layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, residual_template, num_residuals_blocks, channels, stride):
        identity_downsample = None

        if stride != 1 or self.in_channels != channels * residual_template.expansion:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    channels * residual_template.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(channels * residual_template.expansion),
            )

        layers = []
        layers.append(
            residual_template(self.in_channels, channels, stride, identity_downsample)
        )
        self.in_channels = channels * residual_template.expansion

        for i in range(1, num_residuals_blocks):
            layers.append(residual_template(self.in_channels, channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def ResNet50(img_channel):
    return ResNet(residual_template, [3, 4, 6, 3], img_channel)


def ResNet101(img_channel):
    return ResNet(residual_template, [3, 4, 23, 3], img_channel)


def ResNet152(img_channel):
    return ResNet(residual_template, [3, 8, 36, 3], img_channel)


def test():
    net = ResNet152(img_channel=1)
    y = net(torch.randn(64, 1, 32, 32))
    print(y.size())


# test()

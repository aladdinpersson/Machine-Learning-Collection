import torch
import torch.nn as nn


VGG_types = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "VGG19": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


class VGG(nn.Module):
    def __init__(
        self, vgg_type, in_channels, init_weights=True, batch_norm=True, num_classes=10
    ):
        super().__init__()

        self.batch_norm = batch_norm
        self.in_channels = in_channels

        self.layout = self.create_architecture(VGG_types[vgg_type])
        self.fc = nn.Linear(512, num_classes)

        # self.fcs = nn.Sequential(
        #     nn.Linear(512* 1 * 1, 4096),
        #     nn.ReLU(inplace = False),
        #     nn.Dropout(),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(inplace = False),
        #     nn.Dropout(),
        #     nn.Linear(4096, num_classes),
        # )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        out = self.layout(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def create_architecture(self, architecture):
        layers = []

        for x in architecture:
            if type(x) == int:
                out_channels = x

                conv2d = nn.Conv2d(
                    self.in_channels, out_channels, kernel_size=3, padding=1
                )

                if self.batch_norm:
                    layers += [
                        conv2d,
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=False),
                    ]
                else:
                    layers += [conv2d, nn.ReLU(inplace=False)]

                self.in_channels = out_channels

            elif x == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]

        return nn.Sequential(*layers)


def test():
    net = VGG("VGG16", 1)
    x = torch.randn(64, 1, 32, 32)
    y = net(x)
    print(y.size())


# test()

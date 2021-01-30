import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self, in_channels, init_weights=True, num_classes=10):
        super(LeNet, self).__init__()

        self.num_classes = num_classes

        if init_weights:
            self._initialize_weights()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        z1 = self.conv1(x)  # 6 x 28 x 28
        a1 = F.relu(z1)  # 6 x 28 x 28
        a1 = F.max_pool2d(a1, kernel_size=2, stride=2)  # 6 x 14 x 14
        z2 = self.conv2(a1)  # 16 x 10 x 10
        a2 = F.relu(z2)  # 16 x 10 x 10
        a2 = F.max_pool2d(a2, kernel_size=2, stride=2)  # 16 x 5 x 5
        flatten_a2 = a2.view(a2.size(0), -1)
        z3 = self.fc1(flatten_a2)
        a3 = F.relu(z3)
        z4 = self.fc2(a3)
        a4 = F.relu(z4)
        z5 = self.fc3(a4)
        return z5

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


def test_lenet():
    net = LeNet(1)
    x = torch.randn(64, 1, 32, 32)
    y = net(x)
    print(y.size())


test_lenet()

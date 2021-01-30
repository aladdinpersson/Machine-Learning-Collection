from torch import nn
from efficientnet_pytorch import EfficientNet

class Net(nn.Module):
    def __init__(self, net_version, num_classes):
        super(Net, self).__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-'+net_version)
        self.backbone._fc = nn.Sequential(
            nn.Linear(1280, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)
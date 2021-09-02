from torch import nn


class AlexNet(nn.Module):
    """
    Adapted from: https://github.com/Mayurji/Image-Classification-PyTorch/blob/main/AlexNet.py
    """
    def __init__(self, num_classes: int = 10):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            # conv1
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=3),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # conv2
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # conv3
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # conv4
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # conv5
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten()
        )

        self.fc = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )

        # init param with Xavier method
        self.conv.apply(self.init_weights)
        self.fc.apply(self.init_weights)

    @staticmethod
    def init_weights(layer):
        if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

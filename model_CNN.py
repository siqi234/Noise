import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioDenoisingCNN(nn.Module):
    def __init__(self):
        super(AudioDenoisingCNN, self).__init__()
        # Define the layers of the CNN
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # Assuming mono-channel input
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.out_conv = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # Forward pass through the network
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.upsample(x)
        x = self.upsample(x)
        x = self.upsample(x)
        x = self.out_conv(x)
        return x


# To create an instance of the model:
# model = AudioDenoisingCNN()

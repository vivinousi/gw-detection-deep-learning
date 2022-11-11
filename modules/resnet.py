import torch
from torch import nn
from torch.nn import functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        if out_channels != in_channels or stride > 1:
            self.x_transform = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.x_transform = nn.Identity()

        self.body = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        x = F.relu(self.body(x) + self.x_transform(x))
        return x


class ResNet54(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            ResBlock(2, 8),
            ResBlock(8, 8),
            ResBlock(8, 8),
            ResBlock(8, 8),
            ResBlock(8, 16, stride=2),
            ResBlock(16, 16),
            ResBlock(16, 16),
            ResBlock(16, 32, stride=2),
            ResBlock(32, 32),
            ResBlock(32, 32),
            ResBlock(32, 64, stride=2),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64, stride=2),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64, stride=2),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 32),
            ResBlock(32, 32),
            ResBlock(32, 32),
            ResBlock(32, 32),
            ResBlock(32, 32),
            ResBlock(32, 16),
            ResBlock(16, 16),
            ResBlock(16, 16),
        )
        self.cls_head = nn.Sequential(
            nn.Conv1d(16, 32, 64), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 2, 1), nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.cls_head(x).squeeze(2)


class ResNet54Double(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            ResBlock(2, 16),
            ResBlock(16, 16),
            ResBlock(16, 16),
            ResBlock(16, 16),
            ResBlock(16, 32, stride=2),
            ResBlock(32, 32),
            ResBlock(32, 32),
            ResBlock(32, 64, stride=2),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 128, stride=2),
            ResBlock(128, 128),
            ResBlock(128, 128),
            ResBlock(128, 128, stride=2),
            ResBlock(128, 128),
            ResBlock(128, 128),
            ResBlock(128, 128, stride=2),
            ResBlock(128, 128),
            ResBlock(128, 128),
            ResBlock(128, 64),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 32),
            ResBlock(32, 32),
            ResBlock(32, 32),
        )
        self.cls_head = nn.Sequential(
            nn.Conv1d(32, 64, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 2, 1), nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.cls_head(x).squeeze(2)

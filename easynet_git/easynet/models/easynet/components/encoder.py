from __future__ import annotations

import torch
import torch.nn as nn


class ReconstructiveEncoder(nn.Module):
    def __init__(self, in_channels: int, base_width: int, num_scales: int = 1) -> None:
        super().__init__()
        self.num_scales = num_scales

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, base_width, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace = True),
            nn.Conv2d(base_width, base_width, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace = True),
        )
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(base_width, base_width * 2, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace = True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace = True),
        )
        self.maxpool2 = nn.MaxPool2d(2)
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(base_width * 2, base_width * 4, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace = True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace = True),
        )
        self.maxpool3 = nn.MaxPool2d(2)
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(base_width * 4, base_width * 8, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace = True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace = True),
        )
        self.maxpool4 = nn.MaxPool2d(2)
        self.conv_block5 = nn.Sequential(
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace = True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace = True),
        )

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        features1 = self.conv_block1(x)
        features2 = self.conv_block2(self.maxpool1(features1))
        features3 = self.conv_block3(self.maxpool2(features2))
        features4 = self.conv_block4(self.maxpool3(features3))
        features5 = self.conv_block5(self.maxpool4(features4))

        encoder_features = [features1, features2, features3][: self.num_scales]
        return encoder_features, features5
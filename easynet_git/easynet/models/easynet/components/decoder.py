from __future__ import annotations

import torch
import torch.nn as nn


class ReconstructiveDecoder(nn.Module):

    def __init__(self, base_width: int, out_channels: int = 3, num_scales: int = 1) -> None:
        super().__init__()
        self.num_scales = num_scales

        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor = 2, mode = "bilinear", align_corners = True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace = True),
        )
        self.decoder_block1 = nn.Sequential(
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace = True),
            nn.Conv2d(base_width * 8, base_width * 4, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace = True),
        )

        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor = 2, mode = "bilinear", align_corners = True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace = True),
        )
        self.decoder_block2 = nn.Sequential(
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace = True),
            nn.Conv2d(base_width * 4, base_width * 2, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace = True),
        )

        self.upsample3 = nn.Sequential(
            nn.Upsample(scale_factor = 2, mode = "bilinear", align_corners = True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace = True),
        )
        self.decoder_block3 = nn.Sequential(
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace = True),
            nn.Conv2d(base_width * 2, base_width, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace = True),
        )

        self.upsample4 = nn.Sequential(
            nn.Upsample(scale_factor = 2, mode = "bilinear", align_corners = True),
            nn.Conv2d(base_width, base_width, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace = True),
        )
        self.decoder_block4 = nn.Sequential(
            nn.Conv2d(base_width, base_width, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace = True),
            nn.Conv2d(base_width, base_width, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace = True),
        )

        self.output_conv = nn.Conv2d(base_width, out_channels, kernel_size = 3, padding = 1)

    def forward(self, bottleneck: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        features1 = self.decoder_block1(self.upsample1(bottleneck))
        features2 = self.decoder_block2(self.upsample2(features1))
        features3 = self.decoder_block3(self.upsample3(features2))
        features4 = self.decoder_block4(self.upsample4(features3))

        out = self.output_conv(features4)

        decoder_features = [features4, features3, features2][: self.num_scales][::-1]
        return out, decoder_features
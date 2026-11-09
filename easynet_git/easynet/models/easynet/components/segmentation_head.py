from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

MERGE_INPUT_CHANNEL_MULT = {1: 2, 2: 3, 3: 6}


class MultiScaleMerger(nn.Module):
    def __init__(self, base_width: int, num_scales: int, hidden_size: int) -> None:
        super().__init__()
        self.num_scales = num_scales
        self.hidden_size = hidden_size
        self.base_width = base_width

        projections = []
        if num_scales > 1:
            for i in range(1, num_scales + 1):
                projections.append(
                    nn.Sequential(
                        nn.Conv2d(base_width * MERGE_INPUT_CHANNEL_MULT[i], hidden_size, kernel_size = 1, stride = 1),
                        nn.BatchNorm2d(hidden_size),
                        nn.LeakyReLU(0.2),
                    )
                )
        self.projections = nn.ModuleList(projections)

    @property
    def merged_channels(self) -> int:
        if self.num_scales == 1:
            return self.base_width * MERGE_INPUT_CHANNEL_MULT[1]
        return self.hidden_size * self.num_scales

    def forward(self, encoder_features: list[torch.Tensor], decoder_features: list[torch.Tensor]) -> torch.Tensor:
        if self.num_scales == 1:
            return torch.cat((encoder_features[0], decoder_features[0]), dim = 1)

        paired_decoder_features = list(reversed(decoder_features))

        output_size = encoder_features[0].shape[2:]
        merges = []
        for i, (encoder_feature, decoder_feature, projection) in enumerate(
            zip(encoder_features, paired_decoder_features, self.projections), start = 1
        ):
            merge = projection(torch.cat((encoder_feature, decoder_feature), dim = 1))
            if i > 1:
                merge = F.interpolate(merge, size = output_size, mode = "bilinear", align_corners = False)
            merges.append(merge)

        return torch.cat(merges, dim = 1)


class AnomalyClassifierHead(nn.Module):
    def __init__(self, in_channels: int, hidden_size: int, hidden_size1: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden_size, kernel_size = 1, stride = 1),
            nn.BatchNorm2d(hidden_size),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_size, hidden_size1, kernel_size = 1, stride = 1),
            nn.BatchNorm2d(hidden_size1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_size1, 2, kernel_size = 1, stride = 1),
            nn.BatchNorm2d(2),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


def mask_logits_to_anomaly_map_and_score(mask_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    anomaly_map = torch.softmax(mask_logits, dim = 1)[:, 1:2, :, :]
    smoothed = F.avg_pool2d(anomaly_map, 21, stride = 1, padding = 21 // 2)
    pred_score = smoothed.amax(dim = (1, 2, 3))
    return anomaly_map, pred_score
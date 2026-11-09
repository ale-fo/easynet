from __future__ import annotations

import math
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

from moviad.models.vad_model import VADModel
from moviad.models.training_args import TrainingArgs

from .components.encoder import ReconstructiveEncoder
from .components.decoder import ReconstructiveDecoder
from .components.segmentation_head import (
    MultiScaleMerger,
    AnomalyClassifierHead,
    mask_logits_to_anomaly_map_and_score,
)
from .losses_draem import FocalLoss, SSIM


ENTROPY_MARGINS = {
    "bagel": 0.38,
    "cable_gland": 0.44,
    "carrot": 0.0,
    "cookie": 0.0,
    "dowel": 0.0,
    "foam": 0.0,
    "peach": -0.29,
    "potato": 0.0,
    "rope": -0.11,
    "tire": 0.0,
}

IMAGE_CHANNELS = 3

HIDDEN_SIZES = {
    1: (256, 1024),
    2: (512, 256),
    3: (512, 256),
}

class EasyNetFusionMode(str, Enum):
    SINGLE = "single"
    DEPTH = "depth"
    DUAL = "dual"
    FUSION1 = "fusion1"
    FUSION2 = "fusion2"

def feature_map_entropy(feature_map: torch.Tensor) -> torch.Tensor:
    flat = feature_map.reshape(-1)
    hist = torch.histc(flat, bins = 256, min = 0, max = 1)
    prob = hist / hist.sum()
    return (-prob * torch.log2(prob + 1e-12)).sum()

def select_fused_mask(
    mask: torch.Tensor,
    mask_rgb: torch.Tensor,
    pooled_rgb: torch.Tensor,
    pooled_fused: torch.Tensor,
    entropy_margin: float,
) -> torch.Tensor:
    use_fused_per_sample = []
    for i in range(mask.shape[0]):
        use_fused_per_sample.append(
            feature_map_entropy(pooled_fused[i]) > feature_map_entropy(pooled_rgb[i]) + entropy_margin
        )
    use_fused = torch.stack(use_fused_per_sample)
    return torch.where(use_fused.view(-1, 1, 1, 1), mask, mask_rgb)

SELECTION_METRICS = ("image_auroc", "image_ap", "pixel_auroc", "au_pro")
TOP_EPOCHS = 3

def selection_score(metrics: dict) -> float:
    total = 0.0
    for name in SELECTION_METRICS:
        total += metrics[name]
    return total / len(SELECTION_METRICS)

def top_epochs_order(entry: dict) -> tuple:
    return (-entry["score"], entry["epoch"])

def update_top_epochs(top_epochs: list, epoch: int, metrics: dict) -> list:
    score = selection_score(metrics)
    if not math.isfinite(score):
        return top_epochs
    candidates = list(top_epochs)
    candidates.append({"epoch": epoch, "score": score, "metrics": metrics})
    candidates.sort(key = top_epochs_order)
    return candidates[:TOP_EPOCHS]

class EasyNet(VADModel):
    @staticmethod
    def init_weight(m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)

    def __init__(
        self,
        fusion_mode: EasyNetFusionMode,
        num_scales: int = 1,
        base_width: int = 128,
        entropy_margin: float = 0.0,
        rgb_model: Optional["EasyNet"] = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()

        if not 1 <= num_scales <= 3:
            raise ValueError(f"num_scales must be in [1, 3], got {num_scales}")
        if fusion_mode == EasyNetFusionMode.FUSION2 and rgb_model is None:
            raise ValueError("FUSION2 requires a pretrained frozen SINGLE mode")
        if rgb_model is not None and rgb_model.fusion_mode != EasyNetFusionMode.SINGLE:
            raise ValueError("rgb_model must be a SINGLE mode")

        self.fusion_mode = fusion_mode
        self.num_scales = num_scales
        self.entropy_margin = entropy_margin

        hidden_size, hidden_size1 = HIDDEN_SIZES[num_scales]

        single_stream = fusion_mode in (EasyNetFusionMode.SINGLE, EasyNetFusionMode.DEPTH)
        dual_stream = fusion_mode in (EasyNetFusionMode.DUAL, EasyNetFusionMode.FUSION1)

        if fusion_mode != EasyNetFusionMode.FUSION2:
            self.encoder = ReconstructiveEncoder(IMAGE_CHANNELS, base_width, num_scales)
            self.decoder = ReconstructiveDecoder(base_width, IMAGE_CHANNELS, num_scales)
            self.merger = MultiScaleMerger(base_width, num_scales, hidden_size)

        if dual_stream or fusion_mode == EasyNetFusionMode.FUSION2:
            self.depth_encoder = ReconstructiveEncoder(IMAGE_CHANNELS, base_width, num_scales)
            self.depth_decoder = ReconstructiveDecoder(base_width, IMAGE_CHANNELS, num_scales)
            self.depth_merger = MultiScaleMerger(base_width, num_scales, hidden_size)

        if single_stream:
            self.classifier = AnomalyClassifierHead(self.merger.merged_channels, hidden_size, hidden_size1)
        elif fusion_mode == EasyNetFusionMode.DUAL:
            self.classifier = AnomalyClassifierHead(self.merger.merged_channels + self.depth_merger.merged_channels, hidden_size, hidden_size1)
        elif fusion_mode == EasyNetFusionMode.FUSION1:
            self.classifier_rgb = AnomalyClassifierHead(self.merger.merged_channels, hidden_size, hidden_size1)
            self.classifier = AnomalyClassifierHead(self.merger.merged_channels + self.depth_merger.merged_channels, hidden_size, hidden_size1)
        elif fusion_mode == EasyNetFusionMode.FUSION2:
            self.classifier = AnomalyClassifierHead(rgb_model.merger.merged_channels + self.depth_merger.merged_channels, hidden_size, hidden_size1)

        self.ssim_loss = SSIM()
        self.l2_loss = nn.MSELoss()
        self.focal_loss = FocalLoss()

        self.apply(self.init_weight)

        if fusion_mode == EasyNetFusionMode.FUSION2:
            self.rgb_model = rgb_model
            for p in self.rgb_model.parameters():
                p.requires_grad = False

        self.to(device)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.fusion_mode == EasyNetFusionMode.FUSION2:
            self.rgb_model.eval()
        return self

    def run_stream(self, x: Tensor, encoder: ReconstructiveEncoder, decoder: ReconstructiveDecoder, merger: MultiScaleMerger):
        encoder_features, bottleneck = encoder(x)
        out, decoder_features = decoder(bottleneck)
        merged = merger(encoder_features, decoder_features)
        return out, merged, encoder_features, decoder_features

    def compute_merge(self, x: Tensor) -> Tensor:
        if self.fusion_mode != EasyNetFusionMode.SINGLE:
            raise RuntimeError("compute_merge is only available on SINGLE mode")
        return self.run_stream(x, self.encoder, self.decoder, self.merger)[1]

    def fusion1_rgb_features(self, merged: Tensor, encoder_features: list[Tensor], depth_decoder_features: list[Tensor]) -> Tensor:
        if self.num_scales == 1:
            return torch.cat((encoder_features[0], depth_decoder_features[0]), dim = 1)
        return merged

    def forward(self, x: Tensor, d: Optional[Tensor] = None, training_outputs: Optional[bool] = None):
        if training_outputs is None:
            training_outputs = self.training

        if self.fusion_mode in (EasyNetFusionMode.SINGLE, EasyNetFusionMode.DEPTH):
            out, merged = self.run_stream(x, self.encoder, self.decoder, self.merger)[:2]
            mask = self.classifier(merged)
            if training_outputs:
                return out, mask
            return mask_logits_to_anomaly_map_and_score(mask)

        if self.fusion_mode == EasyNetFusionMode.FUSION2:
            with torch.no_grad():
                merged_rgb = self.rgb_model.compute_merge(x)
            depth_out, depth_merged = self.run_stream(d, self.depth_encoder, self.depth_decoder, self.depth_merger)[:2]
            fused = torch.cat((merged_rgb, depth_merged), dim = 1)
            mask = self.classifier(fused)
            if training_outputs:
                return depth_out, mask
            with torch.no_grad():
                mask_rgb = self.rgb_model.classifier(merged_rgb)
            pooled_rgb = merged_rgb.mean(dim = (2, 3), keepdim = True)
            pooled_fused = fused.mean(dim = (2, 3), keepdim = True)
            selected_mask = select_fused_mask(mask, mask_rgb, pooled_rgb, pooled_fused, self.entropy_margin)
            return mask_logits_to_anomaly_map_and_score(selected_mask)

        out, merged, encoder_features = self.run_stream(x, self.encoder, self.decoder, self.merger)[:3]
        depth_stream = self.run_stream(d, self.depth_encoder, self.depth_decoder, self.depth_merger)
        depth_out, depth_merged, depth_decoder_features = depth_stream[0], depth_stream[1], depth_stream[3]
        fused = torch.cat((merged, depth_merged), dim = 1)
        mask = self.classifier(fused)

        if self.fusion_mode == EasyNetFusionMode.DUAL:
            if training_outputs:
                return out, depth_out, mask
            return mask_logits_to_anomaly_map_and_score(mask)

        rgb_features = self.fusion1_rgb_features(merged, encoder_features, depth_decoder_features)
        mask_rgb = self.classifier_rgb(rgb_features)
        if training_outputs:
            return out, depth_out, mask, mask_rgb
        pooled_rgb = rgb_features.mean(dim = (2, 3), keepdim = True)
        pooled_fused = fused.mean(dim = (2, 3), keepdim = True)
        selected_mask = select_fused_mask(mask, mask_rgb, pooled_rgb, pooled_fused, self.entropy_margin)
        return mask_logits_to_anomaly_map_and_score(selected_mask)

    def rgb_reconstruction_loss(self, out: Tensor, image: Tensor) -> Tensor:
        return self.ssim_loss(image, out) + self.l2_loss(out, image)

    def depth_reconstruction_loss(self, depth_out: Tensor, depth: Tensor) -> Tensor:
        return self.l2_loss(depth_out, depth)

    def segmentation_loss(self, pred_mask: Tensor, gt_mask: Tensor) -> Tensor:
        return self.focal_loss(torch.softmax(pred_mask, dim = 1), gt_mask)

    def compute_loss(self, batch) -> Tensor:
        device = next(self.parameters()).device
        batch_on_device = []
        for t in batch:
            batch_on_device.append(t.to(device))
        image, depth, augmented_image, augmented_depth, mask = batch_on_device
        gt_mask = mask.long()

        if self.fusion_mode == EasyNetFusionMode.SINGLE:
            out, pred_mask = self(augmented_image, training_outputs = True)
            return self.rgb_reconstruction_loss(out, image) + self.segmentation_loss(pred_mask, gt_mask)
        if self.fusion_mode == EasyNetFusionMode.DEPTH:
            depth_out, pred_mask = self(augmented_depth, training_outputs = True)
            return self.depth_reconstruction_loss(depth_out, depth) + self.segmentation_loss(pred_mask, gt_mask)
        if self.fusion_mode == EasyNetFusionMode.FUSION2:
            depth_out, pred_mask = self(augmented_image, augmented_depth, training_outputs = True)
            return self.depth_reconstruction_loss(depth_out, depth) + self.segmentation_loss(pred_mask, gt_mask)
        if self.fusion_mode == EasyNetFusionMode.DUAL:
            out, depth_out, pred_mask = self(augmented_image, augmented_depth, training_outputs = True)
            return (
                self.rgb_reconstruction_loss(out, image)
                + self.depth_reconstruction_loss(depth_out, depth)
                + self.segmentation_loss(pred_mask, gt_mask)
            )
        out, depth_out, pred_mask, pred_mask_rgb = self(augmented_image, augmented_depth, training_outputs = True)
        rgb_reconstruction = self.rgb_reconstruction_loss(out, image)
        loss_rgb = rgb_reconstruction + self.segmentation_loss(pred_mask_rgb, gt_mask)
        loss_total = (
            rgb_reconstruction
            + self.depth_reconstruction_loss(depth_out, depth)
            + self.segmentation_loss(pred_mask, gt_mask)
        )
        return loss_rgb + loss_total

    def train_epoch(self, epoch, train_dataloader, training_args: TrainingArgs):
        avg_batch_loss = 0
        self.train()

        for batch in tqdm(train_dataloader, desc = f"Training epoch {epoch}", mininterval = 5.0):
            avg_batch_loss += self.train_step(batch, training_args)

        avg_batch_loss /= len(train_dataloader)
        if not math.isfinite(avg_batch_loss):
            raise FloatingPointError(f"non-finite train loss ({avg_batch_loss}) at epoch {epoch}")

        self.eval()
        return avg_batch_loss

    def train_step(self, batch, training_args: TrainingArgs):
        training_args.optimizer.zero_grad()
        loss = self.compute_loss(batch)
        loss.backward()
        training_args.optimizer.step()
        return loss.item()

    def train_chunk(self, train_dataloader, training_args: TrainingArgs):
        return self.train_epoch(epoch = 0, train_dataloader = train_dataloader, training_args = training_args)

    def save(self, save_path: str):
        torch.save(self.state_dict(), save_path)

    def reset_model(self):
        for name, module in self.named_children():
            if name != "rgb_model":
                module.apply(self.init_weight)

    def get_model_size(self):
        total_params = 0
        for p in self.parameters():
            total_params += p.numel()
        total_mb = total_params * 4 / (1024 ** 2)
        return {"model": total_mb, "total": total_mb}
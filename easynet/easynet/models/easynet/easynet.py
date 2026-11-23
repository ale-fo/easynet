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
    """Settings of EasyNet.
    
    - SINGLE: RGB only
    - DEPTH: depth only.
    - DUAL: RGB and depth streams, one classifier on their concatenated features ("RGBD").
    - FUSION1: as DUAL, plus a second classifier -> at inference the mask is chosen per sample by entropy.
    - FUSION2: as FUSION1, but the RGB stream and the RGB classifier are those of a frozen already trained SINGLE model ("Fusion2").
    """

    SINGLE = "single"
    DEPTH = "depth"
    DUAL = "dual"
    FUSION1 = "fusion1"
    FUSION2 = "fusion2"

def feature_map_entropy(feature_map: torch.Tensor) -> torch.Tensor:
    """Entropy of a feature map.

    Args:
        feature_map: tensor of any shape.

    Returns:
        Scalar tensor, NaN if no value falls in [0, 1].
    """
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
    """Chooses, sample by sample, the fused mask or the RGB one.

    Args:
        mask: classifier output on the fused features with shape (B, 2, H, W).
        mask_rgb: output of the RGB classifier with shape (B, 2, H, W).
        pooled_rgb: input features of the RGB classifier averaged over H and W with shape (B, C_rgb, 1, 1).
        pooled_fused: fused features averaged over H and W with shape (B, C_fused, 1, 1).
        entropy_margin: threshold alpha.

    Returns:
        Selected masks with shape (B, 2, H, W).
    """
    use_fused_per_sample = []
    for i in range(mask.shape[0]):
        use_fused_per_sample.append(
            feature_map_entropy(pooled_fused[i]) > feature_map_entropy(pooled_rgb[i]) + entropy_margin
        )
    use_fused = torch.stack(use_fused_per_sample)
    return torch.where(use_fused.view(-1, 1, 1, 1), mask, mask_rgb)

TOP_EPOCHS = 3

def top_epochs_order(entry: dict) -> tuple:
    """Sort key of the ranking: higher score first, the earlier epoch on ties."""
    return (-entry["score"], entry["epoch"])

class EasyNet(VADModel):
    """EasyNet anomaly detector. num_scales 1, 2 and 3 match the 1layer, 2layer and 3layer files of the official code. 
    
    - SINGLE, DEPTH: one stream (encoder, decoder, merger).
    - DUAL: also the depth stream (depth_encoder, depth_decoder, depth_merger).
    - FUSION1: as DUAL, plus classifier_rgb.
    - FUSION2: the depth stream and classifier, which reads the merged features of rgb_model and of the depth stream.
    The new layers get init_weight; rgb_model keeps its trained weights.
    """

    @staticmethod
    def init_weight(m: nn.Module) -> None:
        """Initialization of the weights of Conv2d layers."""
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
        """Builds the streams and classifiers of fusion_mode.
        
        Args:
            fusion_mode: modalities and fusion.
            num_scales: number of scales merged for the classifier (1 to 3).
            base_width: channels of the first encoder block.
            entropy_margin: threshold alpha of the entropy choice in FUSION1 and FUSION2, the values are in ENTROPY_MARGINS.
            rgb_model: trained SINGLE model, required in FUSION2 and used only there.
            device: device.
        """
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
        """It does NOT train. Sets train or eval mode. Training is done by train_epoch.
        Overrides nn.Module.train: in FUSION2 rgb_model always stays in eval mode, so its
        BatchNorm statistics do not change.
        
        Args:
            mode: True for train mode, False for eval mode.

        Returns:
            The model itself.
        """
        super().train(mode)
        if self.fusion_mode == EasyNetFusionMode.FUSION2:
            self.rgb_model.eval()
        return self

    def run_stream(self, x: Tensor, encoder: ReconstructiveEncoder, decoder: ReconstructiveDecoder, merger: MultiScaleMerger):
        """Runs one stream, encoder, decoder and merger, on a batch.

        Args:
            x: images with shape (B, 3, H, W).
            encoder: encoder of the stream.
            decoder: decoder of the stream.
            merger: merger of the stream.

        Returns:
            Tuple (out, merged, encoder_features, decoder_features):
            - out: reconstructed images with shape (B, 3, H, W);
            - merged: merged features with shape (B, merger.merged_channels, H, W);
            - encoder_features, decoder_features: the lists returned by the encoder and by the decoder.
        """
        encoder_features, bottleneck = encoder(x)
        out, decoder_features = decoder(bottleneck)
        merged = merger(encoder_features, decoder_features)
        return out, merged, encoder_features, decoder_features

    def compute_merge(self, x: Tensor) -> Tensor:
        """Merged RGB features of a SINGLE model.

        Args:
            x: RGB images with shape (B, 3, H, W).

        Returns:
            Merged features with shape (B, merger.merged_channels, H, W).
        """
        if self.fusion_mode != EasyNetFusionMode.SINGLE:
            raise RuntimeError("compute_merge is only available on SINGLE mode")
        return self.run_stream(x, self.encoder, self.decoder, self.merger)[1]

    def fusion1_rgb_features(self, merged: Tensor, encoder_features: list[Tensor], depth_decoder_features: list[Tensor]) -> Tensor:
        """Features read by classifier_rgb in FUSION1.

        Args:
            merged: merged RGB features with shape (B, merger.merged_channels, H, W).
            encoder_features: list returned by the RGB encoder.
            depth_decoder_features: list returned by the depth decoder.

        Returns:
            Features with shape (B, merger.merged_channels, H, W).
        """
        if self.num_scales == 1:
            return torch.cat((encoder_features[0], depth_decoder_features[0]), dim = 1)
        return merged

    def forward(self, x: Tensor, d: Optional[Tensor] = None, training_outputs: Optional[bool] = None):
        """Training outputs, or anomaly maps and scores, of a batch.

        Args:
            x: RGB images with shape (B, 3, H, W); depth images in DEPTH mode.
            d: depth images with shape (B, 3, H, W), or None in SINGLE and DEPTH mode.
            training_outputs: whether to return the training outputs; None means self.training.


        Returns:
            With training outputs, reconstructions with shape (B, 3, H, W) and classifier
            outputs with shape (B, 2, H, W), before the softmax:
            - SINGLE, DEPTH: (out, mask);
            - FUSION2: (depth_out, mask);
            - DUAL: (out, depth_out, mask);
            - FUSION1: (out, depth_out, mask, mask_rgb).
            Otherwise (anomaly_map, pred_score) from mask_logits_to_anomaly_map_and_score,
            with shapes (B, 1, H, W) and (B,). In FUSION1 and FUSION2 the mask is chosen by select_fused_mask.
        """
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
        """RGB reconstruction loss: SSIM loss plus mean squared error.

        Args:
            out: reconstructed images with shape (B, 3, H, W).
            image: original images with shape (B, 3, H, W), values in [0, 1].

        Returns:
            Scalar loss.
        """
        return self.ssim_loss(image, out) + self.l2_loss(out, image)

    def depth_reconstruction_loss(self, depth_out: Tensor, depth: Tensor) -> Tensor:
        """Depth reconstruction loss: mean squared error only.

        Args:
            depth_out: reconstructed depth images with shape (B, 3, H, W).
            depth: original depth images with shape (B, 3, H, W).

        Returns:
            Scalar loss.
        """
        return self.l2_loss(depth_out, depth)

    def segmentation_loss(self, pred_mask: Tensor, gt_mask: Tensor) -> Tensor:
        """Focal loss of the classifier output against the anomaly mask.

        Args:
            pred_mask: classifier output with shape (B, 2, H, W).
            gt_mask: anomaly mask with shape (B, 1, H, W).

        Returns:
            Scalar loss.
        """
        return self.focal_loss(torch.softmax(pred_mask, dim = 1), gt_mask)

    def compute_loss(self, batch) -> Tensor:
        """Training loss of a batch for the current mode.

        Args:
            batch: tuple (image, depth, augmented_image, augmented_depth, mask): 
            - images with shape (B, 3, H, W);
            - mask with shape (B, 1, H, W) and values 0 or 1.

        Returns:
            Scalar loss.
        """
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
        """Trains the model for one epoch.

        Args:
            epoch: epoch number
            train_dataloader: loader of the train split of EasyNetMVTec3DDataset.
            training_args: passed to train_step.

        Returns:
            Mean of the batch losses of the epoch.
        """
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
        """One optimization step on a batch.

        Args:
            batch: training batch.
            training_args: holds the optimizer.

        Returns:
            Loss of the batch.
        """
        training_args.optimizer.zero_grad()
        loss = self.compute_loss(batch)
        loss.backward()
        training_args.optimizer.step()
        return loss.item()

    def train_chunk(self, train_dataloader, training_args: TrainingArgs):
        """One epoch on train_dataloader.

        Args:
            train_dataloader: loader of training batches.
            training_args: passed to train_step.

        Returns:
            Mean of the batch losses of the epoch.
        """
        return self.train_epoch(epoch = 0, train_dataloader = train_dataloader, training_args = training_args)

    def save(self, save_path: str):
        """Saves the weights to save_path.

        Args:
            save_path: destination file.
        """
        torch.save(self.state_dict(), save_path)

    def reset_model(self):
        """Re-initializes the Conv2d weights with init_weight."""
        for name, module in self.named_children():
            if name != "rgb_model":
                module.apply(self.init_weight)

    def get_model_size(self):
        """Size of the parameters.

        Returns:
            Dict with keys "model" and "total", both set to that size.
        """
        total_params = 0
        for p in self.parameters():
            total_params += p.numel()
        total_mb = total_params * 4 / (1024 ** 2)
        return {"model": total_mb, "total": total_mb}

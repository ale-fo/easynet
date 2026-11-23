from __future__ import annotations

from math import exp

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss for the per-pixel classification."""

    def __init__(self, gamma = 2, smooth = 1e-5):
        """Stores the parameters.

        Args:
            gamma: focusing parameter, larger values reduce the weight of well classified pixels.
            smooth: smoothing constant.
        """
        super().__init__()
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logit, target):
        """Mean focal loss over all the pixels.

        Args:
            logit: class probabilities with shape (B, C, H, W).
            target: class indices with shape (B, 1, H, W).

        Returns:
            Scalar loss.
        """
        num_class = logit.shape[1]

        logit = logit.view(logit.size(0), logit.size(1), -1)
        logit = logit.permute(0, 2, 1).contiguous()
        logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)

        idx = target.cpu().long()

        one_hot_target = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_target = one_hot_target.scatter_(1, idx, 1)
        if one_hot_target.device != logit.device:
            one_hot_target = one_hot_target.to(logit.device)

        one_hot_target = torch.clamp(one_hot_target, self.smooth / (num_class - 1), 1.0 - self.smooth)
        true_class_prob = (one_hot_target * logit).sum(1) + self.smooth
        log_true_class_prob = true_class_prob.log()

        loss = -1 * torch.pow((1 - true_class_prob), self.gamma) * log_true_class_prob

        return loss.mean()

def gaussian(window_size: int, sigma: float) -> torch.Tensor:
    """Normalized 1D Gaussian window, centered on window_size // 2.

    Args:
        window_size: number of samples.
        sigma: standard deviation.

    Returns:
        Tensor with shape (window_size,), summing to 1.
    """
    gauss_values = []
    for x in range(window_size):
        gauss_values.append(exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)))
    gauss = torch.Tensor(gauss_values)
    return gauss / gauss.sum()

def create_window(window_size: int, channel: int = 1) -> torch.Tensor:
    """2D Gaussian window with sigma 1.5, for a channel grouped convolution.

    Args:
        window_size: side of the window.
        channel: number of channels.

    Returns:
        Tensor with shape (channel, 1, window_size, window_size).
    """
    gaussian_1d = gaussian(window_size, 1.5).unsqueeze(1)
    gaussian_2d = gaussian_1d.mm(gaussian_1d.t()).float().unsqueeze(0).unsqueeze(0)
    return gaussian_2d.expand(channel, 1, window_size, window_size).contiguous()

def ssim(img1, img2, window, window_size = 11):
    """Mean structural similarity (SSIM) of two batches of images.

    Args:
        img1: images with shape (B, C, H, W).
        img2: images with shape (B, C, H, W).
        window: window from `create_window` with C channels, on the device of the images.
        window_size: side of the window.

    Returns:
        Scalar tensor: the mean of the SSIM map.
    """
    max_val = 255 if torch.max(img1) > 128 else 1
    min_val = -1 if torch.min(img1) < -0.5 else 0
    l = max_val - min_val

    padd = window_size // 2
    channel = img1.size(1)

    mu1 = F.conv2d(img1, window, padding = padd, groups = channel)
    mu2 = F.conv2d(img2, window, padding = padd, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    var1 = F.conv2d(img1 * img1, window, padding = padd, groups = channel) - mu1_sq
    var2 = F.conv2d(img2 * img2, window, padding = padd, groups = channel) - mu2_sq
    covar12 = F.conv2d(img1 * img2, window, padding = padd, groups = channel) - mu1_mu2

    c1 = (0.01 * l) ** 2
    c2 = (0.03 * l) ** 2

    contrast_num = 2.0 * covar12 + c2
    contrast_den = var1 + var2 + c2

    ssim_map = ((2 * mu1_mu2 + c1) * contrast_num) / ((mu1_sq + mu2_sq + c1) * contrast_den)

    return ssim_map.mean()

class SSIM(nn.Module):
    """SSIM loss, 1 - SSIM."""

    def __init__(self, window_size = 11):
        """Builds a 1-channel window.

        Args:
            window_size: side of the Gaussian window.
        """
        super().__init__()
        self.window_size = window_size
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        """SSIM loss of two batches of images.

        Args:
            img1: images with shape (B, C, H, W).
            img2: images with shape (B, C, H, W).

        Returns:
            Scalar tensor 1 - SSIM, which is 0 for identical images.
        """
        channel = img1.size(1)

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        ssim_value = ssim(img1, img2, window = window, window_size = self.window_size)
        return 1.0 - ssim_value
from __future__ import annotations

from math import exp

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):

    def __init__(self, gamma = 2, smooth = 1e-5):
        super().__init__()
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logit, target):
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
    gauss_values = []
    for x in range(window_size):
        gauss_values.append(exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)))
    gauss = torch.Tensor(gauss_values)
    return gauss / gauss.sum()

def create_window(window_size: int, channel: int = 1) -> torch.Tensor:
    gaussian_1d = gaussian(window_size, 1.5).unsqueeze(1)
    gaussian_2d = gaussian_1d.mm(gaussian_1d.t()).float().unsqueeze(0).unsqueeze(0)
    return gaussian_2d.expand(channel, 1, window_size, window_size).contiguous()

def ssim(img1, img2, window, window_size = 11):
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

    def __init__(self, window_size = 11):
        super().__init__()
        self.window_size = window_size
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        channel = img1.size(1)

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        ssim_value = ssim(img1, img2, window = window, window_size = self.window_size)
        return 1.0 - ssim_value
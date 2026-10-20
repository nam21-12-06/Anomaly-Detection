from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def create_gaussian_window_2d(window_size: int, sigma: float, channels: int) -> torch.Tensor:
    """
    Creates a 2D Gaussian kernel window for SSIM computation.
    """
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g_1d = torch.exp(-(coords**2) / (2 * sigma**2))
    g_1d = g_1d / g_1d.sum()

    g_2d = g_1d.unsqueeze(1) @ g_1d.unsqueeze(0)
    window = g_2d.expand(channels, 1, window_size, window_size).contiguous()
    return window


class SSIM(nn.Module):
    """
    Structural Similarity Index (SSIM) metric module.
    Computes SSIM between original and reconstructed images.
    """

    def __init__(self, window_size: int = 11, sigma: float = 1.5, in_channels: int = 3):
        super().__init__()
        self.window_size = window_size
        self.in_channels = in_channels
        self.register_buffer("window", create_gaussian_window_2d(window_size, sigma, in_channels))

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        channel = img1.size(1)
        if channel != self.in_channels:
            raise ValueError(f"Input channel count {channel} does not match configured {self.in_channels}")

        window = self.window.to(dtype=img1.dtype, device=img1.device)

        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=channel) - mu1_mu2

        c1 = 0.01**2
        c2 = 0.03**2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        return ssim_map.mean()


class SSIMLoss(nn.Module):
    """
    SSIM Loss defined as 1.0 - SSIM(x, y).
    """

    def __init__(self, window_size: int = 11, sigma: float = 1.5, in_channels: int = 3):
        super().__init__()
        self.ssim = SSIM(window_size=window_size, sigma=sigma, in_channels=in_channels)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 1.0 - self.ssim(pred, target)


class AutoencoderLoss(nn.Module):
    """
    Combined reconstruction loss for Autoencoders:
    Loss = alpha * L1_Loss + (1 - alpha) * SSIM_Loss
    """

    def __init__(self, alpha: float = 0.85, in_channels: int = 3):
        super().__init__()
        self.alpha = alpha
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss(in_channels=in_channels)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        l1 = self.l1_loss(pred, target)
        ssim_l = self.ssim_loss(pred, target)
        total_loss = self.alpha * l1 + (1.0 - self.alpha) * ssim_l
        loss_dict = {
            "loss": total_loss.item(),
            "l1": l1.item(),
            "ssim_loss": ssim_l.item(),
        }
        return total_loss, loss_dict


class VAELoss(nn.Module):
    """
    Variational Autoencoder loss function:
    Loss = Reconstruction_Loss + kl_weight * KL_Divergence
    where KL_Divergence = -0.5 * sum(1 + log(var) - mu^2 - var)
    """

    def __init__(self, alpha: float = 0.85, kl_weight: float = 1e-4, in_channels: int = 3):
        super().__init__()
        self.recon_loss_fn = AutoencoderLoss(alpha=alpha, in_channels=in_channels)
        self.kl_weight = kl_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        recon_loss, loss_dict = self.recon_loss_fn(pred, target)

        # KL divergence per batch item averaged over batch size
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / pred.size(0)

        total_loss = recon_loss + self.kl_weight * kl_div
        loss_dict["kl_div"] = kl_div.item()
        loss_dict["total_loss"] = total_loss.item()
        return total_loss, loss_dict


import torch
import pytest

from src.models.losses import (
    SSIM,
    AutoencoderLoss,
    SSIMLoss,
    VAELoss,
)


def test_ssim_identity_and_bounds():
    x = torch.rand(2, 3, 224, 224)
    ssim_module = SSIM(window_size=11, in_channels=3)

    # Identical images should have SSIM ~ 1.0
    identical_ssim = ssim_module(x, x)
    assert torch.isclose(identical_ssim, torch.tensor(1.0), atol=1e-3)

    # SSIM Loss for identical images should be ~ 0.0
    ssim_loss_fn = SSIMLoss(window_size=11, in_channels=3)
    loss_val = ssim_loss_fn(x, x)
    assert torch.isclose(loss_val, torch.tensor(0.0), atol=1e-3)


def test_autoencoder_loss_calculation():
    pred = torch.rand(2, 3, 224, 224)
    target = torch.rand(2, 3, 224, 224)

    ae_loss_fn = AutoencoderLoss(alpha=0.85, in_channels=3)
    total_loss, loss_dict = ae_loss_fn(pred, target)

    assert total_loss.dim() == 0, "Loss must be scalar"
    assert total_loss.item() > 0.0
    assert "l1" in loss_dict and "ssim_loss" in loss_dict
    expected_total = 0.85 * loss_dict["l1"] + 0.15 * loss_dict["ssim_loss"]
    assert abs(total_loss.item() - expected_total) < 1e-5


def test_vae_loss_calculation_and_kl():
    pred = torch.rand(2, 3, 224, 224)
    target = torch.rand(2, 3, 224, 224)
    mu = torch.randn(2, 64)
    logvar = torch.randn(2, 64)

    vae_loss_fn = VAELoss(alpha=0.85, kl_weight=1e-4, in_channels=3)
    total_loss, loss_dict = vae_loss_fn(pred, target, mu, logvar)

    assert total_loss.dim() == 0
    assert "kl_div" in loss_dict
    assert "total_loss" in loss_dict
    assert loss_dict["kl_div"] >= 0.0, "KL divergence with standard normal prior should be non-negative"


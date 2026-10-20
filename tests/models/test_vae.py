import torch
import pytest

from src.models.autoencoder import ConvVAE


def test_vae_forward_and_shapes():
    batch_size = 2
    in_channels = 3
    latent_dim = 64

    model = ConvVAE(in_channels=in_channels, base_channels=16, latent_dim=latent_dim)
    model.train()

    x = torch.rand(batch_size, in_channels, 224, 224)
    recon, mu, logvar = model(x)

    assert recon.shape == (batch_size, in_channels, 224, 224)
    assert mu.shape == (batch_size, latent_dim)
    assert logvar.shape == (batch_size, latent_dim)
    assert recon.min() >= 0.0 and recon.max() <= 1.0


def test_vae_eval_determinism():
    batch_size = 2
    model = ConvVAE(in_channels=3, base_channels=16, latent_dim=32)
    model.eval()

    x = torch.rand(batch_size, 3, 224, 224)
    recon1, mu1, logvar1 = model(x)
    recon2, mu2, logvar2 = model(x)

    assert torch.allclose(recon1, recon2), "In eval mode, VAE reconstruction must be deterministic (z = mu)"
    assert torch.allclose(mu1, mu2)
    assert torch.allclose(logvar1, logvar2)


def test_vae_anomaly_map_and_score():
    batch_size = 2
    model = ConvVAE(in_channels=3, base_channels=16, latent_dim=32)
    model.eval()

    x = torch.rand(batch_size, 3, 224, 224)
    anomaly_map = model.get_anomaly_map(x)
    anomaly_score = model.get_anomaly_score(x)

    assert anomaly_map.shape == (batch_size, 1, 224, 224)
    assert (anomaly_map >= 0.0).all()
    assert anomaly_score.shape == (batch_size,)
    assert (anomaly_score >= 0.0).all()


def test_vae_gradient_flow():
    model = ConvVAE(in_channels=3, base_channels=16, latent_dim=32)
    model.train()

    x = torch.rand(2, 3, 224, 224)
    recon, mu, logvar = model(x)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss = torch.nn.functional.l1_loss(recon, x) + 1e-4 * kl_div
    loss.backward()

    assert model.encoder.fc_mu.weight.grad is not None
    assert model.encoder.fc_logvar.weight.grad is not None
    assert model.decoder.deconv_blocks[0].weight.grad is not None


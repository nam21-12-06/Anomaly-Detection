import torch
import pytest

from src.models.autoencoder import ConvAutoencoder, ConvDecoder, ConvEncoder


def test_conv_encoder_forward():
    batch_size = 2
    in_channels = 3
    latent_dim = 128
    encoder = ConvEncoder(in_channels=in_channels, base_channels=16, latent_dim=latent_dim, is_vae=False)
    x = torch.rand(batch_size, in_channels, 224, 224)
    z = encoder(x)

    assert z.shape == (batch_size, latent_dim), f"Expected {(batch_size, latent_dim)}, got {z.shape}"


def test_conv_decoder_forward():
    batch_size = 2
    out_channels = 3
    latent_dim = 128
    decoder = ConvDecoder(latent_dim=latent_dim, base_channels=16, out_channels=out_channels)
    z = torch.randn(batch_size, latent_dim)
    recon = decoder(z)

    assert recon.shape == (batch_size, out_channels, 224, 224)
    assert recon.min() >= 0.0 and recon.max() <= 1.0, "Decoder output must be bounded in [0, 1] by Sigmoid"


def test_autoencoder_forward_and_shapes():
    batch_size = 2
    in_channels = 3
    latent_dim = 64
    model = ConvAutoencoder(in_channels=in_channels, base_channels=16, latent_dim=latent_dim)
    model.eval()

    x = torch.rand(batch_size, in_channels, 224, 224)
    recon, z = model(x)

    assert recon.shape == (batch_size, in_channels, 224, 224)
    assert z.shape == (batch_size, latent_dim)
    assert recon.min() >= 0.0 and recon.max() <= 1.0


def test_autoencoder_anomaly_map_and_score():
    batch_size = 2
    model = ConvAutoencoder(in_channels=3, base_channels=16, latent_dim=32)
    model.eval()

    x = torch.rand(batch_size, 3, 224, 224)
    anomaly_map = model.get_anomaly_map(x)
    anomaly_score = model.get_anomaly_score(x)

    assert anomaly_map.shape == (batch_size, 1, 224, 224)
    assert (anomaly_map >= 0.0).all(), "Anomaly residual map must be non-negative"
    assert anomaly_score.shape == (batch_size,)
    assert (anomaly_score >= 0.0).all(), "Image anomaly scores must be non-negative"


def test_autoencoder_gradient_flow():
    model = ConvAutoencoder(in_channels=3, base_channels=16, latent_dim=32)
    model.train()

    x = torch.rand(2, 3, 224, 224)
    recon, _ = model(x)
    loss = torch.nn.functional.l1_loss(recon, x)
    loss.backward()

    assert model.encoder.conv_blocks[0].weight.grad is not None
    assert model.decoder.deconv_blocks[0].weight.grad is not None


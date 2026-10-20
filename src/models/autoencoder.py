from typing import Tuple, Union

import torch
import torch.nn as nn


class ConvEncoder(nn.Module):
    """
    Convolutional Encoder for visual anomaly detection.
    Downsamples an input image of shape (B, in_channels, 224, 224)
    into a compact bottleneck feature representation.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        latent_dim: int = 128,
        is_vae: bool = False,
    ):
        super().__init__()
        self.is_vae = is_vae
        self.latent_dim = latent_dim

        # 224x224 -> 112x112 -> 56x56 -> 28x28 -> 14x14 -> 7x7
        self.conv_blocks = nn.Sequential(
            # Stage 1: (B, 3, 224, 224) -> (B, 32, 112, 112)
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            # Stage 2: (B, 32, 112, 112) -> (B, 64, 56, 56)
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # Stage 3: (B, 64, 56, 56) -> (B, 128, 28, 28)
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # Stage 4: (B, 128, 28, 28) -> (B, 256, 14, 14)
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # Stage 5: (B, 256, 14, 14) -> (B, 512, 7, 7)
            nn.Conv2d(base_channels * 8, base_channels * 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 16),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.flatten_dim = base_channels * 16 * 7 * 7

        if not is_vae:
            self.fc = nn.Linear(self.flatten_dim, latent_dim)
        else:
            self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
            self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        h = self.conv_blocks(x)
        h = torch.flatten(h, start_dim=1)

        if not self.is_vae:
            z = self.fc(h)
            return z
        else:
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)
            return mu, logvar


class ConvDecoder(nn.Module):
    """
    Convolutional Decoder for visual reconstruction.
    Upsamples a latent vector z back to an image of shape (B, out_channels, 224, 224).
    """

    def __init__(
        self,
        latent_dim: int = 128,
        base_channels: int = 32,
        out_channels: int = 3,
    ):
        super().__init__()
        self.base_channels = base_channels
        self.flatten_dim = base_channels * 16 * 7 * 7

        self.fc = nn.Linear(latent_dim, self.flatten_dim)

        # 7x7 -> 14x14 -> 28x28 -> 56x56 -> 112x112 -> 224x224
        self.deconv_blocks = nn.Sequential(
            # Stage 1: (B, 512, 7, 7) -> (B, 256, 14, 14)
            nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # Stage 2: (B, 256, 14, 14) -> (B, 128, 28, 28)
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # Stage 3: (B, 128, 28, 28) -> (B, 64, 56, 56)
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # Stage 4: (B, 64, 56, 56) -> (B, 32, 112, 112)
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            # Stage 5: (B, 32, 112, 112) -> (B, 3, 224, 224)
            nn.ConvTranspose2d(base_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        h = h.view(-1, self.base_channels * 16, 7, 7)
        recon = self.deconv_blocks(h)
        return recon


class ConvAutoencoder(nn.Module):
    """
    Convolutional Autoencoder baseline for Visual Anomaly Detection.

    Learns an identity mapping on normal samples. Anomalies at test time
    yield high reconstruction errors at anomalous regions:
        Anomaly_Map = |x - reconstructed_x|
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        latent_dim: int = 128,
    ):
        super().__init__()
        self.encoder = ConvEncoder(in_channels=in_channels, base_channels=base_channels, latent_dim=latent_dim, is_vae=False)
        self.decoder = ConvDecoder(latent_dim=latent_dim, base_channels=base_channels, out_channels=in_channels)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z

    def get_anomaly_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes per-pixel anomaly residual map: (B, 1, H, W).
        Mean absolute difference across color channels.
        """
        self.eval()
        with torch.no_grad():
            recon, _ = self.forward(x)
            residual = torch.abs(x - recon)
            anomaly_map = residual.mean(dim=1, keepdim=True)
        return anomaly_map

    def get_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes image-level anomaly score: scalar per batch item.
        Average of the top 5% highest reconstruction error pixels.
        """
        anomaly_map = self.get_anomaly_map(x)
        b = anomaly_map.size(0)
        flat = anomaly_map.view(b, -1)
        topk_count = max(1, int(flat.size(1) * 0.05))
        topk_vals, _ = torch.topk(flat, k=topk_count, dim=1)
        return topk_vals.mean(dim=1)


class ConvVAE(nn.Module):
    """
    Convolutional Variational Autoencoder (VAE) baseline for Visual Anomaly Detection.

    Encodes inputs into Gaussian distribution parameters (mu, logvar) and uses the
    reparameterization trick to sample latent vectors z ~ N(mu, var).
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        latent_dim: int = 128,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = ConvEncoder(in_channels=in_channels, base_channels=base_channels, latent_dim=latent_dim, is_vae=True)
        self.decoder = ConvDecoder(latent_dim=latent_dim, base_channels=base_channels, out_channels=in_channels)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def get_anomaly_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes per-pixel anomaly residual map: (B, 1, H, W).
        Evaluated deterministically using z = mu.
        """
        self.eval()
        with torch.no_grad():
            mu, _ = self.encode(x)
            recon = self.decode(mu)
            residual = torch.abs(x - recon)
            anomaly_map = residual.mean(dim=1, keepdim=True)
        return anomaly_map

    def get_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes image-level anomaly score: scalar per batch item.
        Average of top 5% highest reconstruction error pixels.
        """
        anomaly_map = self.get_anomaly_map(x)
        b = anomaly_map.size(0)
        flat = anomaly_map.view(b, -1)
        topk_count = max(1, int(flat.size(1) * 0.05))
        topk_vals, _ = torch.topk(flat, k=topk_count, dim=1)
        return topk_vals.mean(dim=1)


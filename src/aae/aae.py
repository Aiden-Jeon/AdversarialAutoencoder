from math import pi
from typing import Tuple, List, Any

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pytorch_lightning as pl

from .layer import FullyConnectedLayer
from .utils import get_hidden_sizes, plot_latent


class Discriminator(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int, btl_size: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )
        self.D_merge_y = nn.Linear(num_labels, hidden_size, bias=False)
        self.D_merge_z = nn.Linear(btl_size, hidden_size, bias=False)

    def forward(self, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        merge = self.D_merge_y(y) + self.D_merge_z(z)
        logit = self.features(merge)
        return logit


class AdversarialAutoEncoder(pl.LightningModule):
    def __init__(
        self,
        input_size: int = 28 ** 2,
        hidden_size: int = 2,
        disc_hidden_size: int = 1000,
        n_layers: int = 3,
        num_labels: int = 10,
        sample_latent: str = "swiss_roll",
    ):
        if sample_latent == "gaussian_mixture" and hidden_size % 2 != 0:
            raise Exception("hidden_size must be a multiple of 2")
        super().__init__()
        encoder_hidden_sizes = get_hidden_sizes(input_size, hidden_size, n_layers)
        encoder_layers = []
        for i, o in zip(encoder_hidden_sizes[:-2], encoder_hidden_sizes[1:-1]):
            encoder_layers += [FullyConnectedLayer(i, o, "relu")]
        encoder_layers += [
            FullyConnectedLayer(
                encoder_hidden_sizes[-2], encoder_hidden_sizes[-1], act=None
            )
        ]
        decoder_hidden_sizes = get_hidden_sizes(hidden_size, input_size, n_layers)
        decoder_layers = []
        for i, o in zip(decoder_hidden_sizes[:-2], decoder_hidden_sizes[1:-1]):
            decoder_layers += [FullyConnectedLayer(i, o, "relu")]
        decoder_layers += [
            FullyConnectedLayer(
                decoder_hidden_sizes[-2], decoder_hidden_sizes[-1], act=None
            )
        ]
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.sample_latent = sample_latent
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)
        self.discriminator = Discriminator(disc_hidden_size, num_labels, hidden_size)
        self.loss_fn = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.automatic_optimization = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def get_recon_loss(self, x: torch.Tensor) -> torch.Tensor:
        x_recon = self.forward(x)
        recon_loss = self.loss_fn(x_recon, x)
        return recon_loss

    def sample_swiss_roll(self, batch_size: int, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.view(-1, 1)
        uni = torch.FloatTensor(batch_size, 1).uniform_(0, 1)
        uni = uni.add(labels).mul(1 / 10)

        r = uni.sqrt().mul(3)
        rad = uni.sqrt().mul(pi * 4)

        new_x = r.mul(rad.cos())
        new_y = r.mul(rad.sin())
        return torch.cat([new_x, new_y], 1)

    def sample_gaussian_mixture(
        self, batch_size: int, labels: torch.Tensor
    ) -> torch.Tensor:
        x_var = 0.5
        y_var = 0.05
        x = torch.randn(batch_size, self.hidden_size // 2).mul(x_var)
        y = torch.randn(batch_size, self.hidden_size // 2).mul(y_var)

        shift = 1.4
        if labels is None:
            labels = torch.randint(0, self.num_labels, (batch_size,))

        r = labels.type(torch.float32).mul(2 * pi / self.num_labels)

        sin_r = r.sin().view(-1, 1)
        cos_r = r.cos().view(-1, 1)

        new_x = x.mul(cos_r) - y.mul(sin_r)
        new_y = x.mul(sin_r) + y.mul(cos_r)

        new_x += shift * cos_r
        new_y += shift * sin_r

        return torch.cat([new_x, new_y], 1)

    def get_D_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        z_fake = self.encoder(x)
        if self.sample_latent == "gaussian_mixture":
            z_true = self.sample_gaussian_mixture(z_fake.size(0), y).to(self.device)
        else:
            z_true = self.sample_swiss_roll(z_fake.size(0), y).to(self.device)
        y_onehot = torch.zeros(y.size(0), self.num_labels).to(self.device)
        y_onehot.scatter_(1, y.view(-1,1), 1).to(self.device)

        z_true_pred = self.discriminator(y_onehot, z_true)
        z_fake_pred = self.discriminator(y_onehot, z_fake)

        target_ones = torch.ones(x.size(0), 1).to(self.device)
        target_zeros = torch.zeros(x.size(0), 1).to(self.device)

        true_loss = self.bce_loss(z_true_pred, target_ones)
        fake_loss = self.bce_loss(z_fake_pred, target_zeros)

        D_loss = true_loss + fake_loss
        return D_loss

    def get_G_loss_value(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        y_onehot = torch.zeros(y.size(0), self.num_labels).to(self.device)
        y_onehot.scatter_(1, y.view(-1,1), 1).to(self.device)
        target_ones = torch.ones(batch_size, 1).to(self.device)
        z_fake = self.encoder(x)
        z_fake_pred = self.discriminator(y_onehot, z_fake)
        G_loss = self.bce_loss(z_fake_pred, target_ones)
        return G_loss

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        optimizer_idx: int,
    ) -> torch.Tensor:
        encoder_opt, decoder_opt, disc_opt = self.optimizers()
        x, y = batch
        x = x.view(x.size(0), -1)
        #
        # update encoder, decoder
        #
        encoder_opt.zero_grad()
        decoder_opt.zero_grad()
        recon_loss = self.get_recon_loss(x)
        recon_loss.backward()
        decoder_opt.step()
        encoder_opt.step()
        #
        # update discriminator
        #
        disc_opt.zero_grad()
        D_loss = self.get_D_loss(x, y)
        D_loss.backward()
        disc_opt.step()
        #
        # update generator
        #
        encoder_opt.zero_grad()
        G_loss = self.get_G_loss_value(x, y)
        G_loss.backward()
        encoder_opt.step()

        loss = recon_loss + D_loss + G_loss

        self.log_dict({"recon_loss": recon_loss, "D_loss": D_loss, "G_loss": G_loss})
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        x = x.view(x.size(0), -1)
        latent = self.encoder(x)
        recon_x = self.decoder(latent)
        loss = self.loss_fn(x, recon_x)
        self.log("valid_loss", loss)
        return {"loss": loss, "latent": latent, "label": y}

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        latent = []
        label = []
        for output in outputs:
            latent += [output["latent"]]
            label += [output["label"]]
        latent = torch.cat(latent).numpy()
        label = torch.cat(label).numpy()
        fig = plot_latent(latent, label)
        fig.savefig("image_at_epoch_{:04d}.png".format(self.current_epoch))
        plt.close()

    def configure_optimizers(self):
        encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=1e-3)
        decoder_opt = torch.optim.Adam(self.decoder.parameters(), lr=1e-3)
        disc_opt = torch.optim.Adam(self.discriminator.parameters(), lr=1e-3)
        return encoder_opt, decoder_opt, disc_opt

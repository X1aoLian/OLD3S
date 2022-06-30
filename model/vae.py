import torch
from torch import nn

latent_dim = 32
inter_dim = 128
mid_dim = (256, 2, 2)
mid_num = 1
for i in mid_dim:
    mid_num *= i


class ConvVAE(nn.Module):
    def __init__(self, latent=latent_dim):
        super(ConvVAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(.2),

            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(.2),

            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(.2),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(.2),
        )

        self.fc1 = nn.Linear(mid_num, inter_dim)
        self.fc2 = nn.Linear(inter_dim, latent * 2)

        self.fcr2 = nn.Linear(latent, inter_dim)
        self.fcr1 = nn.Linear(inter_dim, mid_num)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(.2),

            nn.ConvTranspose2d(128, 64, 3, 2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(.2),

            nn.ConvTranspose2d(64, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(.2),

            nn.ConvTranspose2d(32, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(.2),

            nn.ConvTranspose2d(32, 16, 3, 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(.2),

            nn.ConvTranspose2d(16, 3, 4, 2),
            nn.Sigmoid()
        )

    def reparameterise(self, mu, logvar):
        epsilon = torch.randn_like(mu)
        return mu + epsilon * torch.exp(logvar / 2)

    def forward(self, x):
        batch = x.size(0)
        x = self.encoder(x)
        x = self.fc1(x.view(batch, -1))
        h = self.fc2(x)

        mu, logvar = h.chunk(2, dim=-1)
        z = self.reparameterise(mu, logvar)

        decode = self.fcr2(z)
        decode = self.fcr1(decode)
        recon_x = self.decoder(decode.view(batch, *mid_dim))

        return recon_x, mu, logvar


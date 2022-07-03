import torch
import torch.nn as nn
import torch.nn.functional as F

from loaddatasets import loadcifar


'''class VAE(nn.Module):
    def __init__(self, imgChannels=3, featureDim=32*32, zDim=32*32):
        super(VAE, self).__init__()

        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder
        self.encConv1 = nn.Conv2d(imgChannels, 12, 3, stride = 1,padding=1, bias=False )
        self.encConv2 = nn.Conv2d(12, 1, 3, stride = 1,padding=1, bias=False )
        self.encConv3 = nn.Conv2d(1, 1, 3, stride=1, padding=1, bias=False)
        self.encFC1 = nn.Linear(featureDim, zDim)
        self.encFC2 = nn.Linear(featureDim, zDim)
        self.bn1 = nn.BatchNorm2d(12)
        self.bn2 = nn.BatchNorm2d(3)
        self.bn3 = nn.BatchNorm2d(1)
        # Initializing the fully-connected layer and 2 convolutional layers for decoder
        self.decFC1 = nn.Linear(zDim, featureDim)
        self.decConv1 = nn.ConvTranspose2d(1, 12, kernel_size=3, stride=1, padding=1)
        self.decConv2 = nn.ConvTranspose2d(12, 12, kernel_size=3, stride=1, padding=1)
        self.decConv3 = nn.ConvTranspose2d(12, 3, kernel_size=3, stride=1, padding=1, bias=False)



    def encoder(self, x):

        # Input is fed into 2 convolutional layers sequentially
        # The output feature map are fed into 2 fully-connected layers to predict mean (mu) and variance (logVar)
        # Mu and logVar are used for generating middle representation z and KL divergence loss

        x = F.relu(self.encConv1(x))

        x = F.relu(self.encConv2(x))

        x = self.encConv3(x)

        x1 = x.view(-1,32*32)
        mu = self.encFC1(x1)
        logVar = self.encFC2(x1)
        return mu, logVar, x

    def reparameterize(self, mu, logVar):

        #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, z):

        # z is fed back into a fully-connected layers and then into two transpose convolutional layers
        # The generated output is the same size of the original input
        x = F.relu(self.decFC1(z))
        x = x.view(-1, 1, 32, 32)
        x = F.relu(self.decConv1(x))
        x = F.relu(self.decConv2(x))
        x = torch.sigmoid(self.decConv3(x))
        return x

    def forward(self, x):

        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mu, and logVar are returned for loss computation
        mu, logVar,x = self.encoder(x)
        z = self.reparameterize(mu, logVar)

        out = self.decoder(z)
        mid = z.view(-1,1,32,32)
        return mid , out, mu, logVar'''


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

            nn.ConvTranspose2d(16, 3, 4, 2, 4),
            nn.Sigmoid()
        )

    def reparameterise(self, mu, logvar):
        epsilon = torch.randn_like(mu)
        return mu + epsilon * torch.exp(logvar / 2)

    def forward(self, x):
        batch = x.size(0)
        x = self.encoder(x)
        mid = x.reshape(1,1,32,32)

        x = self.fc1(x.view(batch, -1))

        h = self.fc2(x)

        mu, logvar = h.chunk(2, dim=-1)

        z = self.reparameterise(mu, logvar)

        decode = self.fcr2(z)
        decode = self.fcr1(decode)

        recon_x = self.decoder(decode.view(batch, *mid_dim))

        return mid, recon_x, mu, logvar


'''print('cifar trainning starts')
x_S1, y_S1, x_S2, y_S2 = loadcifar()
a = ConvVAE()

out, mu, log_var = a(x_S1[1].unsqueeze(0).float())
log_var = torch.sigmoid(log_var)
mu = torch.sigmoid(mu)
kl_divergence = 0.5 * torch.sum(torch.exp(log_var) + torch.pow(mu, 2) - 1. - log_var)
print(kl_divergence)'''


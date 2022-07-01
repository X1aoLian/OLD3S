import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import random

class VAE(nn.Module):
    def __init__(self, imgChannels=3, featureDim=32*32, zDim=256):
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
        return mu, logVar,x

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

        return x, out, mu, logVar
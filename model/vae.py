import torch
import torch.nn as nn
import torch.nn.functional as F
from loaddatasets import *
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
import os
import random
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from scipy.stats import norm
import numpy as np
from scipy.stats import norm


latent_dim = 100
inter_dim = 256
mid_dim = (256, 2, 2)
mid_num = 1
for i in mid_dim:
    mid_num *= i

class ConvVAE(nn.Module):
    def __init__(self, latent=latent_dim):
        super(ConvVAE, self).__init__()



        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 6, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, 6, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, 6, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 3, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 16, 3, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.ConvTranspose2d(16, 3, 2, 2, 3),
            nn.Sigmoid()
        )

        self.fc1 = nn.Linear(mid_num, inter_dim)
        self.fc2 = nn.Linear(inter_dim, latent * 2)
        self.fcr2 = nn.Linear(latent, inter_dim)
        self.fcr1 = nn.Linear(inter_dim, mid_num)


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

        #recon_x = self.decoder(decode.view(-1,1,32,32))
        return z, recon_x, mu, logvar

class VAE_Mnist(nn.Module):

    def __init__(self, input_dim, h_dim, z_dim):
        # 调用父类方法初始化模块的state
        super(VAE_Mnist, self).__init__()

        self.input_dim = input_dim
        self.h_dim = h_dim
        self.z_dim = z_dim

        # 编码器 ： [b, input_dim] => [b, z_dim]
        self.fc1 = nn.Linear(input_dim, h_dim)  # 第一个全连接层
        self.fc2 = nn.Linear(h_dim, z_dim)  # mu
        self.fc3 = nn.Linear(h_dim, z_dim)  # log_var

        # 解码器 ： [b, z_dim] => [b, input_dim]
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, input_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, self.input_dim)
        mu, log_var = self.encode(x)
        sampled_z = self.reparameterization(mu, log_var)
        x_hat = self.decode(sampled_z)
        return sampled_z, x_hat, mu, log_var

    def encode(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc2(h)
        log_var = self.fc3(h)
        return mu, log_var
    def reparameterization(self, mu, log_var):
        sigma = torch.exp(log_var * 0.5)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps
    def decode(self, z):
        h = F.relu(self.fc4(z))
        x_hat = torch.sigmoid(self.fc5(h))
        return x_hat
class VAE_Shallow(nn.Module):

    def __init__(self, input_dim, h_dim, z_dim):
        # 调用父类方法初始化模块的state
        super(VAE_Shallow, self).__init__()

        self.input_dim = input_dim
        self.h_dim = h_dim
        self.z_dim = z_dim

        # 编码器 ： [b, input_dim] => [b, z_dim]
        self.fc1 = nn.Linear(input_dim, h_dim)  # 第一个全连接层
        self.fc2 = nn.Linear(h_dim, z_dim)  # mu
        self.fc3 = nn.Linear(h_dim, z_dim)  # log_var

        # 解码器 ： [b, z_dim] => [b, input_dim]
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, input_dim)

    def forward(self, x):
        mu, log_var = self.encode(x)
        sampled_z = self.reparameterization(mu, log_var)
        x_hat = self.decode(sampled_z)
        return sampled_z, x_hat, mu, log_var

    def encode(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc2(h)
        log_var = self.fc3(h)
        return mu, log_var
    def reparameterization(self, mu, log_var):
        sigma = torch.exp(log_var * 0.5)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps
    def decode(self, z):
        h = F.relu(self.fc4(z))
        x_hat = torch.sigmoid(self.fc5(h))
        return x_hat

def train():
    epochs = 10
    batch_size = 512

    best_loss = 1e9
    best_epoch = 0

    valid_losses = []
    train_losses = []
    '''pokemon_train = torchvision.datasets.FashionMNIST(
        root='./data',
        download=True,
        train=True,
        # Simply put the size you want in Resize (can be tuple for height, width)
        transform=torchvision.transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(hue=0.3),
                torchvision.transforms.ToTensor()]
        )
    )
    pokemon_valid = torchvision.datasets.FashionMNIST(
        root='./data',
        download=True,
        train=False,
        # Simply put the size you want in Resize (can be tuple for height, width)
        transform=torchvision.transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(hue=0.3),
                torchvision.transforms.ToTensor()]
        )
    )'''
    '''pokemon_train =  torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )
    pokemon_valid =  torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )'''
    kl_loss = lambda mu, logvar: -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    recon_loss = lambda recon_x, x: F.mse_loss(recon_x, x, size_average=False)

    #train_loader = DataLoader(pokemon_train, batch_size=batch_size, shuffle=True)
    #test_loader = DataLoader(pokemon_valid, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = VAE(2000,128,20)
    model.to(device)
    x1, y1,x2,y2 = loadreuter('EN_FR')
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        model.train()
        train_loss = 0.
        train_num = len(x1)

        for idx, x in enumerate(x1):

            batch = x.size(0)
            x = x.to(device)
            _, recon_x, mu, logvar = model(x)
            recon = recon_loss(recon_x, x)
            kl = kl_loss(mu, logvar)

            loss = recon + kl
            train_loss += loss.item()
            loss = loss / batch

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        print(f"Training loss {loss: .3f} \t Recon {recon / batch: .3f} \t KL {kl / batch: .3f} ")

        train_losses.append(train_loss / train_num)

    valid_loss = 0.
    valid_recon = 0.
    valid_kl = 0.
    valid_num = len(x1)
    model.eval()
    with torch.no_grad():
        for idx, x  in enumerate(x1):
            x = x.to(device)
            _, recon_x, mu, logvar = model(x)
            recon = recon_loss(recon_x, x)
            kl = kl_loss(mu, logvar)
            loss = recon + kl
            valid_loss += loss.item()
            valid_kl += kl.item()
            valid_recon += recon.item()

        valid_losses.append(valid_loss / valid_num)

        print(
            f"Valid loss {valid_loss / valid_num: .3f} \t Recon {valid_recon / valid_num: .3f} \t KL {valid_kl / valid_num: .3f} in epoch {epoch}")

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch

            torch.save(model.state_dict(), 'D:/pycharmproject/OLD3S/model/data/parameter_enfr/vae_model_1')
            print("Model saved")

def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    '''pokemon_valid = torchvision.datasets.SVHN(
        root='./data',
        split="test",
        download=True,
        transform=transforms.ToTensor()
    )'''
    '''pokemon_valid = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )'''
    pokemon_valid = torchvision.datasets.FashionMNIST(
        root='./data',
        download=False,
        train=True,
        # Simply put the size you want in Resize (can be tuple for height, width)
        transform=torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor()]
        )
    )
    test_loader = DataLoader(pokemon_valid, batch_size=1, shuffle=False)

    state = torch.load('best_model_mnist')
    model = VAE(784,128,20)
    model.load_state_dict(state)


    model.eval()

    to_pil_image = transforms.ToPILImage()
    cnt = 0
    for image,label in test_loader:
        if cnt>=10:      # 只显示3张图片
            break
        print(label)    # 显示label
        _, recon_x, mu, logvar = model(image)

        img = image[0]  # plt.imshow()只能接受3-D Tensor，所以也要用image[0]消去batch那一维
        img = img.numpy()  # FloatTensor转为ndarray
        img = np.transpose(img, (1, 2, 0))  # 把channel那一维放到最后
        # 显示图片
        plt.imshow(img)
        plt.show()
        rec = recon_x[0]  # plt.imshow()只能接受3-D Tensor，所以也要用image[0]消去batch那一维
        rec = rec.detach().numpy()  # FloatTensor转为ndarray
        rec = np.transpose(rec, (1, 2, 0))  # 把channel那一维放到最后
        # 显示图片
        plt.imshow(rec)
        plt.show()

        cnt += 1


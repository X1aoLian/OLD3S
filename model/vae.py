import torch
import torch.nn as nn
import torch.nn.functional as F
from loaddatasets import loadcifar
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


latent_dim = 128
inter_dim = 256
mid_dim = (256, 2, 2)
mid_num = 1
for i in mid_dim:
    mid_num *= i

class ConvVAE(nn.Module):
    def __init__(self, latent=latent_dim):
        super(ConvVAE, self).__init__()

        '''self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, 3),
            #nn.BatchNorm2d(12),
            nn.ReLU(),

            nn.Conv2d(8, 4, 3, 1, 1),
            #nn.BatchNorm2d(1),
            nn.ReLU(),

            #nn.Conv2d(1, 1,3, 1, 1),
            #nn.BatchNorm2d(1),
            #nn.ReLU(),
        )'''
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

            nn.ConvTranspose2d(16, 3, 2, 2, 3),

        )

        self.fc1 = nn.Linear(mid_num, inter_dim)
        self.fc2 = nn.Linear(inter_dim, latent * 2)
        self.fcr2 = nn.Linear(latent, inter_dim)
        self.fcr1 = nn.Linear(inter_dim, mid_num)



        '''self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 8, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 3, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(12),
            nn.ReLU(),
            #nn.ConvTranspose2d(12, 3, kernel_size=3, stride=1, padding=1, bias=False),
            #nn.BatchNorm2d(3)
        )'''

    def reparameterise(self, mu, logvar):
        epsilon = torch.randn_like(mu)
        return mu + epsilon * torch.exp(logvar / 2)

    def forward(self, x):
        batch = x.size(0)
        x1 = self.encoder(x)
        x = self.fc1(x1.view(batch, -1))
        h = self.fc2(x)
        mu, logvar = h.chunk(2, dim=-1)
        z = self.reparameterise(mu, logvar)
        decode = self.fcr2(z)
        decode = self.fcr1(decode)
        recon_x = self.decoder(decode.view(batch, *mid_dim))
        return z, recon_x, mu, logvar

class VAE(nn.Module):

    def __init__(self, input_dim, h_dim, z_dim):
        # 调用父类方法初始化模块的state
        super(VAE, self).__init__()

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
        """
        向前传播部分, 在model_name(inputs)时自动调用
        :param x: the input of our training model [b, batch_size, 1, 28, 28]
        :return: the result of our training model
        """
        batch_size = x.shape[0]  # 每一批含有的样本的个数
        # flatten  [b, batch_size, 1, 28, 28] => [b, batch_size, 784]
        # tensor.view()方法可以调整tensor的形状，但必须保证调整前后元素总数一致。view不会修改自身的数据，
        # 返回的新tensor与原tensor共享内存，即更改一个，另一个也随之改变。
        x = x.view(batch_size, self.input_dim)  # 一行代表一个样本

        # encoder
        mu, log_var = self.encode(x)
        # reparameterization trick
        sampled_z = self.reparameterization(mu, log_var)
        # decoder
        x_hat = self.decode(sampled_z)
        # reshape

        return sampled_z, x_hat, mu, log_var

    def encode(self, x):
        """
        encoding part
        :param x: input image
        :return: mu and log_var
        """
        h = F.relu(self.fc1(x))
        mu = self.fc2(h)
        log_var = self.fc3(h)

        return mu, log_var

    def reparameterization(self, mu, log_var):
        """
        Given a standard gaussian distribution epsilon ~ N(0,1),
        we can sample the random variable z as per z = mu + sigma * epsilon
        :param mu:
        :param log_var:
        :return: sampled z
        """
        sigma = torch.exp(log_var * 0.5)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps  # 这里的“*”是点乘的意思

    def decode(self, z):
        """
        Given a sampled z, decode it back to image
        :param z:
        :return:
        """
        h = F.relu(self.fc4(z))
        x_hat = torch.sigmoid(self.fc5(h))  # 图片数值取值为[0,1]，不宜用ReLU
        return x_hat

def train():
    epochs = 100
    batch_size = 512

    best_loss = 1e9
    best_epoch = 0

    valid_losses = []
    train_losses = []

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    pokemon_train = torchvision.datasets.SVHN(
        root='./data',
        split="train",
        download=True,
        transform=transforms.ToTensor()
    )
    pokemon_valid = torchvision.datasets.SVHN(
        root='./data',
        split="test",
        download=True,
        transform=transforms.ToTensor()
    )
    kl_loss = lambda mu, logvar: -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    recon_loss = lambda recon_x, x: F.mse_loss(recon_x, x, size_average=False)

    train_loader = DataLoader(pokemon_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(pokemon_valid, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ConvVAE()
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        model.train()
        train_loss = 0.
        train_num = len(train_loader.dataset)

        for idx, (x, label) in enumerate(train_loader):

            batch = x.size(0)
            x = x.to(device)
            recon_x, mu, logvar = model(x)
            recon = recon_loss(recon_x, x)
            kl = kl_loss(mu, logvar)

            loss = recon + kl
            train_loss += loss.item()
            loss = loss / batch

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx % 100 == 0:
                print(f"Training loss {loss: .3f} \t Recon {recon / batch: .3f} \t KL {kl / batch: .3f} in Step {idx}")

        train_losses.append(train_loss / train_num)

    valid_loss = 0.
    valid_recon = 0.
    valid_kl = 0.
    valid_num = len(test_loader.dataset)
    model.eval()
    with torch.no_grad():
        for idx, (x, label) in enumerate(test_loader):
            x = x.to(device)
            recon_x, mu, logvar = model(x)
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

            torch.save(model.state_dict(), 'best_model_pokemon')
            print("Model saved")

def test():


    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pokemon_valid = torchvision.datasets.SVHN(
        root='./data',
        split="test",
        download=True,
        transform=transforms.ToTensor()
    )
    test_loader = DataLoader(pokemon_valid, batch_size=1, shuffle=False)

    state = torch.load('best_model_pokemon')
    model = ConvVAE()
    model.load_state_dict(state)


    model.eval()

    to_pil_image = transforms.ToPILImage()
    cnt = 0
    for image,label in test_loader:
        if cnt>=3:      # 只显示3张图片
            break
        print(label)    # 显示label
        recon_x, mu, logvar = model(image)

        # 方法1：Image.show()
        # transforms.ToPILImage()中有一句
        # npimg = np.transpose(pic.numpy(), (1, 2, 0))
        # 因此pic只能是3-D Tensor，所以要用image[0]消去batch那一维
        img = to_pil_image(image[0])
        img.show()
        rec = to_pil_image(recon_x[0])
        rec.show()
        cnt += 1




import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import os



os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"




def show_images(images):
    images = torchvision.utils.make_grid(images)
    show_image(images)

def show_image(img):
    plt.imshow(img.permute(1, 2, 0), cmap="gray")
    plt.show()

'''trial = next(iter(trainloader))
images, labels = trial
print(labels)
show_images(images)'''

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), 1024, 1, 1)


class DCVAE(nn.Module):
    def __init__(self, image_dim, hidden_size, latent_size,image_channels=3):
        super(DCVAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 128, 3, 2,1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2,1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, 2,1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            Flatten(),
        )
        self.fc1 = nn.Linear(8192, hidden_size)
        self.encoder_mean = nn.Linear(hidden_size, latent_size)
        self.encoder_logvar = nn.Linear(hidden_size, latent_size)
        self.fc = nn.Linear(latent_size, hidden_size)
        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(hidden_size, 512, 4, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 2, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 2, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 3,2, 2),
            nn.Sigmoid()
        )

    def sample(self, log_var, mean):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc1(x)
        log_var = self.encoder_logvar(x)
        mean = self.encoder_mean(x)
        z = self.sample(log_var, mean)
        x = self.fc(z)
        x = self.decoder(x)
        return z, x, mean, log_var





def test():



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pokemon_valid = datasets.SVHN('./data', split='test', download=False,
                              transform=transforms.Compose([ transforms.ToTensor()]))
    pokemon_valid = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )
    test_loader = DataLoader(pokemon_valid, batch_size=1, shuffle=False)

    state = torch.load('best_model_pokemon')
    model = DCVAE( image_dim=32, hidden_size=1024, latent_size=100,image_channels=3)
    model.load_state_dict(state)


    model.eval()

    to_pil_image = transforms.ToPILImage()
    cnt = 0
    for image,label in test_loader:
        if cnt>=10:      # 只显示3张图片
            break
        print(label)    # 显示label
        image = image
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

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using %s for computation" % device)
    Newfeature = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(hue=0.3),
        torchvision.transforms.ToTensor()])
    project_dir = 'VAE/'
    dataset_dir = project_dir + 'datasets/'
    images_dir = project_dir + 'images/'
    model_dir = project_dir + 'model/'

    batch_size = 32  # number of inputs in each batch
    epochs = 10  # times to run the model on complete data
    image_size = 32
    hidden_size = 1024  # hidden dimension
    latent_size = 100  # latent vector dimension
    lr = 1e-4  # learning rate
    train_loss = []

    train_data = torchvision.datasets.SVHN(
        root='./data',
        split="train",
        download=True,
        transform=Newfeature
    )
    test_data = torchvision.datasets.SVHN(
        root='./data',
        split="test",
        download=True,
        transform=Newfeature
    )

    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    vae = DCVAE(image_dim=image_size, hidden_size=hidden_size, latent_size=latent_size, image_channels=3).to(device)
    # vae = VAE().to(device)
    optimizer = optim.Adam(vae.parameters(), lr=lr)
    # vae.train()
    print("start train")
    for epoch in range(epochs):
        for i, (images, _) in enumerate(trainloader):
            images = images.to(device)
            optimizer.zero_grad()
            _, reconstructed_image, mean, log_var = vae(images)
            CE = F.binary_cross_entropy(reconstructed_image, images, reduction='sum')
            # for VAE
            # CE = F.binary_cross_entropy(
            #             reconstructed_image, images.view(-1, input_size), reduction="sum"
            #         )
            KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
            loss = CE + KLD
            loss = loss / batch_size
            loss.backward()
            train_loss.append(loss.item())
            optimizer.step()

            if (i % 100 == 0):
                print("Loss:")
                print(loss.item() / len(images))
    torch.save(vae.state_dict(), 'best_model_svhn_2')
    vae.load_state_dict(torch.load('best_model_pokemon'))
    vae.eval()
    vectors = []
    with torch.no_grad():
        for i, (images, labels) in enumerate(testloader):
            images = images.to(device)

            _, reconstructed_image, mean, log_var = vae(images)
            reconstructed_image = reconstructed_image.view(-1, 1, image_size, image_size)
            temp = list(zip(labels.tolist(), mean.tolist()))
            for x in temp:
                vectors.append(x)
            if (i % 100 == 0):
                show_images(reconstructed_image.cpu())
                img_name = images_dir + "evaluation/DCVAE/" + str(i).zfill(3)
                # img_name = images_dir + "evaluation/VAE/" + str(i).zfill(3)
                # plt.savefig(img_name)
                plt.show()

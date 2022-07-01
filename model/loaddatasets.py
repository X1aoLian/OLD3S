import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torch import nn
from sklearn.utils import shuffle

Newfeature = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(hue=0.3),
        torchvision.transforms.ToTensor()])

def loadcifar():


    cifar10_original = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    cifar10_color = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=Newfeature
    )
    x_S1 = torch.Tensor(cifar10_original.data)
    x_S2 = torch.Tensor(cifar10_color.data)
    y_S1, y_S2 = torch.Tensor(cifar10_original.targets), torch.Tensor(cifar10_color.targets)
    x_S1, y_S1 = shuffle(x_S1, y_S1, random_state=30)
    x_S2, y_S2 = shuffle(x_S2, y_S2, random_state=30)
    x_S1 = torch.transpose(x_S1, 3, 2)
    x_S1 = torch.transpose(x_S1, 2, 1)
    x_S2 = torch.transpose(x_S2, 3, 2)
    x_S2 = torch.transpose(x_S2, 2, 1)

    return x_S1, y_S1, x_S2, y_S2

def loadsvhn():
    svhm_original = torchvision.datasets.SVHN(
        root='./data',
        split="train",
        download=True,
        transform=transforms.ToTensor()
    )

    svhm_color = torchvision.datasets.SVHN(
        root='./data',
        split="train",
        download=True,
        transform=Newfeature
    )
    x_S1 = torch.Tensor(svhm_original.data)
    x_S2 = torch.Tensor(svhm_color.data)
    y_S1, y_S2 = svhm_original.labels, svhm_color.labels
    for i in range(len(y_S1)):
        if y_S1[i] == 10:
            y_S1[i] = 0
    y_S1, y_S2 = torch.Tensor(y_S1), torch.Tensor(y_S1)
    x_S1, y_S1 = shuffle(x_S1, y_S1, random_state=30)
    x_S2, y_S2 = shuffle(x_S2, y_S2, random_state=30)
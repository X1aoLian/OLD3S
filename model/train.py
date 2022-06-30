import random
import torch
import numpy as np
import torchvision
import matplotlib as mpl
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision import models
from torch import nn
from sklearn.utils import shuffle
from OLD3S import OLD3S
from deepFesl_nohedge import Resnet18_Cifar
from Linear_cifar import SimpleLinear_Cifar
from FES import FES
from deepFesl_linear import deepFESL_linear


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(30)




def loaddatasets():
    Newfeature = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(hue=0.3),
        torchvision.transforms.ToTensor()])

    cifar10_original = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=False,
        transform=transforms.ToTensor()
    )

    cifar10_color = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=False,
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

if __name__ == '__main__':
    x_S1, y_S1, x_S2, y_S2 = loaddatasets()
    begin = OLD3S(x_S1, y_S1, x_S2, y_S2, 50000, 5000)
    begin.SecondPeriod()





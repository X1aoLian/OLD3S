import numpy as np
import pandas as pd
import torch
import torchvision
from sklearn import preprocessing
from torchvision.transforms import transforms
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
    svhm_original =  torchvision.datasets.SVHN('./data', split='train', download=False,
                               transform=transforms.Compose([ transforms.ToTensor()]))
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
    return x_S1, y_S1, x_S2, y_S2

def loadmagic():
    data = pd.read_csv(r"./data/magic04_X.csv", header=None).values
    label = pd.read_csv(r"./data/magic04_y.csv", header=None).values
    for i in label:
        if i[0] == -1:
            i[0] = 0
    rd1 = np.random.RandomState(1314)
    data = preprocessing.scale(data)
    matrix1 = rd1.random((10, 30))
    x_S2 = np.dot(data, matrix1)
    x_S1 = torch.sigmoid(torch.Tensor(data))
    x_S2 = torch.sigmoid(torch.Tensor(x_S2))

    y_S1, y_S2 = torch.Tensor(label), torch.Tensor(label)

    x_S1, y_S1 = shuffle(x_S1, y_S1, random_state=50)
    x_S2, y_S2 = shuffle(x_S2, y_S2, random_state=50)
    return x_S1, y_S1, x_S2, y_S2

def loadadult():
    df1 = pd.read_csv(r"D:/pycharmproject/pytorch/OLD3/adult/data/adult.data", header=1)
    df1.columns=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o']
    le = preprocessing.LabelEncoder()
    le.fit(df1.o)
    df1['o'] = le.transform(df1.o)
    le.fit(df1.b)
    df1['b'] = le.transform(df1.b)
    le.fit(df1.d)
    df1['d'] = le.transform(df1.d)
    le.fit(df1.f)
    df1['f'] = le.transform(df1.f)
    le.fit(df1.g)
    df1['g'] = le.transform(df1.g)
    le.fit(df1.h)
    df1['h'] = le.transform(df1.h)
    le.fit(df1.i)
    df1['i'] = le.transform(df1.i)
    le.fit(df1.j)
    df1['j'] = le.transform(df1.j)
    le.fit(df1.n)
    df1['n'] = le.transform(df1.n)
    data = np.array(df1.iloc[:, :-1])
    label = np.array(df1.o)
    rd1 = np.random.RandomState(1314)
    data = preprocessing.scale(data)
    matrix1 = rd1.random((14, 30))
    x_S2 = np.dot(data, matrix1)
    x_S1 = torch.sigmoid(torch.Tensor(data))
    x_S2 = torch.sigmoid(torch.Tensor(x_S2))
    y_S1, y_S2 = torch.Tensor(label), torch.Tensor(label)
    x_S1, y_S1 = shuffle(x_S1, y_S1, random_state=30)
    x_S2, y_S2 = shuffle(x_S2, y_S2, random_state=30)
    return x_S1, y_S1, x_S2, y_S2

def loadreuter(name):
    x_S1 = torch.Tensor(torch.load('./data/' + name +'/x_S1_pca'))
    y_S1 = torch.Tensor(torch.load('./data/' + name +'/y_S1_multiLinear'))
    x_S2 = torch.Tensor(torch.load('./data/' + name +'/x_S2_pca'))
    y_S2 = torch.Tensor(torch.load('./data/' + name +'/y_S2_multiLinear'))
    return x_S1, y_S1, x_S2, y_S2

def loadmnist():
    mnist_original = torchvision.datasets.FashionMNIST(
        root='./data',
        download=True,
        train=True,
        # Simply put the size you want in Resize (can be tuple for height, width)
        transform=torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor()]
        ),
    )
    mnist_color = torchvision.datasets.FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(hue=0.3),
                torchvision.transforms.ToTensor()]
        ),
    )
    x_S1 = mnist_original.data
    x_S2 = mnist_color.data
    y_S1, y_S2 = mnist_original.targets, mnist_color.targets
    x_S1, y_S1 = shuffle(x_S1, y_S1, random_state=1000)
    x_S2, y_S2 = shuffle(x_S2, y_S2, random_state=1000)
    return x_S1, y_S1, x_S2, y_S2


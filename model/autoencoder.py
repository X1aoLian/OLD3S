import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):    # BasicBlock from ResNet [He et al.2016]
    EXPANSION = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.EXPANSION*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.EXPANSION*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.EXPANSION*planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class AutoEncoder_Deep(nn.Module):
    def __init__(self):
        super(AutoEncoder_Deep,self).__init__()
        self.conv1 = nn.Conv2d(3, 12, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.ConvTranspose2d(12, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(1)
        self.encoder = BasicBlock(12, 1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1, 12, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.ConvTranspose2d(12, 12, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(3)
        )

    def forward(self, x):

        encoded = self.encoder(F.relu(self.bn1(self.conv1(x))))     # maps the feature size from 3*32*32 to 32*32
        decoded = self.decoder(encoded)
        return encoded, decoded

class AutoEncoder_Shallow(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(AutoEncoder_Shallow, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(inplanes, outplanes),
            nn.ReLU()

        )
        self.decoder = nn.Sequential(
            nn.Linear(outplanes, inplanes),
            nn.ReLU()
        )

    def forward(self, x):
        encoder = self.encoder(x)
        decoder = self.decoder(encoder)
        return encoder, decoder


class BasicBlock_Mnist(nn.Module):    # BasicBlock from ResNet [He et al.2016]
    EXPANSION = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_Mnist, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.EXPANSION*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.EXPANSION*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.EXPANSION*planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class AutoEncoder_Mnist(nn.Module):
    def __init__(self):
        super(AutoEncoder_Mnist,self).__init__()
        self.conv1 = nn.Conv2d(1, 12, kernel_size=3, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.ConvTranspose2d(12, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(3)
        self.encoder = BasicBlock(12, 1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1, 12, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.ConvTranspose2d(12, 12, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.ConvTranspose2d(12, 1, kernel_size=3, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(1)
        )

    def forward(self, x):

        encoded = self.encoder(F.relu(self.bn1(self.conv1(x))))     # maps the feature size from 3*32*32 to 32*32
        decoded = self.decoder(encoded)
        return encoded, decoded




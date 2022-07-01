import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class BasicBlock(nn.Module):
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
        if stride != 1 or in_planes != self.EXPANSION * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.EXPANSION * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.EXPANSION * planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    EXPANSION = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, self.EXPANSION * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.EXPANSION * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.EXPANSION * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.EXPANSION * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.EXPANSION * planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        woha = self.shortcut(x)
        out += woha
        out = F.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        super(ResNet, self).__init__()
        self.num_blocks = num_blocks
        self.in_planes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.num_classes = num_classes
        self.hidden_layers = []
        self.output_layers = []

        self._make_layer(block, 64, num_blocks[0], stride=1)

        self._make_layer(block, 128, num_blocks[1], stride=2)

        self._make_layer(block, 256, num_blocks[2], stride=2)

        self._make_layer(block, 512, num_blocks[3], stride=2)


        self.output_layers.append(self._make_mlp1(64, 2))  # 32
        self.output_layers.append(self._make_mlp1(64, 2))  # 32
        self.output_layers.append(self._make_mlp2(128, 2))  # 16
        self.output_layers.append(self._make_mlp2(128, 2))  # 16
        self.output_layers.append(self._make_mlp3(256, 2))  # 8
        self.output_layers.append(self._make_mlp3(256, 2))  # 8
        self.output_layers.append(self._make_mlp4(512, 2))  # 4
        self.output_layers.append(self._make_mlp4(512, 2))  # 4

        self.hidden_layers = nn.ModuleList(self.hidden_layers)  #
        self.output_layers = nn.ModuleList(self.output_layers)  #

    def _make_mlp1(self, in_planes,  kernel_size_pool, padding_pool=0):
        classifier = nn.Sequential(
            nn.MaxPool2d(kernel_size=kernel_size_pool, padding=padding_pool),
            nn.MaxPool2d(kernel_size=kernel_size_pool, padding=padding_pool),
            #nn.MaxPool2d(kernel_size=kernel_size_pool, padding=padding_pool),
            nn.Flatten(),
            nn.Linear(in_planes*8*8, in_planes*8*2),
            nn.Linear(in_planes*8*2, 256),
            nn.Linear(256, self.num_classes),
        )
        return classifier
    def _make_mlp2(self, in_planes, kernel_size_pool, padding_pool=0):
        classifier = nn.Sequential(
            nn.MaxPool2d(kernel_size=kernel_size_pool, padding=padding_pool,ceil_mode=True),
            nn.MaxPool2d(kernel_size=kernel_size_pool, padding=padding_pool,ceil_mode=True),
            # nn.MaxPool2d(kernel_size=kernel_size_pool, padding=padding_pool),
            nn.Flatten(),
            nn.Linear(in_planes*4*4, self.num_classes),
        )
        return classifier
    def _make_mlp3(self, in_planes, kernel_size_pool, padding_pool=0):
        classifier = nn.Sequential(
            nn.MaxPool2d(kernel_size=kernel_size_pool, padding=padding_pool,ceil_mode=True),
            nn.AvgPool2d(kernel_size=kernel_size_pool, padding=padding_pool,ceil_mode=True),
            # nn.MaxPool2d(kernel_size=kernel_size_pool, padding=padding_pool),
            nn.Flatten(),
            nn.Linear(in_planes*2*2, self.num_classes),
        )
        return classifier

    def _make_mlp4(self, in_planes, kernel_size_pool, padding_pool=0):
        classifier = nn.Sequential(
            nn.AvgPool2d(kernel_size=kernel_size_pool, padding=padding_pool,ceil_mode=True),
            # nn.MaxPool2d(kernel_size=kernel_size_pool, padding=padding_pool),
            nn.Flatten(),
            nn.Linear(in_planes*2*2, self.num_classes),
        )
        return classifier



    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        for stride in strides:
            self.hidden_layers.append(block(self.in_planes, planes, stride))
            self.in_planes = block.EXPANSION * planes

    def forward(self, x):
        hidden_connections = []
        hidden_connections.append(F.relu(self.bn1(self.conv1(x))))

        for i in range(len(self.hidden_layers)):
            hidden_connections.append(self.hidden_layers[i](hidden_connections[i]))

        output_class = []
        for i in range(len(self.output_layers)):
            #print(hidden_connections[i].shape)
            output_class.append(self.output_layers[i](hidden_connections[i]))

        return output_class


def Dynamic_ResNet18():

    return ResNet(BasicBlock, [1, 2, 2, 2])




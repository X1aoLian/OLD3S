import torch
from torch import nn
import torch.nn.functional as F


class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes):
        super(BasicBlock, self).__init__()
        self.Linear1 = nn.Linear(
            in_planes, planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.Linear1(x)
        out = self.relu(out)
        return out


class MLP(nn.Module):

    def __init__(self, in_planes, num_classes=2):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        super(MLP, self).__init__()
        self.in_planes = in_planes
        self.num_classes = num_classes
        self.Linear = nn.Linear(self.in_planes, self.in_planes)
        self.hidden_layers = []
        self.output_layers = []

        self.hidden_layers.append(BasicBlock(self.in_planes,self.in_planes))
        self.hidden_layers.append(BasicBlock(self.in_planes, self.in_planes))
        self.hidden_layers.append(BasicBlock(self.in_planes, self.in_planes))
        self.hidden_layers.append(BasicBlock(self.in_planes, self.in_planes))

        self.output_layers.append(self._make_mlp1(self.in_planes))
        self.output_layers.append(self._make_mlp2(self.in_planes))
        self.output_layers.append(self._make_mlp3(self.in_planes))
        self.output_layers.append(self._make_mlp4(self.in_planes))
        self.output_layers.append(self._make_mlp4(self.in_planes))

        self.hidden_layers = nn.ModuleList(self.hidden_layers)  #
        self.output_layers = nn.ModuleList(self.output_layers)  #

    def _make_mlp1(self, in_planes):
        classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_planes, 2),

        )
        return classifier

    def _make_mlp2(self, in_planes):
        classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_planes, 2),

        )
        return classifier

    def _make_mlp3(self, in_planes):
        classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_planes, 2),

        )
        return classifier

    def _make_mlp4(self, in_planes):
        classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_planes, 2),

        )
        return classifier

    def _make_mlp5(self, in_planes):
        classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_planes, 2),

        )
        return classifier

    def forward(self, x):
        hidden_connections = []
        hidden_connections.append(F.relu(self.Linear(x)))

        for i in range(len(self.hidden_layers)):
            hidden_connections.append(self.hidden_layers[i](hidden_connections[i]))

        output_class = []
        for i in range(len(self.output_layers)):
            output = self.output_layers[i](hidden_connections[i])
            output_class.append(output)

        return output_class
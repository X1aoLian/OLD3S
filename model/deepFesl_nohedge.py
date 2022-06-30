import numpy as np
import torch
import torch.nn as nn
import torchvision
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from torchvision import transforms
import random
from autoencoder import AutoEncoder 
from model import Dynamic_ResNet18
import torch.nn.functional as F
import copy
import math

def plot(x_axis,y_axis,title,y_label,save):
    plt.figure()
    plt.plot(x_axis, y_axis)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel(y_label)
    if save==False:
        plt.show()
    else :
        path= "/home/hlian001/fesdl/Images/" +title+".png"
        plt.savefig(path)

'''class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out'''
    

    
def plot(x_axis,y_axis,title,y_label,save):
    plt.figure()
    plt.plot(x_axis, y_axis)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel(y_label)
    if save==False:
        plt.show()
    else :
        path= title+".png"
        plt.savefig(path)


class Resnet18_Cifar:
    def __init__(self, x_S1, y_S1, x_S2, y_S2, T1, t):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.correct = 0
        self.accuracy = 0

        self.T1 = T1
        self.t = t
        self.B = self.T1 - self.t
        self.data1 = x_S1
        self.data2 = x_S2
        self.label1 = y_S1
        self.label2 = y_S2
        self.a_1 = 0.8
        self.a_2 = 0.2
        self.cl_1 = [ ]
        self.cl_2 = [ ]
        
        self.CELoss = nn.CrossEntropyLoss()
        self.BCELoss = nn.BCELoss()
        self.SmoothL1Loss = nn.SmoothL1Loss()
        "For plot"
        self.Figure_i = [ ]
        self.Accuracy = [ ]
        self.autoencoder = AutoEncoder().to(self.device)
        
        self.autoencoder_2 = AutoEncoder().to(self.device)
        
    def Train_T1(self):
        self.classifier_1 = Dynamic_ResNet18().to(self.device)
        data1 = self.data1  # B donate the cashe begins time and B+t donate during the cashe time
        data2 = self.data2[self.B:]
        "use the Network"
        
        optimizer_1 =torch.optim.SGD(self.classifier_1.parameters(), lr=0.01)
        optimizer_2 = torch.optim.SGD(self.autoencoder.parameters(), lr = 0.01)
        
        b = -0.01
        for (i, x) in enumerate(data1):
            self.i = i
            x1 = x.unsqueeze(0).float().to(self.device)
            y = self.label1[i]
            y = y.unsqueeze(0).long().to(self.device)

            if self.i < self.B:  # Before evolve
                encoded, decoded = self.autoencoder(x1)

                optimizer_2.zero_grad()
                loss_classifier_1, output = self.HB_Fit(self.classifier_1,
                                                              encoded, y, optimizer_1)

                loss_2 = self.BCELoss(torch.sigmoid(decoded), x1)
          

                loss_2.backward()

                optimizer_2.step()
                y_hat = output
            else:  # Evolving start
                x2 = data2[self.i - self.B].unsqueeze(0).float().to(self.device)
                if i == self.B:
                    self.classifier_2 = copy.deepcopy(self.classifier_1)
                    torch.save(self.classifier_1.state_dict(), '/home/hlian001/fesdl/data/parameter_nohedge/net_model1.pth')
                    print("model1 保存成功")
                    optimizer_1_1 = torch.optim.SGD(self.classifier_1.parameters(), lr=0.01)
                    optimizer_1_2 = torch.optim.SGD(self.classifier_2.parameters(), lr=0.01)
                    optimizer_2_2 = torch.optim.SGD(self.autoencoder_2.parameters(), lr = 0.01)
                      
                encoded_1, decoded_1 = self.autoencoder(x1) #torch.relu(self.conv1(x1))
                encoded_2, decoded_2 = self.autoencoder_2(x2)
               
                optimizer_2_2.zero_grad()
                loss_1_1, y_hat_1 = self.HB_Fit(self.classifier_1, encoded_1, y, optimizer_1_1)
                loss_1_2, y_hat_2 = self.HB_Fit(self.classifier_2, encoded_2, y, optimizer_1_2)
                
                y_hat = self.a_1 * y_hat_1 + self.a_2 * y_hat_2
                
                self.cl_1.append(loss_1_1)
                self.cl_2.append(loss_1_2)
        
                if len(self.cl_1) == 100:
                    self.cl_1.pop(0)
                    self.cl_2.pop(0)

                try:
                    a_cl_1 = math.exp(b * sum(self.cl_1))
                    a_cl_2 = math.exp(b * sum(self.cl_2))
                    self.a_1 = (a_cl_1) / (a_cl_2 + a_cl_1)

                except OverflowError:
                    self.a_1 = float('inf')

                self.a_2 = 1 - self.a_1
                
                
                loss_2_2 = self.BCELoss(torch.sigmoid(decoded_2), x2)
                loss_2_0 = self.SmoothL1Loss(encoded_2, encoded_1)      # Regularizer to enforce the latent representations' similarity
                loss_2 = loss_2_0+loss_2_2
                loss_2.backward(retain_graph = True)
                optimizer_2_2.step()
                
             
            _, predicted = torch.max(y_hat.data, 1)
            self.correct += (predicted == y).item()
            
            if i == 0:
                print("finish 0")
            if (i + 1) % 100 == 0:
                print("step : %d" % (i + 1), end=", ")
                print("correct: %d" % (self.correct))

            if (i + 1) % 1000 == 0:
                self.accuracy = self.correct / 1000
                self.Figure_i.append(i)
                self.Accuracy.append(self.accuracy)
                self.correct = 0
                print("Accuracy: ", self.accuracy)
                
        torch.save(self.classifier_2.state_dict(), '/home/hlian001/fesdl/data/parameter_nohedge/net_model2.pth')
  

    def Train_T2(self):
        self.Train_T1()
        self.correct = 0
        "load data for T2"
        data2 = self.data2[:self.B]
        
        
        "use the Network"
        net_model1 = Dynamic_ResNet18().to(self.device)
        pretrain_dict = torch.load('/home/hlian001/fesdl/data/parameter_nohedge/net_model1.pth')    # One model, no ensembling
        model_dict = net_model1.state_dict()
        pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
        model_dict.update(pretrain_dict)
        net_model1.load_state_dict(model_dict)
        net_model1.to(self.device)
        
        net_model2 = Dynamic_ResNet18().to(self.device)
        pretrain_dict = torch.load('/home/hlian001/fesdl/data/parameter_nohedge/net_model2.pth')    # One model, no ensembling
        model_dict = net_model2.state_dict()
        pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
        model_dict.update(pretrain_dict)
        net_model2.load_state_dict(model_dict)
        net_model2.to(self.device)
 
        optimizer_1 = torch.optim.SGD(net_model1.parameters(), lr=0.01)
        optimizer_2 = torch.optim.SGD(net_model2.parameters(), lr=0.01)
        optimizer_3 = torch.optim.SGD(self.autoencoder_2.parameters(), lr = 0.01)
        "train the network"
        
        self.a_1 = 0.2
        self.a_2 = 0.8
        self.cl_1 = [ ]
        self.cl_2 = [ ]
        b = -0.01
        for (i, x) in enumerate(data2):
            x = x.unsqueeze(0).float().to(self.device)
            
            y = self.label2[i]
            y = y.unsqueeze(0).long().to(self.device)
            
            encoded, decoded = self.autoencoder_2(x)
            optimizer_3.zero_grad()
            
            loss2, y_hat_2 = self.HB_Fit(net_model2, encoded, y, optimizer_2)
            loss1, y_hat_1 = self.HB_Fit(net_model1, encoded, y, optimizer_1)
            
            loss_3 = self.BCELoss(torch.sigmoid(decoded), x)
            loss_3.backward()
            optimizer_3.step()
            
            y_hat = self.a_1 * y_hat_1 + self.a_2 * y_hat_2
                
            self.cl_1.append(loss1)
            self.cl_2.append(loss2)
            if len(self.cl_1) == 100:
                self.cl_1.pop(0)
                self.cl_2.pop(0)

            try:
                a_cl_1 = math.exp(b * sum(self.cl_1))
                a_cl_2 = math.exp(b * sum(self.cl_2))
                self.a_1 = (a_cl_1) / (a_cl_2 + a_cl_1)
            except OverflowError:
                self.a_1 = float('inf')

            self.a_2 = 1 - self.a_1
            
            _, predicted = torch.max(y_hat.data, 1)
            self.correct += (predicted == y).item()
            

            if i == 0:
                print("finish 1")
            if (i + 1) % 100 == 0:
                print("step : %d" % (i + 1), end=", ")
                print("correct: %d" % (self.correct))

            if (i + 1) % 1000 == 0:
                self.accuracy = self.correct / 1000
                self.Figure_i.append(i)
                self.Accuracy.append(self.accuracy)
                self.correct = 0
                print("Accuracy: ", self.accuracy)
                
        torch.save(self.Accuracy, '/home/hlian001/fesdl/data/nohedge_accuracy')
   

    def zero_grad(self, model):
        for child in model.children():
            for param in child.parameters():
                if param.grad is not None:
                    # param.grad.detach_()
                    param.grad.zero_()  # data.fill_(0)


    def updateLoss(self, model, x, y, optimizer):
        optimizer.zero_grad()
        prediction = model.forward(x)
        loss = self.CELoss(prediction, y)
        loss.backward(retain_graph=True)
        optimizer.step()
        return loss, prediction
        
    def HB_Fit(self, model, X, Y, optimizer, block_split=6):  # hedge backpropagation
        predictions_per_layer = model.forward(X)

        output = predictions_per_layer[-1]
        loss = self.CELoss(output, Y)

        optimizer.zero_grad()

        loss.backward(retain_graph=True)
        optimizer.step()

    
        return loss, output
        

import numpy as np
import torch
import torch.nn as nn
import torchvision
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from torchvision import transforms
import random

class MLP(nn.Module):
    def __init__(self,inplanes, classes):
        super(MLP, self).__init__()
        self.inplanes = inplanes
        self.classes = classes
        self.model = nn.Sequential(

            nn.Linear(self.inplanes,self.classes),

        )
    def forward(self, x):

        x = self.model(x)
        return x

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




class SimpleLinear_Cifar:
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

        self.CELoss = nn.CrossEntropyLoss()
        self.BCELoss = nn.BCELoss()
        self.SmoothL1Loss = nn.SmoothL1Loss()
        "For plot"
        self.Figure_i = []
        self.Accuracy = []

    def Train_T1(self):
        data1 = self.data1  # B donate the cashe begins time and B+t donate during the cashe time
        data2 = self.data2[self.B:]
        "use the Network"
        self.classifier_1 = MLP(3 * 32 * 32 * 2, 10).to(self.device).to(self.device)
        optimizer_classifier_1 = torch.optim.SGD(self.classifier_1.parameters(), lr=0.01)

        for (i, x) in enumerate(data1):
            self.i = i
            mid_x = x.flatten().unsqueeze(0).float().to(self.device)
            zeors = torch.zeros((1,3*32*32)).to(self.device)
            x = torch.cat((mid_x, zeors), dim = 1).to(self.device)
            label = self.label1[i]
            label = label.unsqueeze(0).long().to(self.device)

            if self.i < self.B:  # Before evolve
                #
                output, loss_classifier_1 = self.updateLoss(self.classifier_1,

                                                            x, label, optimizer_classifier_1)

            else:  # Evolving start

                x2 = data2[self.i - self.B]
                mid_x2 = x2.flatten().unsqueeze(0).float().to(self.device)
                x_new = torch.cat((x,mid_x2), dim = 1)

                if i == self.B:
                    optimizer_classifier_2 = torch.optim.SGD(self.classifier_1.parameters(), lr=0.01)
                output, loss_classifier_1 = self.updateLoss(self.classifier_1,
                                                            x, label, optimizer_classifier_1)


            _, predicted = torch.max(output.data, 1)
            self.correct += (predicted == label).item()

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

    def Train_T2(self):
        self.Train_T1()
        self.correct = 0
        "load data for T2"
        data2 = self.data2[:self.B]
        "use the Network"
        optimizer_classifier_1 = torch.optim.SGD(self.classifier_1.parameters(), lr=0.01)

        "train the network"
        for (i, x) in enumerate(data2):
            x = x.flatten().unsqueeze(0).float().to(self.device)
            zeros = torch.zeros((1,3*32*32)).to(self.device)
            x_new = torch.cat((zeros,x), dim = 1).to(self.device)
            y = self.label2[i].unsqueeze(0).long().to(self.device)

            output_2, loss_classifier_2 = self.updateLoss(self.classifier_1,
                                                          x_new, y, optimizer_classifier_1)
            _, predicted = torch.max(output_2.data, 1)
            self.correct += (predicted == y).item()

            if i == 0:
                print("finish 0")
            if (i + 1) % 100 == 0:
                print("step : %d" % (i + 1), end=", ")
                print("correct: %d" % (self.correct))
            if (i + 1) % 1000 == 0:
                self.accuracy = self.correct / 1000
                self.Figure_i.append(i + 50000)
                self.Accuracy.append(self.accuracy)
                self.correct = 0
                print("Accuracy: ", self.accuracy)
                
        torch.save(self.Accuracy, '/home/hlian001/fesdl/data/Linear_accuracy')
        torch.save(self.Figure_i, '/home/hlian001/fesdl/data/y_times')
       

    def zero_grad(self, model):
        for child in model.children():
            for param in child.parameters():
                if param.grad is not None:
                    # param.grad.detach_()
                    param.grad.zero_()  # data.fill_(0)


    def isEqual(self, predict, label):
        predictor = predict.argmax()

        real = label.argmax()
        return predictor == real

    def updateLoss(self, model, x, y, optimizer):
        optimizer.zero_grad()
        prediction = model.forward(x)

        loss = self.CELoss(prediction, y)
        loss.backward(retain_graph=True)
        optimizer.step()
        return prediction, loss




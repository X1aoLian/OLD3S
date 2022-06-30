import numpy as np
import torch
import torch.nn as nn
import torchvision
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from torchvision import transforms
import random
import math
from matplotlib import pyplot as plt
from model_nohedge import Dynamic_ResNet18
import torch.nn.functional as F

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
        
def mse(predictions, targets):
    return ((predictions - targets) ** 2).mean()   
    


class FES:
    def __init__(self, data_S1, label_S1, data_S2, label_S2, T1, t):
      

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.correct = 0
        self.accuracy = 0
        self.T1 = T1  # for data come from S1
        self.t = t
        self.B = self.T1 - self.t  # start of an evolving period of time
        self.x_S1 = data_S1
        self.y_S1 = label_S1
        self.x_S2 = data_S2
        self.y_S2 = label_S2
        self.BCELoss = nn.BCELoss()
        self.SmoothL1Loss = nn.SmoothL1Loss()
        self.CELoss = nn.CrossEntropyLoss()
        "For plot"
        self.lossFigure_i = []
        self.Accuracy = []

        #self.classifier_1 = Dynamic_ResNet18().to(self.device)
        #self.classifier_2 = Dynamic_ResNet18().to(self.device)
        self.classifier_1 = MLP(3*32*32,10).to(self.device)
        self.classifier_2 = MLP(3*32*32,10).to(self.device)

    def Time_1(self):

        optimizer_classifier_1 = torch.optim.Adam(self.classifier_1.parameters(), lr=0.0001)

        self.a_1 = 0.8
        self.a_2 = 0.2
        self.cl_1 = []
        self.cl_2 = []
        b = -0.01

        for (i, x) in enumerate(self.x_S1):

            self.i = i
            x1 = x.flatten().unsqueeze(0).float().to(self.device)
       

            y = self.y_S1[i].unsqueeze(0).long().to(self.device)
            
            '''if self.y_S1[i] == 0:
                y1 = torch.Tensor([1,0,0,0,0,0,0,0,0,0]).reshape(1,10).float().to(self.device)
            elif self.y_S1[i] == 1:
                y1 = torch.Tensor([0,1,0,0,0,0,0,0,0,0]).reshape(1,10).float().to(self.device)
            elif self.y_S1[i] == 2:
                y1 = torch.Tensor([0,0,1,0,0,0,0,0,0,0]).reshape(1,10).float().to(self.device)
            elif self.y_S1[i] == 3:
                y1 = torch.Tensor([0,0,0,1,0,0,0,0,0,0]).reshape(1,10).float().to(self.device)
            elif self.y_S1[i] == 4:
                y1 = torch.Tensor([0,0,0,0,1,0,0,0,0,0]).reshape(1,10).float().to(self.device)
            elif self.y_S1[i] == 5:
                y1 = torch.Tensor([0,0,0,0,0,1,0,0,0,0]).reshape(1,10).float().to(self.device)
            elif self.y_S1[i] == 6:
                y1 = torch.Tensor([0,0,0,0,0,0,1,0,0,0]).reshape(1,10).float().to(self.device)
            elif self.y_S1[i] == 7:
                y1 = torch.Tensor([0,0,0,0,0,0,0,1,0,0]).reshape(1,10).float().to(self.device)
            elif self.y_S1[i] == 9:
                y1 = torch.Tensor([0,0,0,0,0,0,0,0,1,0]).reshape(1,10).float().to(self.device)
           
            else:
                y1 = torch.Tensor([0,0,0,0,0,0,0,0,0,1]).reshape(1,10).float().to(self.device)'''
           
            
            '''train Autoencoder'''
            if self.i < self.B:  # Before evolve
                
             
                prediction, loss_classifier_1 = self.updateLoss(self.classifier_1,
                                                           x1, y, optimizer_classifier_1)
                


            else:  # Evolving start
                x2 = self.x_S2[self.i].flatten().unsqueeze(0).float().to(self.device)
                if i == self.B:
                    self.fn = nn.Linear(3*32*32, 3*32*32).to(self.device)
                    optimizer_map = torch.optim.SGD(self.fn.parameters(), lr = 0.01)
                    optimizer_classifier_2 = torch.optim.Adam(self.classifier_2.parameters(), lr=0.0001)
                    
                new_x2 = x2.flatten().float().to(self.device)
                new_x1 = self.fn(new_x2).float().to(self.device)
                old_x1 = x1.flatten().float().to(self.device)
                optimizer_map.zero_grad()
                loss_map = self.SmoothL1Loss(old_x1, new_x1)
                loss_map.backward(retain_graph = True)
                optimizer_map.step()
                
                y_hat_2, loss_classifier_2 = self.updateLoss(self.classifier_2,
                                                             x2, y, optimizer_classifier_2)
                y_hat_1, loss_classifier_1 = self.updateLoss(self.classifier_1,
                                                             x1, y, optimizer_classifier_1)
                                                             
                prediction = self.a_1 * y_hat_1 + self.a_2 * y_hat_2

                self.cl_1.append(loss_classifier_1)
                self.cl_2.append(loss_classifier_2)

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

                

            
            _, predicted = torch.max(prediction.data, 1)
            self.correct += (predicted == y).item()
            
            if i == 0:
                print("finish 0")
            if (i + 1) % 100 == 0:
                print("step : %d" % (i + 1), end=", ")
                print("correct: %d" % (self.correct))

            if (i + 1) % 1000 == 0:
                self.accuracy = self.correct / 1000
                self.lossFigure_i.append(i)
                self.Accuracy.append(self.accuracy)
                self.correct = 0
                print("Accuracy: ", self.accuracy)



    def Time_2(self):
        "use FESA when i<T1 "
        self.Time_1()
        self.correct = 0
        "load data for T2"
        data2 = self.x_S2[:self.B]
        "use the Network"
        optimizer_classifier_1_FES = torch.optim.Adam(self.classifier_1.parameters(), lr=0.0001)
        optimizer_classifier_2_FES = torch.optim.Adam(self.classifier_2.parameters(), lr=0.0001)


        # loss_3_array = []
        "train the network"
        data_2 = self.x_S2[:self.B]
        label_2 = self.y_S1[:self.B]
        
        self.a_1 = 0.2
        self.a_2 = 0.8
        self.cl_1 = []
        self.cl_2 = []
        b = -0.01
        for (i, x) in enumerate(data_2):
            x = x.flatten().unsqueeze(0).float().to(self.device)
            y = label_2[i].unsqueeze(0).long().to(self.device)
            
           
           
            x2_flatten = x.flatten().float().to(self.device)
            x1_new = (self.fn(x2_flatten)).unsqueeze(0).to(self.device)
            
           
            y_hat_2, loss_classifier_2 = self.updateLoss(self.classifier_2,
                                                         x, y, optimizer_classifier_2_FES)
            
            y_hat_1, loss_classifier_1 = self.updateLoss(self.classifier_1,
                                                         x1_new, y, optimizer_classifier_1_FES)   

          

            prediction = self.a_1 * y_hat_1 + self.a_2 * y_hat_2

            self.cl_1.append(loss_classifier_1)
            self.cl_2.append(loss_classifier_2)

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

            
            _, predicted = torch.max(prediction.data, 1)
            self.correct += (predicted == y).item()
            
            if i == 0:
                print("finish 1")
            if (i + 1) % 100 == 0:
                print("step : %d" % (i + 1), end=", ")
                print("correct: %d" % (self.correct))
            if (i + 1) % 1000 == 0:
                self.accuracy = self.correct / 1000
                self.lossFigure_i.append(i + 50000)
                self.Accuracy.append(self.accuracy)
                self.correct = 0
                print("Accuracy: ", self.accuracy)
                print(self.a_1, self.a_2)
        torch.save(self.Accuracy,'/home/hlian001/fesdl/data/fes_accuracy')
#plot(x_axis=self.lossFigure_i, y_axis=self.Accuracy, title="FES_Figure_Accuracy",
 #            y_label="accuracy", save=True)

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

        return prediction,loss

    def HB_Fit(self, model, X, Y, optimizer, block_split=6):  # hedge backpropagation
        predictions_per_layer = model.forward(X)

        output = predictions_per_layer[-1]
        loss = self.CELoss(output, Y)

        optimizer.zero_grad()

        loss.backward(retain_graph=True)
        optimizer.step()

    
        return loss, output
        



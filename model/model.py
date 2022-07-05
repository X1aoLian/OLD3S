import torch
import torch.nn as nn
import math
import copy
import torch.nn.functional as F

from torch.nn.parameter import Parameter
from cnn import Dynamic_ResNet18
from autoencoder import *
from mlp import MLP

class OLD3S_Deep:
    def __init__(self, data_S1, label_S1, data_S2, label_S2, T1, t, path, lr=0.01, b=0.9, eta = -0.01, s=0.008, m=0.9,
                 spike=9e-5, thre=10000, RecLossFunc = 'KL'):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.autoencoder = AutoEncoder_Deep().to(self.device)
        self.autoencoder_2 = AutoEncoder_Deep().to(self.device)
        self.spike = spike
        self.thre = thre
        self.beta1 = 1
        self.beta2 = 0
        self.correct = 0
        self.accuracy = 0
        self.T1 = T1
        self.t = t
        self.B = self.T1 - self.t
        self.path = path
        self.data_S1 = data_S1
        self.label_S1 = label_S1
        self.data_S2 = data_S2
        self.label_S2 = label_S2
        self.lr = Parameter(torch.tensor(lr), requires_grad=False).to(self.device)
        self.b = Parameter(torch.tensor(b), requires_grad=False).to(self.device)
        self.eta = Parameter(torch.tensor(eta), requires_grad=False).to(self.device)
        self.s = Parameter(torch.tensor(s), requires_grad=False).to(self.device)
        self.m = Parameter(torch.tensor(m), requires_grad=False).to(self.device)
        self.s1 = Parameter(torch.tensor(0.01), requires_grad=False).to(self.device)
        self.num_block = 8
        self.alpha1 = Parameter(torch.Tensor(self.num_block).fill_(1 / self.num_block), requires_grad=False).to(
            self.device)
        self.alpha2 = Parameter(torch.Tensor(self.num_block).fill_(1 / self.num_block), requires_grad=False).to(
            self.device)
        self.RecLossFunc = self.ChoiceOfRecLossFnc(RecLossFunc)

        self.CELoss = nn.CrossEntropyLoss()
        self.KLDivLoss = nn.KLDivLoss()
        self.BCELoss = nn.BCELoss()
        self.SmoothL1Loss = nn.SmoothL1Loss()
        self.Accuracy = []
        self.a_1 = 0.8
        self.a_2 = 0.2
        self.cl_1 = []
        self.cl_2 = []

    def FirstPeriod(self):
        data1 = self.data_S1
        data2 = self.data_S2[self.B:]
        self.net_model1 = Dynamic_ResNet18().to(self.device)
        optimizer_1 = torch.optim.SGD(self.net_model1.parameters(), lr=self.lr)
        optimizer_2 = torch.optim.SGD(self.autoencoder.parameters(), lr=0.03)


        for (i, x) in enumerate(data1):
            self.i = i
            x1 = x.unsqueeze(0).float().to(self.device)

            self.i = i
            y = self.label_S1[i].unsqueeze(0).long().to(self.device)

            if self.i < self.B:
                optimizer_2.zero_grad()
                encoded, decoded = self.autoencoder(x1)
                loss_1, y_hat = self.HB_Fit(self.net_model1, encoded, y, optimizer_1)
                loss_2 = self.RecLossFunc(torch.sigmoid(decoded), x1)
                loss_2.backward()
                optimizer_2.step()

                if self.i < self.thre:
                    self.beta2 = self.beta2 + self.spike
                    self.beta1 = 1 - self.beta2
            else:
                x2 = data2[self.i - self.B].unsqueeze(0).float().to(self.device)
                if i == self.B:
                    self.net_model2 = copy.deepcopy(self.net_model1)
                    torch.save(self.net_model1.state_dict(), './data/'+self.path +'/net_model1.pth')
                    optimizer_1_1 = torch.optim.SGD(self.net_model1.parameters(), lr=self.lr)
                    optimizer_1_2 = torch.optim.SGD(self.net_model2.parameters(), lr=self.lr)
                    #optimizer_2_1 = torch.optim.SGD(self.autoencoder.parameters(), lr=0.02)
                    optimizer_2_2 = torch.optim.SGD(self.autoencoder_2.parameters(), lr=0.08)

                encoded_1, decoded_1 = self.autoencoder(x1)
                encoded_2, decoded_2 = self.autoencoder_2(x2)
                loss_1_1, y_hat_1 = self.HB_Fit(self.net_model1, encoded_1, y, optimizer_1_1)
                loss_1_2, y_hat_2 = self.HB_Fit(self.net_model2, encoded_2, y, optimizer_1_2)
                y_hat = self.a_1 * y_hat_1 + self.a_2 * y_hat_2

                self.cl_1.append(loss_1_1)
                self.cl_2.append(loss_1_2)
                if len(self.cl_1) == 100:
                    self.cl_1.pop(0)
                    self.cl_2.pop(0)
                try:
                    a_cl_1 = math.exp(self.eta * sum(self.cl_1))
                    a_cl_2 = math.exp(self.eta * sum(self.cl_2))
                    self.a_1 = (a_cl_1) / (a_cl_2 + a_cl_1)
                except OverflowError:
                    self.a_1 = float('inf')
                self.a_2 = 1 - self.a_1

                optimizer_2_2.zero_grad()
                loss_2_1 = self.RecLossFunc(torch.sigmoid(x2), decoded_2)
                loss_2_2 = self.RecLossFunc(encoded_2, encoded_1)
                loss_2 = loss_2_1 + loss_2_2
                loss_2.backward(retain_graph=True)
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
                self.Accuracy.append(self.accuracy)
                self.correct = 0
                print("Accuracy: ", self.accuracy)
        torch.save(self.net_model2.state_dict(), './data/'+self.path +'/net_model2.pth')


    def SecondPeriod(self):
        print("use FirstPeriod when i<T1 ")
        self.FirstPeriod()
        self.correct = 0
        data2 = self.data_S2[:self.B]
        net_model1 = self.loadmodel('./data/'+self.path +'/net_model1.pth')
        net_model2 = self.loadmodel('./data/'+self.path +'/net_model2.pth')
        optimizer_3 = torch.optim.SGD(net_model1.parameters(), lr=self.lr)
        optimizer_4 = torch.optim.SGD(net_model2.parameters(), lr=self.lr)
        optimizer_5 = torch.optim.SGD(self.autoencoder_2.parameters(), lr=self.lr)

        self.a_1 = 0.2
        self.a_2 = 0.8
        self.cl_1 = []
        self.cl_2 = []
        for (i, x) in enumerate(data2):
            x = x.unsqueeze(0).float().to(self.device)
            y = self.label_S2[i].unsqueeze(0).long().to(self.device)
            encoded, decoded = self.autoencoder_2(x)
            optimizer_5.zero_grad()
            loss_4, y_hat_2 = self.HB_Fit(net_model2, encoded, y, optimizer_4)
            loss_3, y_hat_1 = self.HB_Fit(net_model1, encoded, y, optimizer_3)
            loss_5 = self.RecLossFunc(torch.sigmoid(x), decoded)
            loss_5.backward()
            optimizer_5.step()
            y_hat = self.a_1 * y_hat_1 + self.a_2 * y_hat_2

            self.cl_1.append(loss_3)
            self.cl_2.append(loss_4)
            if len(self.cl_1) == 100:
                self.cl_1.pop(0)
                self.cl_2.pop(0)
            try:
                a_cl_1 = math.exp(self.eta * sum(self.cl_1))
                a_cl_2 = math.exp(self.eta * sum(self.cl_2))
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
                self.Accuracy.append(self.accuracy)
                self.correct = 0
                print("Accuracy: ", self.accuracy)

        torch.save(self.Accuracy, './data/'+self.path +'/Accuracy')


    def zero_grad(self, model):
        for child in model.children():
            for param in child.parameters():
                if param.grad is not None:
                    param.grad.zero_()
    def ChoiceOfRecLossFnc(self, name):
        if name == 'Smooth':
            return nn.SmoothL1Loss()
        elif name == 'KL':
            return nn.KLDivLoss()
        elif name == 'BCE':
            return nn.BCELoss()
        else:
            print('Enter correct loss function name!')

    def SmoothReconstruction(self, X, decoded, optimizer):
        optimizer.zero_grad()

        loss_2 = self.SmoothL1Loss(torch.sigmoid(X), decoded)
        loss_2.backward()
        optimizer.step()

    def loadmodel(self, path):
        net_model = Dynamic_ResNet18().to(self.device)
        pretrain_dict = torch.load(path)
        model_dict = net_model.state_dict()
        pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
        model_dict.update(pretrain_dict)
        net_model.load_state_dict(model_dict)
        net_model.to(self.device)
        return net_model

    def HB_Fit(self, model, X, Y, optimizer, block_split=6):
        predictions_per_layer = model.forward(X)
        losses_per_layer = []
        for out in predictions_per_layer:
            loss = self.CELoss(out, Y)
            losses_per_layer.append(loss)
        output = torch.empty_like(predictions_per_layer[0])
        for i, out in enumerate(predictions_per_layer):
            output += self.alpha1[i] * out
        for i in range(self.num_block):
            if i < block_split:
                if i == 0:
                    alpha_sum_1 = self.alpha1[i]
                else:
                    alpha_sum_1 += self.alpha1[i]
            else:
                if i == block_split:
                    alpha_sum_2 = self.alpha1[i]
                else:
                    alpha_sum_2 += self.alpha1[i]
        Loss_sum = torch.zeros_like(losses_per_layer[0])
        for i, loss in enumerate(losses_per_layer):
            if i < block_split:
                loss_ = (self.alpha1[i] / alpha_sum_1) * self.beta1 * loss
            else:
                loss_ = (self.alpha1[i] / alpha_sum_2) * self.beta2 * loss
            Loss_sum += loss_
        optimizer.zero_grad()
        Loss_sum.backward(retain_graph=True)
        optimizer.step()
        for i in range(len(losses_per_layer)):
            self.alpha1[i] *= torch.pow(self.b, losses_per_layer[i])
            self.alpha1[i] = torch.max(self.alpha1[i], self.s / self.num_block)
            self.alpha1[i] = torch.min(self.alpha1[i], self.m)
        z_t = torch.sum(self.alpha1)
        self.alpha1 = Parameter(self.alpha1 / z_t, requires_grad=False).to(self.device)
        return Loss_sum, output


class OLD3S_Shallow:
    def __init__(self, data_S1, label_S1, data_S2, label_S2, T1, t, dimension1, dimension2, path, lr = 0.001, b=0.9,
                 eta = -0.001, s=0.008, m=0.99, RecLossFunc = 'BCE'):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.correct = 0
        self.accuracy = 0
        self.lr = lr
        self.T1 = T1
        self.t = t
        self.B = self.T1 - self.t
        self.path = path
        self.x_S1 = data_S1
        self.y_S1 = label_S1
        self.x_S2 = data_S2
        self.y_S2 = label_S2
        self.dimension1 = dimension1
        self.dimension2 = dimension2
        self.b = Parameter(torch.tensor(b), requires_grad=False).to(self.device)
        self.eta = Parameter(torch.tensor(eta), requires_grad=False).to(self.device)
        self.s = Parameter(torch.tensor(s), requires_grad=False).to(self.device)
        self.m = Parameter(torch.tensor(m), requires_grad=False).to(self.device)
        self.CELoss = nn.CrossEntropyLoss()
        self.BCELoss = nn.BCELoss()
        self.SmoothL1Loss = nn.SmoothL1Loss()
        self.MSELoss = nn.MSELoss()
        self.RecLossFunc = self.ChoiceOfRecLossFnc(RecLossFunc)
        self.Accuracy = []
        self.a_1 = 0.5
        self.a_2 = 0.5
        self.cl_1 = []
        self.cl_2 = []
        self.alpha = Parameter(torch.Tensor(5).fill_(1 / 5), requires_grad=False).to(
            self.device)
        self.autoencoder_1 = AutoEncoder_Shallow(self.dimension1, 1024).to(self.device)
        self.autoencoder_2 = AutoEncoder_Shallow(self.dimension2, 1024).to(self.device)

    def FirstPeriod(self):
        classifier_1 = MLP(1024,2).to(self.device)
        optimizer_classifier_1 = torch.optim.Adam(classifier_1.parameters(), self.lr)

        optimizer_autoencoder_1 = torch.optim.Adam(self.autoencoder_1.parameters(), self.lr)
        # eta = -8 * math.sqrt(1 / math.log(self.t))

        for (i, x) in enumerate(self.x_S1):

            self.i = i
            x1 = x.unsqueeze(0).float().to(self.device)
            y = self.y_S1[i].unsqueeze(0).long().to(self.device)
            if self.y_S1[i] == 0:
                y1 = torch.Tensor([1, 0]).reshape(1, 2).float().to(self.device)
            else:
                y1 = torch.Tensor([0, 1]).reshape(1, 2).float().to(self.device)
            if self.i < self.B:  # Before evolve
                encoded_1, decoded_1 = self.autoencoder_1(x1)
                optimizer_autoencoder_1.zero_grad()
                y_hat, loss_classifier_1 = self.HB_Fit(classifier_1,
                                                       encoded_1, y1, optimizer_classifier_1)

                loss_autoencoder_1 = self.RecLossFunc(torch.sigmoid(decoded_1), x1)
                loss_autoencoder_1.backward()
                optimizer_autoencoder_1.step()

            else:
                x2 = self.x_S2[self.i].unsqueeze(0).float().to(self.device)
                if i == self.B:
                    classifier_2 = copy.deepcopy(classifier_1)

                    torch.save(classifier_1.state_dict(),
                               './data/'+self.path +'/net_model1.pth')
                    optimizer_classifier_2 = torch.optim.Adam(classifier_2.parameters(), self.lr)
                    optimizer_autoencoder_2 = torch.optim.Adam(self.autoencoder_2.parameters(), self.lr)

                encoded_2, decoded_2 = self.autoencoder_2(x2)
                encoded_1, decoded_1 = self.autoencoder_1(x1)

                y_hat_2, loss_classifier_2 = self.HB_Fit(classifier_2,
                                                         encoded_2, y1, optimizer_classifier_2)
                y_hat_1, loss_classifier_1 = self.HB_Fit(classifier_1,
                                                         encoded_1, y1, optimizer_classifier_1)

                y_hat = self.a_1 * y_hat_1 + self.a_2 * y_hat_2

                self.cl_1.append(loss_classifier_1)
                self.cl_2.append(loss_classifier_2)
                if len(self.cl_1) == 50:
                    self.cl_1.pop(0)
                    self.cl_2.pop(0)

                try:
                    a_cl_1 = math.exp(self.eta * sum(self.cl_1))
                    a_cl_2 = math.exp(self.eta * sum(self.cl_2))
                    self.a_1 = (a_cl_1) / (a_cl_2 + a_cl_1)

                except OverflowError:
                    self.a_1 = float('inf')

                self.a_2 = 1 - self.a_1

                optimizer_autoencoder_2.zero_grad()
                loss_2_0 = self.RecLossFun(torch.sigmoid(decoded_2), x2)
                loss_2_1 = self.RecLossFun(encoded_2, encoded_1)
                loss_autoencoder_2 = loss_2_0 + loss_2_1
                loss_autoencoder_2.backward()
                optimizer_autoencoder_2.step()



            _, predicted = torch.max(y_hat.data, 1)

            self.correct += (predicted == y).item()
            if i == 0:
                print("finish 0")
            if (i + 1) % 100 == 0:
                print("step : %d" % (i + 1), end=", ")
                print("correct: %d" % (self.correct))

            if (i + 1) % 500 == 0:
                self.accuracy = self.correct / 500
                self.Accuracy.append(self.accuracy)
                self.correct = 0
                print("Accuracy: ", self.accuracy)

        torch.save(classifier_2.state_dict(), './data/'+self.path +'/net_model2.pth')

    def SecondPeriod(self):
        print('use FESA when i<T1')
        self.FirstPeriod()
        self.correct = 0
        net_model1 = self.loadmodel('./data/' + self.path + '/net_model1.pth')
        net_model2 = self.loadmodel('./data/' + self.path + '/net_model2.pth')

        optimizer_classifier_1_FES = torch.optim.Adam(net_model1.parameters(), self.lr)
        optimizer_classifier_2_FES = torch.optim.Adam(net_model2.parameters(), self.lr)
        optimizer_autoencoder_2_FES = torch.optim.Adam(self.autoencoder_2.parameters(), self.lr)
        data_2 = self.x_S2[:self.B]
        label_2 = self.y_S1[:self.B]

        self.a_1 = 0.2
        self.a_2 = 0.8
        self.cl_1 = []
        self.cl_2 = []

        # eta = -8 * math.sqrt(1 / math.log(self.B))
        for (i, x) in enumerate(data_2):
            x = x.unsqueeze(0).float().to(self.device)
            self.i = i + self.T1
            y = label_2[i].long().to(self.device)
            if label_2[i] == 0:
                y1 = torch.Tensor([1, 0]).unsqueeze(0).float().to(self.device)
            else:
                y1 = torch.Tensor([0, 1]).unsqueeze(0).float().to(self.device)


            encoded_2, decoded_2 = self.autoencoder_2(x)
            optimizer_autoencoder_2_FES.zero_grad()
            y_hat_2, loss_classifier_2 = self.HB_Fit(net_model2,
                                                     encoded_2, y1, optimizer_classifier_2_FES)
            y_hat_1, loss_classifier_1 = self.HB_Fit(net_model1,
                                                     encoded_2, y1, optimizer_classifier_1_FES)

            loss_autoencoder_2 = self.RecLossFun(torch.sigmoid(decoded_2), x)
            loss_autoencoder_2.backward()
            optimizer_autoencoder_2_FES.step()
            y_hat = self.a_1 * y_hat_1 + self.a_2 * y_hat_2
            self.cl_1.append(loss_classifier_1)
            self.cl_2.append(loss_classifier_2)

            if len(self.cl_1) == 50:
                self.cl_1.pop(0)
                self.cl_2.pop(0)

            try:
                a_cl_1 = math.exp(self.eta * sum(self.cl_1))
                a_cl_2 = math.exp(self.eta * sum(self.cl_2))
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
            if (i + 1) % 500 == 0:
                self.accuracy = self.correct / 500
                self.Accuracy.append(self.accuracy)
                self.correct = 0
                print("Accuracy: ", self.accuracy)

        torch.save(self.Accuracy, './data/'+self.path +'/Accuracy')


    def zero_grad(self, model):
        for child in model.children():
            for param in child.parameters():
                if param.grad is not None:
                    # param.grad.detach_()
                    param.grad.zero_()  # data.fill_(0)

    def loadmodel(self, path):
        net_model = MLP(1024,2).to(self.device)
        pretrain_dict = torch.load(path)
        model_dict = net_model.state_dict()
        pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
        model_dict.update(pretrain_dict)
        net_model.load_state_dict(model_dict)
        net_model.to(self.device)
        return net_model

    def ChoiceOfRecLossFnc(self, name):
        if name == 'Smooth':
            return nn.SmoothL1Loss()
        elif name == 'KL':
            return nn.KLDivLoss()
        elif name == 'BCE':
            return nn.BCELoss()
        else:
            print('Enter correct loss function name!')

    def HB_Fit(self, model, X, Y, optimizer):  # hedge backpropagation
        predictions_per_layer = model.forward(X)

        losses_per_layer = []

        for out in predictions_per_layer:
            loss = self.BCELoss(out, Y)

            losses_per_layer.append(loss)

        output = torch.empty_like(predictions_per_layer[0])
        for i, out in enumerate(predictions_per_layer):
            output += self.alpha[i] * out

        for i in range(5):  # First 6 are shallow and last 2 are deep

            if i == 0:
                alpha_sum_1 = self.alpha[i]
            else:
                alpha_sum_1 += self.alpha[i]

        Loss_sum = torch.zeros_like(losses_per_layer[0])

        for i, loss in enumerate(losses_per_layer):
            loss_ = (self.alpha[i] / alpha_sum_1) * loss
            Loss_sum += loss_
        optimizer.zero_grad()

        Loss_sum.backward(retain_graph=True)
        optimizer.step()

        for i in range(len(losses_per_layer)):
            self.alpha[i] *= torch.pow(self.b, losses_per_layer[i])
            self.alpha[i] = torch.max(self.alpha[i], self.s / 5)
            self.alpha[i] = torch.min(self.alpha[i], self.m)  # exploration-exploitation

        z_t = torch.sum(self.alpha)
        self.alpha = Parameter(self.alpha / z_t, requires_grad=False).to(self.device)

        return output, Loss_sum

class OLD3S_Reuter:
    def __init__(self, data_S1, label_S1, data_S2, label_S2, T1, t, dimension1, dimension2,  path,  lr=0.01, b=0.9,
                 eta=-0.001, s=0.008, m=0.99, RecLossFunc='BCE'):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.correct = 0
        self.accuracy = 0
        self.lr = lr
        self.T1 = T1
        self.t = t
        self.B = self.T1 - self.t
        self.path = path
        self.x_S1 = data_S1
        self.y_S1 = label_S1
        self.x_S2 = data_S2
        self.y_S2 = label_S2
        self.dimension1 = dimension1
        self.dimension2 = dimension2
        self.b = Parameter(torch.tensor(b), requires_grad=False).to(self.device)
        self.eta = Parameter(torch.tensor(eta), requires_grad=False).to(self.device)
        self.s = Parameter(torch.tensor(s), requires_grad=False).to(self.device)
        self.m = Parameter(torch.tensor(m), requires_grad=False).to(self.device)
        self.CELoss = nn.CrossEntropyLoss()
        self.BCELoss = nn.BCELoss()
        self.SmoothL1Loss = nn.SmoothL1Loss()
        self.MSELoss = nn.MSELoss()
        self.RecLossFunc = self.ChoiceOfRecLossFnc(RecLossFunc)
        self.Accuracy = []
        self.a_1 = 0.5
        self.a_2 = 0.5
        self.cl_1 = []
        self.cl_2 = []
        self.alpha = Parameter(torch.Tensor(5).fill_(1 / 5), requires_grad=False).to(
            self.device)
        self.autoencoder_1 = AutoEncoder_Shallow(self.dimension1, 1024).to(self.device)
        self.autoencoder_2 = AutoEncoder_Shallow(self.dimension2, 1024).to(self.device)

    def FirstPeriod(self):
        classifier_1 = MLP(1024,6).to(self.device)
        optimizer_classifier_1 = torch.optim.SGD(classifier_1.parameters(), self.lr)

        optimizer_autoencoder_1 = torch.optim.SGD(self.autoencoder_1.parameters(), self.lr)
        # eta = -8 * math.sqrt(1 / math.log(self.t))

        for (i, x) in enumerate(self.x_S1):

            self.i = i
            x1 = x.unsqueeze(0).float().to(self.device)
            y1 = self.y_S1[i].long().to(self.device)

            if self.i < self.B:  # Before evolve
                encoded_1, decoded_1 = self.autoencoder_1(x1)
                optimizer_autoencoder_1.zero_grad()
                y_hat, loss_classifier_1 = self.HB_Fit(classifier_1,
                                                       encoded_1, y1, optimizer_classifier_1)

                loss_autoencoder_1 = self.RecLossFunc(torch.sigmoid(decoded_1), x1)
                loss_autoencoder_1.backward()
                optimizer_autoencoder_1.step()

            else:
                x2 = self.x_S2[self.i].unsqueeze(0).float().to(self.device)
                if i == self.B:
                    classifier_2 = copy.deepcopy(classifier_1)

                    torch.save(classifier_1.state_dict(),
                               './data/' + self.path + '/net_model1.pth')
                    optimizer_classifier_2 = torch.optim.SGD(classifier_2.parameters(), self.lr)
                    optimizer_autoencoder_2 = torch.optim.SGD(self.autoencoder_2.parameters(), 0.9)

                encoded_2, decoded_2 = self.autoencoder_2(x2)
                encoded_1, decoded_1 = self.autoencoder_1(x1)

                y_hat_2, loss_classifier_2 = self.HB_Fit(classifier_2,
                                                         encoded_2, y1, optimizer_classifier_2)
                y_hat_1, loss_classifier_1 = self.HB_Fit(classifier_1,
                                                         encoded_1, y1, optimizer_classifier_1)

                y_hat = self.a_1 * y_hat_1 + self.a_2 * y_hat_2

                self.cl_1.append(loss_classifier_1)
                self.cl_2.append(loss_classifier_2)
                if len(self.cl_1) == 50:
                    self.cl_1.pop(0)
                    self.cl_2.pop(0)

                try:
                    a_cl_1 = math.exp(self.eta * sum(self.cl_1))
                    a_cl_2 = math.exp(self.eta * sum(self.cl_2))
                    self.a_1 = (a_cl_1) / (a_cl_2 + a_cl_1)

                except OverflowError:
                    self.a_1 = float('inf')

                self.a_2 = 1 - self.a_1

                optimizer_autoencoder_2.zero_grad()
                loss_2_0 = self.BCELoss(torch.sigmoid(decoded_2), x2)
                loss_2_1 = self.SmoothL1Loss(encoded_2, encoded_1)
                loss_autoencoder_2 = loss_2_0 + loss_2_1
                loss_autoencoder_2.backward()
                optimizer_autoencoder_2.step()

            _, predicted = torch.max(y_hat.data, 1)

            self.correct += (predicted == y1).item()
            if i == 0:
                print("finish 0")
            if (i + 1) % 100 == 0:
                print("step : %d" % (i + 1), end=", ")
                print("correct: %d" % (self.correct))

            if (i + 1) % 500 == 0:
                self.accuracy = self.correct / 500
                self.Accuracy.append(self.accuracy)
                self.correct = 0
                print("Accuracy: ", self.accuracy)

        torch.save(classifier_2.state_dict(), './data/' + self.path + '/net_model2.pth')

    def SecondPeriod(self):
        print('use FESA when i<T1')
        self.FirstPeriod()
        self.correct = 0
        net_model1 = self.loadmodel('./data/' + self.path + '/net_model1.pth')
        net_model2 = self.loadmodel('./data/' + self.path + '/net_model2.pth')

        optimizer_classifier_1_FES = torch.optim.SGD(net_model1.parameters(), self.lr)
        optimizer_classifier_2_FES = torch.optim.SGD(net_model2.parameters(), self.lr)
        optimizer_autoencoder_2_FES = torch.optim.SGD(self.autoencoder_2.parameters(), self.lr)
        data_2 = self.x_S2[:self.B]
        label_2 = self.y_S1[:self.B]

        self.a_1 = 0.2
        self.a_2 = 0.8
        self.cl_1 = []
        self.cl_2 = []

        # eta = -8 * math.sqrt(1 / math.log(self.B))
        for (i, x) in enumerate(data_2):
            x = x.unsqueeze(0).float().to(self.device)
            self.i = i + self.T1
            y1 = label_2[i].long().to(self.device)


            encoded_2, decoded_2 = self.autoencoder_2(x)
            optimizer_autoencoder_2_FES.zero_grad()
            y_hat_2, loss_classifier_2 = self.HB_Fit(net_model2,
                                                     encoded_2, y1, optimizer_classifier_2_FES)
            y_hat_1, loss_classifier_1 = self.HB_Fit(net_model1,
                                                     encoded_2, y1, optimizer_classifier_1_FES)

            loss_autoencoder_2 = self.BCELoss(torch.sigmoid(decoded_2), x)
            loss_autoencoder_2.backward()
            optimizer_autoencoder_2_FES.step()
            y_hat = self.a_1 * y_hat_1 + self.a_2 * y_hat_2
            self.cl_1.append(loss_classifier_1)
            self.cl_2.append(loss_classifier_2)

            if len(self.cl_1) == 50:
                self.cl_1.pop(0)
                self.cl_2.pop(0)

            try:
                a_cl_1 = math.exp(self.eta * sum(self.cl_1))
                a_cl_2 = math.exp(self.eta * sum(self.cl_2))
                self.a_1 = (a_cl_1) / (a_cl_2 + a_cl_1)
            except OverflowError:
                self.a_1 = float('inf')

            self.a_2 = 1 - self.a_1

            _, predicted = torch.max(y_hat.data, 1)
            self.correct += (predicted == y1).item()
            if i == 0:
                print("finish 1")
            if (i + 1) % 100 == 0:
                print("step : %d" % (i + 1), end=", ")
                print("correct: %d" % (self.correct))
            if (i + 1) % 500 == 0:
                self.accuracy = self.correct / 500
                self.Accuracy.append(self.accuracy)
                self.correct = 0
                print("Accuracy: ", self.accuracy)

        torch.save(self.Accuracy, './data/' + self.path + '/Accuracy')

    def zero_grad(self, model):
        for child in model.children():
            for param in child.parameters():
                if param.grad is not None:
                    # param.grad.detach_()
                    param.grad.zero_()  # data.fill_(0)

    def loadmodel(self, path):
        net_model = MLP(1024,6).to(self.device)
        pretrain_dict = torch.load(path)
        model_dict = net_model.state_dict()
        pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
        model_dict.update(pretrain_dict)
        net_model.load_state_dict(model_dict)
        net_model.to(self.device)
        return net_model

    def ChoiceOfRecLossFnc(self, name):
        if name == 'Smooth':
            return nn.SmoothL1Loss()
        elif name == 'KL':
            return nn.KLDivLoss()
        elif name == 'BCE':
            return nn.BCELoss()
        else:
            print('Enter correct loss function name!')

    def HB_Fit(self, model, X, Y, optimizer):  # hedge backpropagation
        predictions_per_layer = model.forward(X)

        losses_per_layer = []

        for out in predictions_per_layer:
            loss = self.CELoss(out, Y)

            losses_per_layer.append(loss)

        output = torch.empty_like(predictions_per_layer[0])
        for i, out in enumerate(predictions_per_layer):
            output += self.alpha[i] * out

        for i in range(5):  # First 6 are shallow and last 2 are deep

            if i == 0:
                alpha_sum_1 = self.alpha[i]
            else:
                alpha_sum_1 += self.alpha[i]

        Loss_sum = torch.zeros_like(losses_per_layer[0])

        for i, loss in enumerate(losses_per_layer):
            loss_ = (self.alpha[i] / alpha_sum_1) * loss
            Loss_sum += loss_
        optimizer.zero_grad()

        Loss_sum.backward(retain_graph=True)
        optimizer.step()

        for i in range(len(losses_per_layer)):
            self.alpha[i] *= torch.pow(self.b, losses_per_layer[i])
            self.alpha[i] = torch.max(self.alpha[i], self.s / 5)
            self.alpha[i] = torch.min(self.alpha[i], self.m)  # exploration-exploitation

        z_t = torch.sum(self.alpha)
        self.alpha = Parameter(self.alpha / z_t, requires_grad=False).to(self.device)

        return output, Loss_sum
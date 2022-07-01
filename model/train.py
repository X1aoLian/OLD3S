import random
import torch
import numpy as np
import torchvision
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision import models
from torch import nn
from sklearn.utils import shuffle
from Cifar import OLD3S
from deepFesl_nohedge import Resnet18_Cifar
from Linear_cifar import SimpleLinear_Cifar
from FES import FES
from deepFesl_linear import deepFESL_linear
from vae import VAE
from loaddatasets import *

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    parser = argparse.ArgumentParser(description="Options")
    parser.add_argument('-DataName', action='store', dest='DataName', default='cifar')
    parser.add_argument('-FromLanguage', action='store', dest='FromLanguage', default='EN')
    parser.add_argument('-ToLanguage', action='store', dest='ToLanguage', default='FR')
    parser.add_argument('-beta', action='store', dest='beta', default=0.9)
    parser.add_argument('-eta', action='store', dest='beta', default=-0.01)
    parser.add_argument('-learningrate', action='store', dest='learningrate', default=0.01)

    args = parser.parse_args()
    learner = OLDS(args)
    learner.train()


class OLDS:
    def __init__(self, args):
        '''
            Data is stored as list of dictionaries.
            Label is stored as list of scalars.
        '''
        self.datasetname = args.DataName
        self.FromLan = args.FromLanguage
        self.ToLan = args.ToLanguage
        self.beta = args.beta
        self.eta = args.eta
        self.learningrate = args.learningrate

    def train(self):
        if self.datasetname == 'cifar':
            print('trainning starts')
            x_S1, y_S1, x_S2, y_S2 = loadcifar()
            train = OLD3S(x_S1, y_S1, x_S2, y_S2, 50000, 5000,self.beta,self.learningrate)
            train.SecondPeriod()
        '''else:
            if self.FromLan =='EN':
                self.samplesize = 18758
                self.overlap = 2758
                self.dimension1 = 21531
                self.dimension1_pca = 2000
                if self.ToLan == 'FR':
                    self.dimension2 = 24893
                    self.dimension2_pca = 2500
                elif self.ToLan == 'IT':
                    self.dimension2 = 15506
                    self.dimension2_pca = 1500
                else:
                    self.dimension2 = 11547
                    self.dimension2_pca = 1000
            else:
                self.samplesize = 26648
                self.overlap = 3648
                self.dimension1 = 24893
                self.dimension1_pca = 2500
                if self.ToLan == 'IT':
                    self.dimension2 = 15503
                    self.dimension2_pca = 1500
                else:
                    self.dimension2 = 11547
                    self.dimension2_pca = 1000

            x_S1, x_S2, y_S1, y_S2 = loadreuter(self.FromLan,self.ToLan,
                                                self.samplesize, self.dimension1, self.dimension2)
            train = Reuter(x_S1, y_S1, x_S2, y_S2, self.samplesize, self.samplesize,
                           self.dimension1_pca,self.dimension2_pca,self.FromLan,self.ToLan,self.beta,self.learningrate)
            train.T_2()'''



if __name__ == '__main__':
    setup_seed(30)
    main()






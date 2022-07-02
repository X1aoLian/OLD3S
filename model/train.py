import random
import numpy as np
import argparse
from model import *
from loaddatasets import *

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    parser = argparse.ArgumentParser(description="Options")
    parser.add_argument('-DataName', action='store', dest='DataName', default='adult')
    parser.add_argument('-FromLanguage', action='store', dest='FromLanguage', default='EN')
    parser.add_argument('-ToLanguage', action='store', dest='ToLanguage', default='FR')
    parser.add_argument('-beta', action='store', dest='beta', default=0.9)
    parser.add_argument('-eta', action='store', dest='eta', default=-0.01)
    parser.add_argument('-learningrate', action='store', dest='learningrate', default=0.01)
    parser.add_argument('-RecLossFunc', action='store', dest='RecLossFunc', default='Smooth')

    args = parser.parse_args()
    learner = OLD3S(args)
    learner.train()


class OLD3S:
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
        self.RecLossFunc = args.RecLossFunc

    def train(self):
        if self.datasetname == 'cifar':
            print('cifar trainning starts')
            x_S1, y_S1, x_S2, y_S2 = loadcifar()
            train = OLD3S_Deep(x_S1, y_S1, x_S2, y_S2, 50000, 5000,'parameter_cifar')
            train.SecondPeriod()
        elif self.datasetname == 'svhn':
            print('svhn trainning starts')
            x_S1, y_S1, x_S2, y_S2 = loadsvhn()
            train = OLD3S_Deep(x_S1, y_S1, x_S2, y_S2, 73257, 7257,'parameter_svhn')
            train.SecondPeriod()
        elif self.datasetname == 'magic':
            print('magic trainning starts')
            x_S1, y_S1, x_S2, y_S2 = loadmagic()
            train = OLD3S_Shallow(x_S1, y_S1, x_S2, y_S2, 19019, 1919, 10, 30, 'parameter_magic')
            train.SecondPeriod()
        elif self.datasetname == 'adult':
            x_S1, y_S1, x_S2, y_S2 = loadadult()
            train = OLD3S_Shallow(x_S1, y_S1, x_S2, y_S2, 32559, 3559, 14, 30, 'parameter_adult')
            train.SecondPeriod()
        else:
            print('Choose a correct dataset name please')
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






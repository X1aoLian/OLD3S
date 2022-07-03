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
    parser.add_argument('-DataName', action='store', dest='DataName', default='frit')
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
            print('adult trainning starts')
            x_S1, y_S1, x_S2, y_S2 = loadadult()
            train = OLD3S_Shallow(x_S1, y_S1, x_S2, y_S2, 32559, 3559, 14, 30, 'parameter_adult')
            train.SecondPeriod()
        elif self.datasetname == 'enfr':
            print('reuter-en-fr trainning starts')
            x_S1, y_S1, x_S2, y_S2 = loadreuter('EN_FR')
            train = OLD3S_Reuter(x_S1, y_S1, x_S2, y_S2, 18758, 2758,2000, 2500, 'parameter_enfr')
            train.SecondPeriod()
        elif self.datasetname == 'enit':
            print('reuter-en-it trainning starts')
            x_S1, y_S1, x_S2, y_S2 = loadreuter('EN_IT')
            train = OLD3S_Reuter(x_S1, y_S1, x_S2, y_S2, 18758, 2758, 2000, 1500, 'parameter_enit')
            train.SecondPeriod()
        elif self.datasetname == 'ensp':
            print('reuter-en-sp trainning starts')
            x_S1, y_S1, x_S2, y_S2 = loadreuter('EN_SP')
            train = OLD3S_Reuter(x_S1, y_S1, x_S2, y_S2, 18758, 2758, 2000, 1000, 'parameter_ensp')
            train.SecondPeriod()
        elif self.datasetname == 'frit':
            print('reuter-fr-it trainning starts')
            x_S1, y_S1, x_S2, y_S2 = loadreuter('FR_IT')
            train = OLD3S_Reuter(x_S1, y_S1, x_S2, y_S2, 26648, 3648, 2500, 1500, 'parameter_frit')
            train.SecondPeriod()
        elif self.datasetname == 'frsp':
            print('reuter-fr-sp trainning starts')
            x_S1, y_S1, x_S2, y_S2 = loadreuter('FR_SP')
            train = OLD3S_Reuter(x_S1, y_S1, x_S2, y_S2, 26648, 3648, 2500, 1000, 'parameter_frsp')
            train.SecondPeriod()
        else:
            print('Choose a correct dataset name please')

if __name__ == '__main__':
    setup_seed(30)
    main()






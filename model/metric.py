import numpy as np
import torch
from matplotlib import pyplot as plt
from numpy import *
from scipy.interpolate import make_interp_spline
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def plot_reuter(y_axi_1, x, path, a, b):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.plot(range(20))
    ax.axvspan(a, b, alpha=0.5, color='#86C6F4')
    plt.grid()
    x_smooth = np.linspace(x.min(), x.max(), 25)
    y_smooth_1 = make_interp_spline(x, y_axi_1)(x_smooth)
    ACR(y_smooth_1)
    STD(25, y_axi_1, y_smooth_1)
    plt.plot(x_smooth, y_smooth_1, color='#7E2F8E', marker='d')
    ax.set_xlim(250, a + b)
    plt.tick_params(labelsize=20)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    plt.xlabel('# of instances', fontsize=30)
    plt.ylabel('OCA', fontsize=30)
    plt.tight_layout()
    plt.savefig(path)


def ACR(accuracy):
    f_star = max(accuracy)
    acr = mean([f_star - i for i in accuracy])
    print(acr)


def STD(filternumber, elements,smoothlist):
    gap = len(elements)//filternumber
    std = 0
    for i in range(filternumber):
        for j in range(int(gap)):
            std += np.abs(smoothlist[i] - elements[i*gap: (i+1)*gap][j])
    print(std/len(elements))


x = np.array([i for i in range(1000, 50000 + 45000 + 1, 1000)])
path = 'D:\pycharmproject\OLD3S\model\data\CIFAR-Hedge.png'
y_axi_1 = np.array(torch.load('./data/parameter_cifar/parameter_cifarAccuracy')).tolist()

plot_reuter(y_axi_1, x, path, 45000, 50000)
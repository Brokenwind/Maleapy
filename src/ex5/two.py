import sys
sys.path.append('..')
import numpy as np
import scipy.optimize as op
import matplotlib
from matplotlib import pyplot as plt
from leacurve import *

if __name__ == '__main__':
    fig, ax = plt.subplots()
    x = np.loadtxt('x.txt')
    y = np.loadtxt('y.txt')
    xval = np.loadtxt('xval.txt')
    yval = np.loadtxt('yval.txt')
    m = y.size
    errtrain,errval =  leacurve(x,y,xval,yval,0.0)
    xtick = range(1,m+1)
    ax.plot(xtick,errtrain,c='r',label='Train')
    ax.plot(xtick,errval,c='b',label='Cross validation')
    ax.legend(loc='best')
    plt.show()
    

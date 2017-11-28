import sys
sys.path.append('..')
import numpy as np
import scipy.optimize as op
import matplotlib
from matplotlib import pyplot as plt

def plotScatter(ax,x,y):
    pos = x[y == 0]
    neg = x[y == 1]
    ax.scatter(pos[:,0],pos[:,1],c='r',marker='+',label='Positive')
    ax.scatter(neg[:,0],neg[:,1],c='g',marker='o',label='Negative')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.legend(loc='best')


if __name__ == '__main__':
    path = './data/'
    fig, ax = plt.subplots()
    x1 = np.loadtxt(path+'x1.txt')
    y1 = np.loadtxt(path+'y1.txt')
    m = x1.size
    ex1 = x1.reshape((m,1))
    ex1 = np.hstack((np.ones((m,1)),ex1))
    theta = np.ones(2)
    plotScatter(ax,x1,y1)    
    plt.show()
    

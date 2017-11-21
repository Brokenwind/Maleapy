import numpy as np
import scipy.optimize as op
import matplotlib
from matplotlib import pyplot as plt
from optimlog import *

def plotScatter(ax,data):
    pos = data[data[:,2] == 1]
    neg = data[data[:,2] == 0]
    ax.scatter(pos[:,0],pos[:,1],c='r',marker='+',label='Admitted')
    ax.scatter(neg[:,0],neg[:,1],c='g',marker='o',label='Not admitted')
    ax.set_xlabel('Exame 1 score')
    ax.set_ylabel('Exame 2 score')
    ax.legend(loc='best')

def plotBoundary(ax,theta,x):
    xmax = np.max(x[:,1])
    xmin = np.min(x[:,1])
    score1 = np.arange(xmin,xmax,0.1)
    score2 = (-1.0/theta[2])*(theta[0]+theta[1]*score1)
    ax.plot(score1,score2,label='Boundary')
    ax.legend(loc='best')

if __name__ == '__main__':
    fig, ax = plt.subplots()
    data = np.loadtxt('ex2data1.txt',delimiter=',')
    x = np.delete(data,-1,axis=1)
    x = np.hstack((np.ones((x.shape[0],1)),x))
    y = data[:,-1]
    m,n = x.shape
    plotScatter(ax,data)

    theta = np.zeros(n)
    """
    alpha = 0.043
    iters = 1000000
    gradientSolve(theta,x,y,alpha,iters)
    """
    res = optimSolve(theta,x,y)
    #print predict([45,85],theta)
    plotBoundary(ax,res,x)
    #print costFunction(x,y,theta)
    plt.show()

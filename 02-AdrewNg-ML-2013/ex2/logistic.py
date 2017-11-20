#!/usr/bin/python

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def sigmoid(z):
    """sigmoid(z)
    z: It can be a array, a narray, a vector and a matrix
    """
    return 1.0 / (1.0 + np.exp(-z))

"""
def costFunction(x,y,theta):
    data = np.hstack((x,y))
    data = np.array(data.tolist())
    return costFunction(data,theta)
"""

def costFunction(x,y,theta):
    m = x.shape[0]
    y1 = np.array(sigmoid(x*theta))
    #print y1
    y0 = np.array(y)
    cost = (-1.0/m)*np.sum(y0 * np.log(y1) + (1.0 - y0) * np.log(1.0 - y1))
    return cost

def decent(x,y,alpha,theta,iters):
    m = x.shape[0]
    while iters > 0:
        h = np.array(sigmoid(x*theta))
        error = h - y
        deri = (1.0/m)*(x.T * error)
        theta = theta - alpha*deri
        iters -= 1
    return theta

def plotScatter(ax,data):
    pos = data[data[:,2] == 1]
    neg = data[data[:,2] == 0]
    ax.scatter(pos[:,0],pos[:,1],c='r',marker='+',label='Admitted')
    ax.scatter(neg[:,0],neg[:,1],c='g',marker='o',label='Not admitted')
    ax.set_xlabel('Exame 1 score')
    ax.set_ylabel('Exame 2 score')
    ax.legend(loc='best')

def plotBoundary(ax,x,theta):
    xmax = np.max(x[:,1])
    xmin = np.min(x[:,1])
    theta = np.array(theta.T)[0]
    score1 = np.arange(xmin,xmax,0.1)
    score2 = (-1.0/theta[2])*(theta[0]+theta[1]*score1)
    ax.plot(score1,score2,label='Boundary')
    ax.legend(loc='best')

def predict(x,theta):
    x = np.hstack(([1],x))
    x = np.mat(x)
    print ("the probability is %.3f" % (sigmoid(x*theta)))
    h = np.sum(sigmoid(x*theta))
    if h > 0.5:
        return 1.0
    else:
        return 0.0
    
if __name__ == '__main__':
    alpha = 0.03
    theta = np.mat([0,0,0]).T
    iters = 50000
    fig, ax = plt.subplots()
    data = np.loadtxt('ex2data1.txt',delimiter=',')
    plotScatter(ax,data)
    x = np.mat(np.delete(data,-1,axis=1))
    x = np.hstack((np.ones((x.shape[0],1)),x))
    y = np.mat(data[:,-1]).T
    theta = decent(x,y,alpha,theta,iters)    
    print theta
    print predict([45,85],theta)
    plotBoundary(ax,x,theta)
    #print costFunction(x,y,theta)
    plt.show()

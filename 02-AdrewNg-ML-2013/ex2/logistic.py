#!/usr/bin/python

import numpy as np
import scipy.optimize as op
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def sigmoid(z):
    """sigmoid(z)
    z: It can be a array, a narray, a vector and a matrix
    """
    return 1.0 / (1.0 + np.exp(-z))

def costFunction(theta):
    y1 = np.array(sigmoid(x*theta))
    y0 = np.array(y)
    cost = np.sum(y0 * np.log(y1) + (1.0 - y0) * np.log(1.0 - y1))/m
    return cost

def costDer(theta):
    """
    compute the derivative of cost funtion
    """
    h = np.array(sigmoid(x*theta))
    error = h - y
    deri = (x.T * error)/m
    return deri

def decent(alpha,theta,iters):
    """
    Use gradient decent method to compute parameters
    """
    while iters > 0:
        deri = costDer(theta)
        theta = theta - alpha*deri
        iters -= 1
    return theta

def plotScatter(ax):
    pos = data[data[:,2] == 1]
    neg = data[data[:,2] == 0]
    ax.scatter(pos[:,0],pos[:,1],c='r',marker='+',label='Admitted')
    ax.scatter(neg[:,0],neg[:,1],c='g',marker='o',label='Not admitted')
    ax.set_xlabel('Exame 1 score')
    ax.set_ylabel('Exame 2 score')
    ax.legend(loc='best')

def plotBoundary(ax,theta):
    xmax = np.max(x[:,1])
    xmin = np.min(x[:,1])
    theta = np.array(theta.T)[0]
    score1 = np.arange(xmin,xmax,0.1)
    score2 = (-1.0/theta[2])*(theta[0]+theta[1]*score1)
    ax.plot(score1,score2,label='Boundary')
    ax.legend(loc='best')

def predict(x0,theta):
    x0 = np.hstack(([1],x0))
    x0 = np.mat(x0)
    print ("the probability is %.3f" % (sigmoid(x0*theta)))
    h = np.sum(sigmoid(x0*theta))
    if h > 0.5:
        return 1.0
    else:
        return 0.0

data = np.loadtxt('ex2data1.txt',delimiter=',')
x = np.mat(np.delete(data,-1,axis=1))
x = np.hstack((np.ones((x.shape[0],1)),x))
y = np.mat(data[:,-1]).T
m,n = x.shape
    
if __name__ == '__main__':
    alpha = 0.043
    theta = np.mat(np.zeros((n,1)))
    iters = 1000000
    fig, ax = plt.subplots()
    plotScatter(ax)
    theta = decent(alpha,theta,iters)    
    print theta
    print predict([45,85],theta)
    plotBoundary(ax,theta)
    #print costFunction(x,y,theta)
    plt.show()


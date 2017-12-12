#!/usr/bin/python

#Exercise 1: Linear Regression

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from linear import *

def normalization(data):
    """By looking at the values, note that house sizes are about 1000 times the number of
    bedrooms.When features differ by orders of magnitude, first performing feature scaling can make gradient descent coverge much more quickly. 
    """
    data = np.array(data)
    mean = np.mean(data,axis=0)
    stdDev = np.sqrt(np.var(data,axis=0))
    for i in np.arange(0,data.shape[0]):
        data[i] = (data[i] - mean)/stdDev
    return data

def originalSolve(data):
    iters = [10000, 1000, 100, 10]
    # if the alpha is bigger, for example 0.0001, the cost function will not converge and not find the result
    alphas = [0.000000001, 0.00000001, 0.0000001, 0.000001]
    theta = np.zeros(3)
    x = data[:,0:2]
    y = data[:,-1]
    x = np.hstack((np.ones((x.shape[0],1)),x))
    fig, ax = plt.subplots(2,2)
    for i in np.arange(0,4):
        thet,j = gradientSolve(theta,x,y,alphas[i],iters[i])
        #print normalEqn(mx,my)
        row = i / 2
        col = i % 2
        ax[row,col].plot(np.arange(1,iters[i]+1),j)
        ax[row,col].set_title('the alpha is %.10f' % (alphas[i]))
    plt.grid(True)
    plt.show()

def normalizationSovle(data):
    # features' normalization
    data = normalization(data)
    iters = 10000
    alpha = 0.001
    theta = np.zeros(3)
    x = data[:,0:2]
    y = data[:,-1]
    x = np.hstack((np.ones((x.shape[0],1)),x))
    theta,j = gradientSolve(theta,x,y,alpha,iters)
    print theta
    # use normal equation to get theta 
    print normalEqnSolve(x,y)
    fig, ax = plt.subplots()

    ax.plot(np.arange(1,iters+1),j)
    ax.set_title('The the curve between cost value  and iteration times')
    plt.show()

if __name__ == '__main__':
    data = np.loadtxt('ex1data2.txt',delimiter=',') 
    #originalSolve(data)
    normalizationSovle(data)

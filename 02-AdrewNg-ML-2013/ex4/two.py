#coding:utf-8
import os
import sys
sys.path.append('..')
import numpy as np
import scipy.optimize as op
import matplotlib
from matplotlib import pyplot as plt
from one import *


def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z));

def sigmoidGrad(z):
    """
    compute the gradient of sigmoid
    z: a number or a np.ndarray
    """
    return sigmoid(z) * (1 - sigmoid(z))

def randInit(lin,lout):
    """
    lin and lout are  the number of units in the layers adjacent to Θ(l)
    """
    epsilon = 0.12
    # A good choice of epsilon is the following:
    #epsilon = np.sqrt(6)/(np.sqrt(lin) + np.sqrt(lout))
    return np.random.rand(lout,lin+1) * epsilon - epsilon

def backward(x,y,units,reg=True,lamda=0.0):
    m = np.size(x,0)
    # the number of output  class
    labnum = units[len(units)-1]
    # initialize thetas
    thetas = []
    for i in np.arange(1,len(units)):
        lin = units[i-1]
        lout = units[i]
        thetas.append( randInit(lin,lout) )
    yh,zlist,alist = forward(x,y,thetas)
    levels = len(alist)
    errlist = levels * [None]
    # 0,1..,levels-1
    for lev in range(levels-1,0,-1):
        # if it is the last level( output level)
        if lev == levels - 1:
            errlist[lev] = yh - expandY(y,labnum)
        else:
            # relative with err
            z = zlist[lev]
            err = errlist[lev+1].dot(thetas[lev])
            err = err[:,1:] * sigmoidGrad(z)
            errlist[lev] = err

    # deltas[i].shape == thetas[i].shape
    deltas = len(thetas)*[None]
    grads = len(thetas)*[None]
    for i in range(0,len(thetas)):
        # relative with delta
        a = alist[i]
        a = np.hstack((np.ones((np.size(a,0),1)),a))
        delta = (errlist[i+1].T).dot(a)
        deltas[i] = delta
        grads[i] = 1.0/m * delta

    if reg:
        for i in range(0,len(thetas)):
            theta = thetas[i]
            # the first col of theta is not involed in calculation
            zero = np.zeros((np.size(theta,0),1))
            theta = np.hstack((zero,theta[:,1:]))
            grads[i] = grads[i] + lamda/m * theta

    return grads

if __name__ == '__main__':
    path='../ex3/'
    # load and rearrange data
    x = np.loadtxt(path+'x.txt')
    #x = np.hstack((np.ones((np.size(x,0),1)),x))
    y = np.loadtxt(path+'y.txt')
    theta1 = np.loadtxt(path+'theta1.txt')
    theta2 = np.loadtxt(path+'theta2.txt')
    #print predict(x,y,[theta1,theta2])
    #expandY(y,10)
    #print costFunc(x,y,[theta1,theta2])
    #print costFunc(x,y,[theta1,theta2],reg=True,lamda=1.0)
    #print randInit(4,5)
    grands = backward(x,y,[400,100,25,10],reg=True,lamda=0.0)

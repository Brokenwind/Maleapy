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

def backward(x,y,thetas,units,reg=True,lamda=0.0):
    """backward(x,y,units,reg=True,lamda=0.0)
    Implement the backpropagation algorithm to compute the gradient for the neural network cost function.
    x,y: the input data
    units: it is a list, units[i] means how many features in level i+1
    reg: if it is True, means using regularized neural networks
    lamda: it will be used when reg is True
    """
    # --------------------------------STEP 0--------------------------------
    m = np.size(x,0)
    # the number of output  class
    labnum = units[len(units)-1]
    # --------------------------------STEP 1--------------------------------
    yh,zlist,alist = forward(x,y,thetas)
    # ------------------------------STEP (2,3)------------------------------
    levels = len(alist)
    errlist = levels * [None]
    # 0,1..,levels-1
    for lev in range(levels-1,0,-1):
    # --------------------------------STEP 2--------------------------------
        # if it is the last level( output level)
        if lev == levels - 1:
            errlist[lev] = yh - expandY(y,labnum)
        else:
            z = zlist[lev]
            err = errlist[lev+1].dot(thetas[lev])
    # --------------------------------STEP 3--------------------------------
            err = err[:,1:] * sigmoidGrad(z)
            errlist[lev] = err
    # --------------------------------STEP 4--------------------------------
    deltas = len(thetas)*[None]
    grads = len(thetas)*[None]
    one = np.ones((m,1))
    for i in range(0,len(thetas)):
        a = alist[i]
        # when we use forward propatation algorithm to compute one specified level, we will use the output of last level a as the input, but it needs to be added bias col. So when we  compute delta conversely, the relative a should add a col of bias too.
        a = np.hstack((one,a))
        delta = (errlist[i+1].T).dot(a)
        deltas[i] = delta
    # --------------------------------STEP 5--------------------------------
        grads[i] = 1.0/m * delta

    if reg:
        for i in range(0,len(thetas)):
            theta = thetas[i]
            # the first col of theta is not involed in calculation
            zero = np.zeros((np.size(theta,0),1))
            theta = np.hstack((zero,theta[:,1:]))
            grads[i] = grads[i] + lamda/m * theta

    return grads

def numericalGradient(costFunc,x,y,thetas,reg=False,lamda=0.0):
    check = 1e-4
    m = len(thetas)
    # stores the shape of each element of thetas
    shapes = m * [None]
    # stores the flattened thetas
    ftheta = np.array([])
    for i in range(0,m):
        shapes[i] = thetas[i].shape
        ftheta = np.hstack( (ftheta,thetas[i].flatten()) )
    # the total number of theta(neural network parameters)
    thetanum = ftheta.size
    # stores all the approximate gradients of thetase in cost function 
    numegrad = np.zeros(thetanum)
    for i in range(0,thetanum):
        tmp = ftheta[i]
        ftheta[i] = tmp + check
        up = costFunc(x,y,ftheta,reg,lamda)
        ftheta[i] = tmp - check
        down = costFunc(x,y,ftheta,reg,lamda)
        numegrad[i] = (up - down)/(2*check)
        # restore the ftheta
        ftheta[i] = tmp

    # reshape the flattened numegrad
    grads = m * [None]
    start = 0
    for i in range(0,m):
        row,col = shapes[i]
        end = row * col
        grads[i] = np.reshape(numegrad[start:end],(row,col))
        start = end
    return grands

if __name__ == '__main__':
    path='../ex3/'
    # load and rearrange data
    x = np.loadtxt(path+'x.txt')
    #x = np.hstack((np.ones((np.size(x,0),1)),x))
    y = np.loadtxt(path+'y.txt')
    #theta1 = np.loadtxt(path+'theta1.txt')
    #theta2 = np.loadtxt(path+'theta2.txt')
    #print predict(x,y,[theta1,theta2])
    #expandY(y,10)
    #print costFunc(x,y,[theta1,theta2])
    #print costFunc(x,y,[theta1,theta2],reg=True,lamda=1.0)
    #print randInit(4,5)
        # initialize thetas
    units = [400,25,10]
    thetas = []
    for i in np.arange(1,len(units)):
        lin = units[i-1]
        lout = units[i]
        thetas.append( randInit(lin,lout) )
    gra1 = backward(x,y,thetas,units,reg=False,lamda=0.0)
    gra2 = numericalGradient(costFunc,x,y,thetas,reg=False,lamda=0.0)

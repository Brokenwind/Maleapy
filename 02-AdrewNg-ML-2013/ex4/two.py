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

def initTheta(units):
    """initialize the theta with given units
    """
    thetas = np.array([])
    for i in np.arange(1,len(units)):
        lin = units[i-1]
        lout = units[i]
        theta = randInit(lin,lout).flatten()
        thetas = np.hstack( (thetas,theta) )
    return thetas

def backward(thetas,x,y,units,reg=True,lamda=0.0):
    """backward(x,y,units,reg=True,lamda=0.0)
    Implement the backpropagation algorithm to compute the gradient for the neural network cost function.
    x,y: the input data
    units: it is a list, units[i] means how many features in level i+1
    reg: if it is True, means using regularized neural networks
    lamda: it will be used when reg is True
    """
    if isinstance(thetas,np.ndarray):
        thetas = reshapeList(thetas,units)
    # --------------------------------STEP 0--------------------------------
    m = np.size(x,0)
    # the number of output  class
    labnum = units[len(units)-1]
    # --------------------------------STEP 1--------------------------------
    yh,zlist,alist = forward(thetas,x,y,units)
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
    # --------------------------------STEP 3--------------------------------
            z = zlist[lev]
            err = errlist[lev+1].dot(thetas[lev])
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
        deltas[i] = (errlist[i+1].T).dot(a)
        grads[i] = 1.0/m * deltas[i]
    # --------------------------------STEP 5--------------------------------
    if reg:
        for i in range(0,len(thetas)):
            theta = thetas[i]
            # the first col of theta is not involed in calculation
            zero = np.zeros((np.size(theta,0),1))
            theta = np.hstack((zero,theta[:,1:]))
            grads[i] =  grads[i] + lamda/m * theta
            
    return flattenList(grads)

def numericalGradient(thetas,x,y,units,reg=False,lamda=0.0):
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
        up = costFunc(ftheta,x,y,units,reg,lamda)
        ftheta[i] = tmp - check
        down = costFunc(ftheta,x,y,units,reg,lamda)
        numegrad[i] = (up - down)/(2.0*check)
        # restore the ftheta
        ftheta[i] = tmp

    return numegrad

def debugInit(lin,lout):
    """
    Initialize the weights of a layer with lin incoming connections and 
    lout outgoing connections using a fixed strategy, 
    this will help you later in debugging

    Return:
     a matrix of size(lout, lin + 1) as  first row of W handles the "bias" terms
    """
    lin += 1
    """
    Initialize theta using "sin", this ensures that W is always of the same
    values and will be useful for debugging
    """
    theta = np.sin(np.arange(1,lin*lout+1))/10.0
    return np.reshape(theta,(lout,lin))

def norm(x,p=2):
    return np.sum(np.abs(x) ** p) ** (1.0/p)

def checkGradient(reg=False,lamda = 0.0):
    m = 5
    units = [3,5,3]
    thetas = []
    level = len(units)
    for i in np.arange(1,len(units)):
        lin = units[i-1]
        lout = units[i]
        thetas.append( debugInit(lin,lout) )

    # Reusing debugInitializeWeights to generate X
    x = debugInit(units[0]-1, m)
    y = 1 + np.mod(np.arange(1,m+1),level)
    """
    gra1 = backward(thetas,x,y,units,reg,lamda)
    gra2 = numericalGradient(thetas,x,y,units,reg,lamda)
    fgra1 = np.array([])
    fgra2 = np.array([])
    for i in range(0,len(gra1)):
        fgra1 = np.hstack((fgra1,gra1[i].flatten()))
        fgra2 = np.hstack((fgra2,gra2[i].flatten()))
    """
    fgra1 = backward(thetas,x,y,units,reg,lamda)
    fgra2 = numericalGradient(thetas,x,y,units,reg,lamda)
    return norm(fgra1 - fgra2)/norm(fgra1 + fgra2)

def optimSolve(theta,x,y,units,reg=False,lamda=0.0):
    lamda *= 1.0
    theta = theta.flatten()
    res = op.minimize(fun = costFunc, x0 = theta,args = (x, y,units,reg,lamda),method = 'TNC',jac = backward);
    # use BFGS minimization algorithm. but it will not work well because costFunc may return a NaN value
    #print op.fmin_bfgs(costFunc, initial_theta, args = (x,y), fprime=gradient)
    return res.success,res.x

if __name__ == '__main__':
    path='../ex3/'
    # load and rearrange data
    x = np.loadtxt(path+'x.txt')
    #x = np.hstack((np.ones((np.size(x,0),1)),x))
    y = np.loadtxt(path+'y.txt')
    units = [400,25,10]
    #theta1 = np.loadtxt(path+'theta1.txt')
    #theta2 = np.loadtxt(path+'theta2.txt')
    #print predict([theta1,theta2],x,y)
    #expandY(y,10)
    #print costFunc([theta1,theta2],x,y,units)
    #print costFunc(thetas,x,y,units)
    #print costFunc([theta1,theta2],x,y,reg=True,lamda=1.0)
    #print randInit(4,5)
    #thetas = initTheta(units)
    #gra1 = backward(thetas,x,y,units,reg=False,lamda=0.0)
    #gra2 = numericalGradient(thetas,x,y,units,reg=False,lamda=0.0)
    #print debugInit(4,3)
    #print checkGradient(True,3.0)
    # print checkGradient()
    thetas = initTheta(units)
    status, res = optimSolve(thetas,x,y,units,True,1.0)
    if status:
        ltheta = reshapeList(res,units)
        for i in range(0,len(ltheta)):
            name = 'theta'+str(i+1)+'.txt'
            np.savetxt(name,ltheta[i])
        pred = predict(res,x,y,units)
        m,n = x.shape
        rate = sum(np.ones(m)[pred==y])*1.0/m
        print ('The accuracy rate is : %.4f' % (rate))
    else:
        print (' Can not converge, please try again!')


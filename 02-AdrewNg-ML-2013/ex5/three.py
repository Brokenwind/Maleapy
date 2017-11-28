#!/usr/bin/python
import sys
sys.path.append('../')
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from ex1.linear import *

def polyFeatures(x,p):
    m = x.size
    X = np.zeros((p,m))
    for i in range(1,p+1):
        X[i-1] = np.power(x,i)
    return X.T

def normalization(data):
    """When features differ by orders of magnitude, first performing feature scaling can make gradient descent coverge much more quickly. 
    """
    data = np.array(data)
    mean = np.mean(data,axis=0)
    stdDev = np.sqrt(np.var(data,axis=0))
    for i in np.arange(0,data.shape[0]):
        data[i] = (data[i] - mean)/stdDev
    return data,mean,stdDev

def showCurve(ax,theta,mean,std):
    min = -60
    max = 45
    x = np.linspace(min,max,100)
    ex = polyFeatures(x,8)
    for i in range(0,np.size(ex,0)):
        ex[i] = (ex[i] - mean)/std
    one = np.ones((np.size(ex,0),1))
    ex = np.hstack((one,ex))
    theta = theta.reshape((np.size(theta,0),1))
    yh = ex.dot(theta)
    ax.plot(x,yh)

if __name__ == '__main__':
    fig, ax = plt.subplots()
    p = 8
    x = np.loadtxt('x.txt')
    y = np.loadtxt('y.txt')
    xval = np.loadtxt('xval.txt')
    yval = np.loadtxt('yval.txt')
    # plot the scatter of original data
    ax.scatter(x,y,c='r',marker='x',label='Original')
    ax.set_xlabel('change in water level')
    ax.set_ylabel('water flowing out of the dam')
    ex =  polyFeatures(x,p)
    ex,mean,std =  normalization(ex)
    one = np.ones((np.size(ex,0),1))
    ex = np.hstack((one,ex))
    m,n = ex.shape
    # calculate the learning parameters
    thetaInit = np.ones(n)
    status,theta = optimSolve(thetaInit,ex,y,reg=True,lamda=1.0)
    print status
    print theta
    # show the result curve
    showCurve(ax,theta,mean,std)
    
    plt.show()

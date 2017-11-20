#!/usr/bin/python

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def expandX(x,exp):
    """
    x: the input data(a matrix)
    exp: the max index of x. (x1^i * x2^j, i + j = exp)
    """
    for index in np.arange(2,exp+1):
        param1 = x[:,1]
        param2 = x[:,2]
        for i in np.arange(0,index+1):
            tmp1 = np.array(np.power(param1,i))
            tmp2 = np.array(np.power(param2,index - i))
            x = np.hstack((x,np.mat(tmp1*tmp2)))
    return x

def sigmoid(z):
    """sigmoid(z)
    z: It can be a array, a narray, a vector and a matrix
    """
    return 1.0 / (1.0 + np.exp(-z))

def costFunction(x,y,theta,lamda):
    m = x.shape[0]
    y1 = np.array(sigmoid(x*theta))
    #print y1
    y0 = np.array(y)
    cost = (-1.0/m)*np.sum(y0 * np.log(y1) + (1.0 - y0) * np.log(1.0 - y1))
    # the theta 0 is not involed
    theta = np.delete(theta,0,axis=0)
    regu = np.sum(np.power(theta,2))*lamda/(2*m)
    return cost + regu

def decent(x,y,alpha,theta,iters,lamda):
    m = x.shape[0]
    while iters > 0:
        h = np.array(sigmoid(x*theta))
        error = h - y
        deri = (1.0/m)*(x.T * error) + (lamda/m)*theta
        # the theta 0 is not involed
        deri[0,0] -= (lamda/m)*theta[0,0]
        theta = theta - alpha*deri
        iters -= 1
    return theta

def plotScatter(ax,data):
    pos = data[data[:,2] == 1]
    neg = data[data[:,2] == 0]
    ax.scatter(pos[:,0],pos[:,1],c='r',marker='+',label='y = 1')
    ax.scatter(neg[:,0],neg[:,1],c='g',marker='o',label='y = 0')
    ax.set_xlabel('Microchip Test1')
    ax.set_ylabel('Microchip Test2')
    ax.legend(loc='best')

def plotBoundary(ax,x,theta):
    x1 = np.linspace(-1,1.5,50)
    x2 = np.linspace(-1,1.5,50)
    z = np.zeros((np.size(x1),np.size(x2)))
    for i in np.arange(0,np.size(x1)):
        for j in np.arange(0,np.size(x2)):
            z[i,j] = expandX(np.mat([1,x1[i],x2[j]]),6)*theta
    x1,x2 = np.meshgrid(x1,x2)
    ax.contour(x1,x2,z,[0],label='Boundary')
    ax.legend(loc='best')

def predict(x,theta):
    x = np.hstack(([1],x))
    x = np.mat(x)
    x = expandX(x,6)
    print ("the probability is %.3f" % (sigmoid(x*theta)))
    h = np.sum(sigmoid(x*theta))
    if h > 0.5:
        return 1.0
    else:
        return 0.0
    
if __name__ == '__main__':
    index = 6
    alpha = 0.05
    iters = 10000
    lamda = 0
    fig, ax = plt.subplots()
    data = np.loadtxt('ex2data2.txt',delimiter=',')
    plotScatter(ax,data)
    x = np.mat(np.delete(data,-1,axis=1))
    x = np.hstack((np.ones((x.shape[0],1)),x))
    x = expandX(x,index)
    y = np.mat(data[:,-1]).T
    row,col = x.shape
    # initialize theta with zero vector
    theta = np.mat(np.zeros((col,1)))
    theta = decent(x,y,alpha,theta,iters,lamda) 
    print theta
    predict([0.25,1.5],theta)
    plotBoundary(ax,x,theta)
    print costFunction(x,y,theta,lamda)

    plt.show()

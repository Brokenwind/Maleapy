#!/usr/bin/python

#Exercise 1: Linear Regression

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from linear import *

# plot the 3D's cost function 
def visualCost3D(x,y):
    """visualCost3D(cost,xrange,yrange):
    cost: the cost function you will call
    xrange: xrange[0]: the start point; xrange[1]: the end point; xrange[2]: the number of data
    yrange: yrange[0]: the start point; yrange[1]: the end point; yrange[2]: the number of data
    """
    theta0 = np.linspace(-10,10,200)
    theta1 = np.linspace(-1,4,200)
    jvals = np.zeros((theta0.size,theta1.size))
    for i in np.arange(0,theta0.size):
        for j in np.arange(0,theta1.size):
            theta = np.hstack((theta0[i],theta1[j])).T
            jvals[i,j] = costFunc(theta,x,y)
    fig = plt.figure()
    ax = Axes3D(fig)
    x,y = np.meshgrid(theta0,theta1)
    ax.plot_surface(x,y,jvals,cmap='rainbow')
    ax.set_title('Cost Function Surface')

def contour(res,x,y):
    res = res.flatten()
    theta0 = np.linspace(-10,10,200)
    theta1 = np.linspace(-1,4,200)
    jvals = np.zeros((theta0.size,theta1.size))
    for i in np.arange(0,theta0.size):
        for j in np.arange(0,theta1.size):
            theta = np.hstack((theta0[i],theta1[j])).T
            jvals[i,j] = costFunc(theta,x,y)
    fig,ax = plt.subplots()
    x,y = np.meshgrid(theta0,theta1)
    ax.scatter(res[0],res[1])
    ax.contour(x,y,jvals,np.logspace(-2,2,50))
    ax.set_title('Contour of Cost Function')
    

if __name__ == '__main__':
    iters = 15000
    theta = np.zeros(2)
    alpha = 0.01
    data = np.loadtxt('ex1data1.txt',delimiter=',') 
    m = np.size(data,0)
    x = data[:,0]
    y = data[:,1]

    # scatterplot of original data
    fig, ax = plt.subplots(1,2,figsize=(20,10))
    ax[0].scatter(x,y)

    mx = x.reshape((m,1))
    mx = np.hstack((np.ones((m,1)),mx))
    my = y.reshape((m,1))
    print costFunc(theta,mx,my)

    theta,j = gradientSolve(theta,mx,my,alpha,iters)
    print theta
    theta = optimSolve(theta,mx,my)
    print theta 
    print predict(theta,[20])
    # plot the  predict values
    theta = theta.reshape((theta.size,1))
    ax[0].plot(x,mx.dot(theta))
    ax[0].set_title('Original scatterplot and Predict line')

    # plot the cost curve
    ax[1].plot(np.arange(1,iters+1),j)
    ax[1].set_title('The the curve between cost value  and iteration times')

    #print predict(theta,3500)
    visualCost3D(mx,my)
    contour(theta,mx,my)
    plt.show()
    

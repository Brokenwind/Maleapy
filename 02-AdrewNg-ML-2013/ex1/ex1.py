#!/usr/bin/python

#Exercise 1: Linear Regression

import numpy as np
import matplotlib
from matplotlib import pyplot as plt

def computeCost(x,y,theta):
    """computeCost(x,y,theta)
    x: is a  coefficient matrix
    y: the result matrix (result vector)
    theta: the learning rate
    """
    m = x.shape[0]
    tmp = x*theta - y
    return ((tmp.T * tmp)/(2*m))[0,0]

"""
# this is recursive implementation
def gradientDescent(x,y,theta,alpha,iterations):
    if iterations < 0:
        return theta
    else:
        down =  ((x*theta - y).T * x)
        down = (alpha/x.shape[0]) * down.T
        theta = theta - down
        return gradientDescent(x,y,theta,alpha,iterations-1)
"""

# iteration implementation
def gradientDescent(x,y,theta,alpha,iterations):
    jhistory = []
    while iterations > 0:
        down =  ((x*theta - y).T * x)
        down = (alpha/x.shape[0]) * down.T
        theta = theta - down
        
        iterations -= 1
        jhistory.append(computeCost(x,y,theta))
    return theta,jhistory

# pridict the the value for the given x, according to the calculated  theta
def predict(theta,x):
    """
    x: the input value
    theta:   calculated parameters matrix
    """
    return np.mat([1,x]) * theta

if __name__ == '__main__':
    iterations = 1500
    theta = np.mat([0,0]).T
    alpha = 0.01
    data = np.loadtxt('ex1data1.txt',delimiter=',') 
    x = data[:,0]
    y = data[:,1]

    # scatterplot of original data
    fig, ax = plt.subplots(1,2,figsize=(20,10))
    print ax
    ax[0].scatter(x,y)

    #mx = np.mat(np.c_[np.ones((data.shape[0],1)),x]) 
    mx = np.hstack((np.ones((data.shape[0],1)),np.mat(x).T))
    my = np.mat(y).T
    
    theta,j = gradientDescent(mx,my,theta,alpha,iterations)

    # plot the  predict values
    ax[0].plot(mx[:,1],mx*theta)

    # plot the cost curve
    #ax[1].plot(np.arange(1,iterations+1),j)
    ax[1].plot(np.arange(1,iterations+1),j)
    plt.show()
    print predict(theta,3500)

#data = loadData('ex1data1.txt')
#plotData(data[0],data[1])

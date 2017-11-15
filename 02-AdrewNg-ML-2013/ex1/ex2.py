#!/usr/bin/python

#Exercise 1: Linear Regression

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def computeCost(x,y,theta):
    """computeCost(x,y,theta)
    x: is a  coefficient matrix
    y: the result matrix (result vector)
    theta: the learning rate
    """
    m = x.shape[0]
    tmp = x*theta - y
    return ((tmp.T * tmp)/(2*m))[0,0]

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

def normalization(data):
    """By looking at the values, note that house sizes are about 1000 times the number of
    bedrooms.When features differ by orders of magnitude, first performing feature scaling can make gradient descent coverge much more quickly. 
    """
    #data = np.mat(data)
    mean = np.mean(data,axis=0)
    max = np.max(data,axis=0)
    min = np.min(data,axis=0)
    wid = max - min
    for i in np.arange(0,data.shape[0]):
        data[i] = (data[i] - mean)/wid
    return data

# pridict the the value for the given x, according to the calculated  theta
def predict(theta,data):
    """
    x: the input value
    theta:   calculated parameters matrix
    """
    return np.mat([1].extend(data)) * theta


if __name__ == '__main__':
    iterations = 1500
    alpha = 0.05
    data = np.loadtxt('ex1data2.txt',delimiter=',') 
    data = normalization(data)
    theta = np.mat(np.zeros((data.shape[1],1)))
    x = np.delete(data,-1,axis=1)
    y = data[:,-1]
    mx = np.hstack((np.ones((x.shape[0],1)),np.mat(x)))
    my = np.mat(y).T

    theta,j = gradientDescent(mx,my,theta,alpha,iterations)
    print theta
    fig, ax = plt.subplots()

    #ax.plot(np.arange(1,iterations+1),j)
    ax.plot(np.arange(1,iterations+1),j)
    ax.set_title('The the curve between cost value  and iteration times')
    #print predict(theta,3500)
    plt.show()



#data = loadData('ex1data1.txt')
#plotData(data[0],data[1])

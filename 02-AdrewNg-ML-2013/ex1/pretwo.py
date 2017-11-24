#!/usr/bin/python

#Exercise 1: Linear Regression

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def costFunc(x,y,theta):
    """costFunc(x,y,theta)
    x: is a  coefficient matrix
    y: the result matrix (result vector)
    theta: the learning rate
    """
    m = x.shape[0]
    tmp = x*theta - y
    return ((tmp.T * tmp)/(2*m))[0,0]

# iteration implementation
def gradient(x,y,theta,alpha,iterations):
    jhistory = []
    while iterations > 0:
        down =  ((x*theta - y).T * x)
        down = (alpha/x.shape[0]) * down.T
        theta = theta - down
        iterations -= 1
        jhistory.append(costFunc(x,y,theta))
    return theta,jhistory

def normalization(data):
    """By looking at the values, note that house sizes are about 1000 times the number of
    bedrooms.When features differ by orders of magnitude, first performing feature scaling can make gradient descent coverge much more quickly. 
    """
    #data = np.mat(data)
    mean = np.mean(data,axis=0)
    """
    max = np.max(data,axis=0)
    min = np.min(data,axis=0)
    wid = max - min
    """
    stdDev = np.sqrt(np.var(data,axis=0))
    for i in np.arange(0,data.shape[0]):
        data[i] = (data[i] - mean)/stdDev
    return data

def normalEqn(x,y):
    return (x.T * x).I * (x.T * y)
    

# pridict the the value for the given x, according to the calculated  theta
def predict(theta,data):
    """
    x: the input value
    theta:   calculated parameters matrix
    """
    return np.mat([1].extend(data)) * theta

def originalSolve(data):
    iterations = [10000, 1000, 100, 10]
    # if the alpha is bigger, for example 0.0001, the cost function will not converge and not find the result
    alphas = [0.000000001, 0.00000001, 0.0000001, 0.000001]
    
    theta = np.mat(np.zeros((data.shape[1],1)))
    x = np.delete(data,-1,axis=1)
    y = data[:,-1]
    mx = np.hstack((np.ones((x.shape[0],1)),np.mat(x)))
    my = np.mat(y).T
    fig, ax = plt.subplots(2,2)
    for i in np.arange(0,4):
        thet,j = gradient(mx,my,theta,alphas[i],iterations[i])
        print thet
        #print normalEqn(mx,my)
        row = i / 2
        col = i % 2
        ax[row,col].plot(np.arange(1,iterations[i]+1),j)
        ax[row,col].set_title('the alpha is %.10f' % (alphas[i]))
    plt.grid(True)
    plt.show()

def normalizationSovle(data):
    # features' normalization
    data = normalization(data)
    iterations = 10000
    alpha = 0.001
    theta = np.mat(np.zeros((data.shape[1],1)))
    x = np.delete(data,-1,axis=1)
    y = data[:,-1]
    mx = np.hstack((np.ones((x.shape[0],1)),np.mat(x)))
    my = np.mat(y).T

    theta,j = gradient(mx,my,theta,alpha,iterations)
    print theta
    # use normal equation to get theta 
    print normalEqn(mx,my)
    fig, ax = plt.subplots()

    ax.plot(np.arange(1,iterations+1),j)
    ax.set_title('The the curve between cost value  and iteration times')
    plt.show()

if __name__ == '__main__':
    data = np.loadtxt('ex1data2.txt',delimiter=',') 
    originalSolve(data)
    #normalizationSovle(data)

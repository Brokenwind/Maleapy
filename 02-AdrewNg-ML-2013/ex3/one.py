import sys
sys.path.append('..')
import numpy as np
import scipy.optimize as op
import matplotlib
from matplotlib import pyplot as plt
from ex2.optimlog import *

def proba(x,theta):
    """calculate probability of x  with the calculated theta
    """
    n = theta.size
    x = np.reshape(x,(1,np.size(x)))
    theta = theta.reshape((n,1))
    if x.size +1 == n:
        x = np.hstack(([1],x))
    if x.size != n:
        print ("cannot adjust x.size == theta.size ")
        return 0.0
    prob = sigmoid(x.dot(theta))[0]
    return prob

def predict(x,theta):
    """Return the most possible digit result of the picture data x
    """
    m,n = theta.shape
    max = 0.0
    res = 0
    for i in np.arange(1,m):
        prob = proba(x,theta[i,:])
        if prob > max:
            max = prob
            res = i
    return res

if __name__ == '__main__':
    # size of given pictures
    wid = 20
    hei = 20
    pixs = wid * hei
    #10 labels, from 1 to 10 
    labels = 10
    lamda = 0.1
    # load and rearrange data
    x = np.loadtxt('x.txt')
    x = np.hstack((np.ones((np.size(x,0),1)),x))
    m, n = x.shape
    y = np.loadtxt('y.txt')

    # all_res will be expanded to a 11 * n(n is 1 plus the number pixels of picture) matrix, which will store all calculated parameters
    # all_res[0,:] is redundant
    # all_res[i,:] ( 1 <= i <= 10 ) will store the  calculated parameters of label i
    all_res = np.zeros((1,n))
    for label in np.arange(1,labels+1):
        init_theta = np.zeros(n)
        yt = y.copy()
        yt[y != label] = 0
        yt[y == label] = 1
        res = optimSolve(init_theta,x,yt,reg=True,lamda=lamda)
        res = res.reshape((1,n))
        # add  classifier parameters of the current label
        all_res = np.vstack((all_res,res))
    print predict(x[100,:],all_res)
    
    

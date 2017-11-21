import sys
sys.path.append('..')
import numpy as np
import scipy.optimize as op
import matplotlib
from matplotlib import pyplot as plt
from ex2.optimlog import *

def predict(x,theta):
    n = theta.size
    if x.size > n:
        print ("x.size > theta.size, failed")
        return None
    if x.size +1 == n:
        x = np.hstack(([1],x))
    if x.size != n:
        print ("cannot adjust x.size == theta.size ")
        return None
    theta = theta.reshape((n,1))
    prob = sigmoid(x.dot(theta))[0]
    print ("the probability is %.3f" % (prob))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

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
    all_res = np.zeros((1,n))
    for label in np.arange(1,labels+1):
        init_theta = np.zeros(n)
        yt = y.copy()
        yt[y != label] = 0
        yt[y == label] = 1
        res = optimSolve(init_theta,x,yt,reg=True,lamda=lamda)
        res = res.reshape((1,n))
        all_res = np.vstack((all_res,res))
    for i in np.arange(0,500):
        predict(x[i,:],all_res[10,:])
    #print res
    
    
    

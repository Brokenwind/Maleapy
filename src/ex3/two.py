import sys
sys.path.append('..')
import numpy as np
import scipy.optimize as op
import matplotlib
from matplotlib import pyplot as plt
from ex2.optimlog import *
from pictrans import *

def predict(x,theta1,theta2):
    """Return the most possible digit result of the picture data x
    """
    labels = np.size(theta1,0)
    m = np.size(x,0)
    # a1: 5000 * 401(400 + 1(bias unit))
    a1 = np.hstack((np.ones((m,1)),x))
    # z2 = 5000 * 25
    z2 = a1.dot(theta1.T)
    a2 = sigmoid(z2)
    # a2: 5000 * 26(25 + 1(bias unit))
    a2 = np.hstack((np.ones((m,1)),a2))
    # z3: 5000 * 10
    z3 = a2.dot(theta2.T)
    a3 = sigmoid(z3)
    # col index of the max probality of each row
    pos = np.argmax(a3,axis=1)
    # predict values of each row of X
    pred = pos + 1
    return pred


if __name__ == '__main__':
    # load and rearrange data
    x = np.loadtxt('x.txt')
    y = np.loadtxt('y.txt')
    m,n = x.shape
    theta1 = np.loadtxt('theta1.txt')
    theta2 = np.loadtxt('theta2.txt')
    pred = predict(x,theta1,theta2)
    rate = sum(np.ones(m)[pred==y])*1.0/m
    print ('The accuracy rate is : %.4f' % (rate))

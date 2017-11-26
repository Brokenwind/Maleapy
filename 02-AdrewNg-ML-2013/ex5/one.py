import sys
sys.path.append('..')
import numpy as np
import scipy.optimize as op
import matplotlib
from matplotlib import pyplot as plt
# use the common functions relative with linear regression implemented in file ex1.linear.py
from ex1.linear import *

if __name__ == '__main__':
    fig, ax = plt.subplots()
    x = np.loadtxt('x.txt')
    y = np.loadtxt('y.txt')
    m = x.size
    ex = x.reshape((m,1))
    ex = np.hstack((np.ones((m,1)),ex))
    theta = np.ones(2)

    ax.scatter(x,y,c='r',marker='x',label='Original')
    ax.set_xlabel('change in water level')
    ax.set_ylabel('water flowing out of the dam')
    # the function gradient is in file ex1.linear.py
    print gradient(theta,ex,y,reg=True,lamda=1.0)
    # the function costFunc is in file ex1.linear.py
    print costFunc(theta,ex,y,reg=True,lamda=1.0)
    # the function optimSolve is in file ex1.linear.py
    status,res = optimSolve(theta,ex,y)
    print res
    # the function predict is in file ex1.linear.py
    ax.plot(x,predict(res,x),label='Predicted')
    ax.legend(loc='best')

    plt.show()
    

import os
import sys
sys.path.append('..')
import numpy as np
import scipy.optimize as op
import matplotlib
from matplotlib import pyplot as plt
from ex2.optimlog import *
from ex3.pictrans import *

def forward(x,y,thetas):
    """forward(x,y,thetas)
    x: the input of test data
    y: the output of test data
    thetas: it is a list of classifier parameters of each level.

    Return: the  final result a and all middle value z
    """
    # a is the output of previos level, and it is the input of current level
    a = x
    # m means how many rows of input data
    m = np.size(a,0)
    #extra bias unit
    bias = np.ones((m,1))
    # store the middle value during calculating
    alist = []
    zlist = []
    # the z of level 1 has nothing.
    zlist.append(0)
    alist.append(a)
    for theta in thetas:
        # add extra bias unit of current input(previos ouput)
        a = np.hstack((bias,a))
        # z middle result of current level
        z = a.dot(theta.T)
        # sigmoid(z) is the final output of current level, and it will be the input of next level
        a = sigmoid(z) 
        zlist.append(z)
        alist.append(a)

    return a,zlist,alist


def predict(x,y,thetas):
    """predict(x,y,thetas)
    x: the input of test data
    y: the output of test data
    thetas: it is a list of classifier parameters of each level.
    """
    res,_,_ = forward(x,y,thetas)
    # col index of the max probality of each row
    pos = np.argmax(res,axis=1)
    # predicted values of each row of X
    pred = pos + 1
    return pred

def expandY(y,n):
    """expandY(y,n)
    use vector to express each y[i]

    y: the array you will expand
    n: the number of class
    """
    m = np.size(y)
    yset = np.eye(n)
    # convert class label to its index in vector. 
    y = (y - 1).astype(int)
    # the result matrix
    yres = np.zeros((m,n))
    for i in np.arange(0,m):
        yres[i] = yset[y[i]]
    return yres

def costFuncOld(x,y,thetas,reg=False,lamda=0.0):
    """
    This function is deprecated, and doesn't use vectorization. You'd better to use  function costFun.

    x: the input test data
    y: the label of relative x
    thetas: a list of all levels of estimated  value of unknown parameter
    reg: if it is True, means using regularized logistic. Default False
    lamda: it is used when reg=True
    """
    res,_,_ = forward(x,y,thetas)
    m,n = res.shape
    # m: the number row of result
    # n: the number of class
    y = expandY(y,n)
    J = 0.0
    for i in np.arange(0,m):
        #the computation of cost function of each y(i)  is simmilar to the logistic regression cost function.
        #And you can compare it with the function costFunc in ex2/optimlog.py
        y1 = res[i,:]
        y0 = y[i,:]
        # when the y0 = 1, the cost function is as following:
        pos = np.log(y1);
        # when the y0 = 0, the cost function is as following:
        neg = np.log(1-y1);
        all = y0 * pos + (1 - y0) * neg;
        J += np.sum(all);
    J = -J/m
    #reg: if it is True, means using regularized logistic
    if reg:
        for theta in thetas:
            regSum = 0.0
            # the first col is not involed
            theta = np.delete(theta,0,axis=1)
            row = np.size(theta,0)
            for i in np.arange(0,row):
                regSum += np.sum(theta[i] * theta[i])
            J += lamda/(2.0*m)*regSum
            
    return J
    
def costFunc(x,y,thetas,reg=False,lamda=0.0):
    """
    x: the input test data
    y: the label of relative x
    thetas: a list of all levels of estimated  value of unknown parameter
    reg: if it is True, means using regularized logistic. Default False
    lamda: it is used when reg=True
    """
    yh,_,_ = forward(x,y,thetas)
    m,n = yh.shape
    # m: the number row of result
    # n: the number of class
    y = expandY(y,n)
    # vectorize the real value and predicted result
    y = y.flatten()
    yh = yh.flatten()
    #the computation of cost function of each y(i)  is simmilar to the logistic regression cost function.
    #And you can compare it with the function costFunc in ex2/optimlog.py
    # when the y = 1, the cost function is as following:
    pos = np.log(yh)
    # when the y = 0, the cost function is as following:
    neg = np.log(1-yh)
    all = y * pos + (1 - y) * neg
    J = -np.sum(all)/m
    #reg: if it is True, means using regularized logistic
    if reg:
        if  isinstance(thetas,list):
            for theta in thetas:
                theta = theta.flatten()
                J += lamda/(2.0*m)*(np.sum(theta * theta))
        if  isinstance(thetas,np.ndarray):
            J += lamda/(2.0*m)*(np.sum(thetas * thetas))

    return J


def showRes(x,theta,num):
    fig,ax = plt.subplots(1,2)
    # random select num digits
    rnd = np.random.randint(1,5000,num)
    sels = x[rnd,:]
    sels = np.delete(sels,0,axis=1)
    for i in np.arange(0,num):
        # show original hand-written digit
        pic = sels[i,:].reshape((20,20))
        ax[0].imshow(pic, cmap=plt.cm.gray)
        # show normal digit
        res = predict(sels[i,:],theta)
        res = int(res)
        filename = "./digit/"+str(res)+".png"
        img = plt.imread(filename)  
        ax[1].imshow(img)
    # adjust the space between pictues to 0
    #plt.subplots_adjust(hspace=0)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    path='../ex3/'
    # load and rearrange data
    x = np.loadtxt(path+'x.txt')
    #x = np.hstack((np.ones((np.size(x,0),1)),x))
    y = np.loadtxt(path+'y.txt')
    theta1 = np.loadtxt(path+'theta1.txt')
    theta2 = np.loadtxt(path+'theta2.txt')
    #print predict(x,y,[theta1,theta2])
    #expandY(y,10)
    print costFunc(x,y,[theta1,theta2])
    print costFunc(x,y,[theta1,theta2],reg=True,lamda=1.0)

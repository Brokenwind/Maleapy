import sys
sys.path.append('..')
import numpy as np
import scipy.optimize as op
import matplotlib
from matplotlib import pyplot as plt
from ex2.optimlog import *
from pictrans import *

def proba(x,theta):
    """calculate probability of x  with the calculated theta
    """
    n = theta.size
    x = x.flatten()
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

def oneVsAll(x,y):
    lamda = 0.1
    #10 labels, from 1 to 10 
    labels = 10
    m, n = x.shape
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
    return all_res

def showRes(x,y,theta):
    fig,ax = plt.subplots(1,3)
    # random select num digits
    rnd = np.random.randint(1,5000,1)
    sel = x[rnd,1:]
    # show original hand-written digit
    pic = sel.reshape((20,20))
    ax[0].imshow(pic, cmap=plt.cm.gray)
    ax[0].set_title('Hand-Written')
    # show correct digit
    filename = "./digit/"+str(int(y[rnd]))+".png"
    img = plt.imread(filename)
    ax[1].imshow(img)
    ax[1].set_title('Corret')
    # show normal digit
    res = predict(sel,theta)
    res = int(res)
    filename = "./digit/"+str(res)+".png"
    img = plt.imread(filename)
    ax[2].imshow(img)
    ax[2].set_title('Predicted')
    # adjust the space between pictues to 0
    #plt.subplots_adjust(hspace=0)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    # load and rearrange data
    x = np.loadtxt('x.txt')
    x = np.hstack((np.ones((np.size(x,0),1)),x))
    y = np.loadtxt('y.txt')
    theta = oneVsAll(x,y)
    showRes(x,y,theta)
    #print predict(x[100,:],theta)
    

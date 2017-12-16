import sys
sys.path.append('..')
import numpy as np
import scipy.optimize as op
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from svm import *

def plotScatter(ax,x,y):
    pos = x[y == 0]
    neg = x[y == 1]
    ax.scatter(pos[:,0],pos[:,1],c='r',marker='+',label='Positive')
    ax.scatter(neg[:,0],neg[:,1],c='g',marker='o',label='Negative')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.legend(loc='best')

def decide(x,y,xval,yval):
    """decide(x,y,xval,yval)
    to use the cross validation set xval, yval to determine the best C and sigma parameter to use
    PARAMETERS:
        x,y are thraining data
        xval,yval are cross validation data
    RETURN:
        the optimal c and sigma
    """
    cs = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30])
    sigmas = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30])
    err = np.zeros((cs.size,sigmas.size))
    for i in range(0,cs.size):
        for j in range(0,sigmas.size):
            model = svmtrain(x,y,cs[i],gaussianKernel,args={'sigma':sigmas[j]})
            pred = svmPredict(model,xval)
            err[i,j] = sum(pred!=yval)*1.0/yval.size
    pos = np.argmin(err)
    c = cs[pos/sigmas.size]
    sigma = sigmas[pos%sigmas.size]
    print ('the error rate is minimal when C is %.3f and sigma is %.3f' % (c,sigma))
    #plot the 3D surface
    fig = plt.figure()
    ax = Axes3D(fig)
    cs,sigmas = np.meshgrid(cs,sigmas)
    ax.plot_surface(cs,sigmas,err,cmap='rainbow')
    ax.set_title('Error Function Surface')

    return c,sigma

if __name__ == '__main__':
    path = './data/'
    fig, ax = plt.subplots()
    x3 = np.loadtxt(path+'x3.txt')
    y3 = np.loadtxt(path+'y3.txt')
    x3val = np.loadtxt(path+'x3val.txt')
    y3val = np.loadtxt(path+'y3val.txt')
    m,n = x3.shape
    plotScatter(ax,x3,y3)

    c,sigma = decide(x3,y3,x3val,y3val)
    print c,sigma
    # predict result with calculated model
    model = svmtrain(x3,y3,c,gaussianKernel,args={'sigma':sigma},tol=1e-3,iters=5)
    yp = svmPredict(model,x3)
    print ('accuracy rate: %.2f' % (sum(y3==yp)*1.0/m))
    curveBoundary(ax,x3,model)

    """
    model = svmtrain(x3,y3,1,linearKernel,tol=1e-3,iters=5)
    yp = svmPredict(model,x3)
    print ('accury rate: %.2f' % (sum(y3==yp)*1.0/m))
    linearBoundary(ax,x3,model)
    """

    plt.show()

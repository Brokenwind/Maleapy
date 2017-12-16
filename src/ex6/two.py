import sys
sys.path.append('..')
import numpy as np
import scipy.optimize as op
import matplotlib
from matplotlib import pyplot as plt
from svm import *

def plotScatter(ax,x,y):
    pos = x[y == 0]
    neg = x[y == 1]
    ax.scatter(pos[:,0],pos[:,1],c='r',marker='+',label='Positive')
    ax.scatter(neg[:,0],neg[:,1],c='g',marker='o',label='Negative')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.legend(loc='best')

if __name__ == '__main__':
    path = './data/'
    fig, ax = plt.subplots()
    x2 = np.loadtxt(path+'x2.txt')
    y2 = np.loadtxt(path+'y2.txt')
    m,n = x2.shape
    plotScatter(ax,x2,y2)

    # predict result with calculated model
    model = svmtrain(x2,y2,1,gaussianKernel,args={'sigma':0.1},tol=1e-3,iters=5)
    yp = svmPredict(model,x2)
    print ('accuray rate: %.2f' % (sum(y2==yp)*1.0/m))
    curveBoundary(ax,x2,model)
    """
    # use different C to calculate model
    for c in [1,10,100]:
        model = svmtrain(x2,y2,c,linearKernel,tol=1e-3,iters=20)
        linearBoundary(ax,x2,model)
    """
    plt.show()

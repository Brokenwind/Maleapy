import sys
sys.path.append('..')
import numpy as np
import scipy.optimize as op
import matplotlib
from matplotlib import pyplot as plt
from svm import *
from boundary import *

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
    x1 = np.loadtxt(path+'x1.txt')
    y1 = np.loadtxt(path+'y1.txt')
    m,n = x1.shape
    plotScatter(ax,x1,y1)
    
    # predict result with calculated model
    #model = svmtrain(x1,y1,1,gaussianKernel,tol=1e-3,iters=20)
    model = svmtrain(x1,y1,1,linearKernel,tol=1e-3,iters=20)
    yp = svmPredict(model,x1)
    print ('accury rate: %.2f' % (sum(y1==yp)*1.0/m))

    # use different C to calculate model
    for c in [1,10,100]:
        model = svmtrain(x1,y1,c,linearKernel,tol=1e-3,iters=20)
        linearBoundary(ax,x1,y1,model['w'],model['b'])
    plt.show()

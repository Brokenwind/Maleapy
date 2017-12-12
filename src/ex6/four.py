import sys
sys.path.append('..')
import numpy as np
import scipy.optimize as op
import matplotlib
from matplotlib import pyplot as plt
from svm import *

if __name__ == '__main__':
    path = './data/'
    x = np.load(path+'spamTrainX.npz')['x']
    x = x[0:1000]
    y = np.loadtxt(path+'spamTrainY.txt')
    y = y[0:1000]
    xt = np.loadtxt(path+'spamTestX.txt')
    yt = np.loadtxt(path+'spamTestY.txt')
    
    c = 0.1
    # predict result with calculated model
    model = svmtrain(x,y,c,linearKernel,tol=1e-3,iters=5)
    np.savez('model.npz',w=model['w'],b=model['b'])
    m = y.size
    yp = svmPredict(model,x)
    print ('accury rate: %.2f' % (sum(y==yp)*1.0/m))

    m = yt.size
    yp = svmPredict(model,xt)
    print ('accury rate: %.2f' % (sum(yt==yp)*1.0/m))

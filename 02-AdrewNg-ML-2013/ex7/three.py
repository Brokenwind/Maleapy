import sys
sys.path.append('..')
import numpy as np
import scipy.optimize as op
import matplotlib
from matplotlib import pyplot as plt
from pca import *

if __name__ == '__main__':
    fig, ax = plt.subplots(1,2)
    ex7data1 = np.load('ex7data1.npz')
    x = ex7data1['X']
    m,n = x.shape
    ax[0].scatter(x[:,0],x[:,1],c='b',marker='o')
    norm,mean,std = normalize(x)
    ax[1].scatter(norm[:,0],norm[:,1],c='b',marker='o')
    u,s = pca(norm)
    z = project(norm,u,1)
    xr = recovery(z,u,1)
    ax[1].scatter(xr[:,0],xr[:,1],c='r',marker='+')
    plt.show()

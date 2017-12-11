import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from gaussian import *

if __name__ == '__main__':
    fig, ax = plt.subplots()
    ex8data1 = np.load('ex8data1.npz')
    x = ex8data1['X']
    xval = ex8data1['Xval']
    yval = ex8data1['yval']
    m,n = x.shape
    ax.scatter(x[:,0],x[:,1],c='b',marker='x')
    # estimate parameters of gaussian distribution
    mu,sigma = paramEstimate(x)
    print sigma
    # calculate gaussian distribution on trainning data
    p =  gaussian(x,mu,sigma)
    # visualize the gaussian distribution
    visualize(ax,x,mu,sigma)
    # calculate gaussian distribution on validation data
    pval = gaussian(xval,mu,sigma)
    # select a best epsilon
    print threshold(yval,pval)
    epsilon,F1 = threshold(yval,pval)
    print epsilon
    outlier = x[p < epsilon]
    ax.scatter(outlier[:,0],outlier[:,1],c='r',marker='o')
    plt.show()    

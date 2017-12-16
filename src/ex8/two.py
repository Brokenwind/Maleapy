import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from gaussian import *

if __name__ == '__main__':
    ex8data2 = np.load('ex8data2.npz')
    x = ex8data2['X']
    xval = ex8data2['Xval']
    yval = ex8data2['yval']
    m,n = x.shape
    # estimate parameters of gaussian distribution
    mu,sigma = paramEstimate(x)
    # calculate gaussian distribution on trainning data
    p =  gaussian(x,mu,sigma)
    # calculate gaussian distribution on validation data
    pval = gaussian(xval,mu,sigma)
    # select a best epsilon
    epsilon,F1 = threshold(yval,pval)
    print('Best epsilon found using cross-validation: %e' % (epsilon))
    print('Best F1 on Cross Validation Set:  %.5f' % (F1))
    outlier = x[p < epsilon]
    print ('There are %d outliers in trainning dataset' % (np.size(outlier,0)))
    pred = pval < epsilon
    rate = 1.0*sum(pred == yval)/pred.size
    print ('There accuracy rate  in cross-validation dataset is %.2f' % (rate))

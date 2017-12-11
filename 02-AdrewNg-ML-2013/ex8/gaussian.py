import numpy as np
from numpy.linalg import det
import matplotlib
from matplotlib import pyplot as plt

def paramEstimate(x,multi=False):
    """
    This function estimates the parameters of a Gaussian distribution using the data in x
    PARAMETERS:
        x:  is the dataset with each n-dimensional data point in one row 
        multi: is this the 
    RETURN:
        mu:  is an n-dimensional vector , the mean of the data set
        sigma: the  covariances, an n x n matrix
    """
    mu = np.mean(x,0)
    if multi:
        sigma = x.T.dot(x)
    else:
        sigma = np.std(x,0)**2
        sigma = np.diag(sigma) 
    
    return mu,sigma
    
def gaussian(x,mu,sigma):
    """
    Computes the probability density function of the multivariate gaussian distribution.
    """
    if x.ndim == 1:
        x = np.reshape((1,x.size))
    x = x - mu
    k = np.size(sigma,0)
    inv = np.array(np.mat(sigma).I)
    p = (2.0 * np.pi)**(- k / 2.0) * det(sigma)**(-0.5) * np.exp(-0.5 *np.sum(x.dot(inv) * x,1))
    return p

def  threshold(yval,pval):
    """
    finds the best threshold to use for selecting outliers based on the results from a validation set (pval) and the ground truth (yval).
    """
    smax = max(pval) 
    smin = min(pval) 
    step = ( smax - smin )/1000
    bestF1 = 0
    bestEps = 0
    # can not start from smin for tp and fp will be zero
    for epsilon in np.arange(smin+step,smax,step) :
        pred = (pval < epsilon) + 0
        tp = sum((pred == 1) & (yval == 1))*1.0
        fp = sum((pred == 1) & (yval == 0))*1.0
        fn = sum((pred == 0) & (yval == 1))*1.0
        #print tp,fp,fn
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        F1 = 2.0 * prec * rec / (prec + rec)
        if F1 > bestF1:
            bestF1 = F1
            bestEps = epsilon

    return bestEps,bestF1

def visualize(ax,x,mu,sigma):
    """
    This visualization shows you the probability density function of the Gaussian distribution. Each example has a location (x1, x2) that depends on its feature values.
    """
    num = 100
    x1 = np.linspace(np.min(x[:,0]),np.max(x[:,0]),num)
    x2 = np.linspace(np.min(x[:,1]),np.max(x[:,1]),num)
    z = np.zeros((num,num))
    x1,x2 = np.meshgrid(x1,x2)
    for i in np.arange(0,num):
        xp = np.vstack((x1[i],x2[i])).T
        z[i] = gaussian(xp,mu,sigma)
    """
    xs = np.hstack((x1,x2))
    z = gaussian(xs,mu,sigma)
    z = z.reshape(x1.shape)
    """
    rang = [10.0 ** i for i in  range(-20,0,2)]
    ax.contour(x1,x2,z,rang,label='Boundary')
    ax.legend(loc='best')

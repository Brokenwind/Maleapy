import numpy as np
from numpy.linalg import svd
import scipy.optimize as op
import matplotlib
from matplotlib import pyplot as plt

def normalize(x):
    """
    returns a normalized version of X where the mean value of each feature is 0 and the standard deviation is 1. This is often a good preprocessing step to do when working with learning algorithms.
    """
    mean = np.mean(x,0)
    std = np.std(x,0)
    norm = (x - mean)/std
    return norm,mean,std

def pca(x):
    """
    computes eigenvectors of the covariance matrix of X
    Returns the eigenvectors U, the eigenvalues (on diagonal) in S
    """
    m,n = x.shape
    Sigma = 1.0/m * x.T.dot(x)
    u,s,v = svd(Sigma)
    return u,s

def project(x,u,k):
    """
computes the projection of  the normalized inputs X into the reduced dimensional space spanned by the first K columns of U. It returns the projected examples.
    """
    return x.dot(u[:,0:k])

def recovery(z,u,k):
    """
    recovers an approximation the  original data that has been reduced to K dimensions. 
    It returns the approximate reconstruction.
    """
    return z.dot(u[:,0:k].T)

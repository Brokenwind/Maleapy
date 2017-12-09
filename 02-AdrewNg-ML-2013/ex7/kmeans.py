import sys
sys.path.append('../')
import numpy as np
import scipy.optimize as op
import matplotlib
import random
from matplotlib import pyplot as plt
from color import *

def closestCentroids(x,centroids):
    """closestCentroids(x,centroids):
    it returns the closest centroids in idx for a dataset X where each row is a single example. 
    idx is a row vector of centroid assignments (i.e. each entry in range [1..K])
    """
    m = np.size(x,0)
    idx = np.zeros(m)
    min = np.zeros(m)
    min.fill(2**32)
    for i in range(0,np.size(centroids,0)):
        dis = x - centroids[i]
        sums = np.sum(dis * dis,1)
        judge = sums < min
        idx[judge] = i
        min[judge] = sums[judge]
    return idx

def computeCentroids(x,idx,k):
    """
    it returns the new centroids by computing the means of the data points assigned to each centroid. 
    It is given a dataset x where each row is a single data point, a vector
    idx of centroid assignments (i.e. each entry in range [1..K]) for each
    example, and k, the number of centroids. You should return a matrix
    centroids, where each row of centroids is the mean of the data points
    assigned to it.
    """
    centroids = np.zeros((k,np.size(x,1)))
    for i in range(0,k):
        sel = idx == i
        centroids[i] = np.sum(x[sel],0)/np.sum(sel)
    return centroids
        
def kmeans(x,centroids,tol=1e-3,iters=100):
    """runs the K-Means algorithm on data matrix X, where each row of X is a single example
    PARAMETERS:
    X:  a matrix where each row of X is a single example. 
    centroids:  used as the initial centroids. 
    tol: the max deviation between two set of centroids
    iters: specifies the max number of interactions of K-Means to execute. 
    """
    k,n = centroids.shape
    history = []
    pre = centroids
    history.append(centroids)
    i = iters
    while i > 0:
        idx = closestCentroids(x,centroids)
        centroids = computeCentroids(x,idx,k)
        history.append(centroids)
        if np.max(np.abs((pre - centroids))) < tol:
            break
        pre = centroids
        i -= 1
    else:
        print ('Can not get correct result within %d iterations' % (iters))
    print ("Get result after %d iterations." % (len(history)))

    return idx,history


def plotResult(ax,x,idx):
    # get different class
    k = len(set(idx.flatten()))
    cols = fixedColors(k)
    for i in range(0,k):
        tmp = x[idx == i]
        ax.scatter(tmp[:,0],tmp[:,1],color=cols[i],marker='o')

def moveTrace(ax,history):
    m,n = history[0].shape
    cols = fixedColors(m)
    hislen = len(history)
    traces = np.zeros((hislen,n))
    for i in range(0,m):
        for j  in range(0,hislen):
            traces[j] = history[j][i]
        ax.plot(traces[:,0],traces[:,1],color=cols[i],marker='x')

def centInit(x,k,dim):
    """
    it selects the first K examples based on the random permutation of the indices. 
    This allows the examples to be selected at random without the risk of selecting 
    the same example twice
    """
    a = range(0,np.size(x,0))
    # the np.random.sample can't do this job
    a = random.sample(a,k)
    return x[a]

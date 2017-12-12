import numpy as np
from kmeans import *

if __name__ == '__main__':
    fig, ax = plt.subplots()
    ex7data2 = np.load('ex7data2.npz')
    x = ex7data2['X']
    m,n = x.shape
    k = 3
    """
    centroids = np.zeros((k,n)) 
    centroids[0] = [3, 3]
    centroids[1] = [6, 2]
    centroids[2] = [8, 5]
    """
    centroids = centInit(x,k,2)
    #idx =  closestCentroids(x,centroids)
    #print computeCentroids(x,idx,k)
    idx,history =  kmeans(x,centroids,tol=1e-5)
    plotResult(ax,x,idx)
    moveTrace(ax,history)
    plt.show()


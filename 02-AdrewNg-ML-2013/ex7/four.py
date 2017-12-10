import numpy as np
from matplotlib import pyplot as plt     
from matplotlib import animation     
from pca import *

def randomShow(x,n,picsize):  
    """
    select n pictures from x randomly
    """
    col = 10
    row = int(np.ceil(n/10.0))
    fig,ax = plt.subplots(row,col)
    #rnd = np.random.randint(0,np.size(x,0),n)
    rnd = np.arange(0,n)
    xl = x[rnd].reshape((n,picsize[0],picsize[1]))
    for i in range(0,row):
        for j in range(0,col):
            # if not transpose xl[i*col+j], the picture will show horizontally
            ax[i,j].imshow(xl[i*col+j].T,cmap=plt.cm.gray)
            ax[i,j].set_xticks([])
            ax[i,j].set_yticks([])
    plt.axis('off')

if __name__ == '__main__':
    k = 100
    x = np.load('ex7faces.npz')['X']
    norm,mean,std = normalize(x)
    u,s = pca(norm)
    z = project(norm,u,k)
    # recovery from projected data
    xr = recovery(z,u,k)
    
    randomShow(x,50,(32,32))
    randomShow(norm,50,(32,32))
    randomShow(z,50,(10,10))
    randomShow(xr,50,(32,32))
    plt.show()

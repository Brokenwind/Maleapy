import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from kmeans import *

if __name__ == '__main__':
    k = 16
    img = Image.open('bird_small.png').getdata()
    # picture size
    leng,wid = img.size
    # number of pixels
    m = leng * wid
    data = np.array(img,dtype=np.float64)/255
    # original picture pixel density
    orgpic = data.copy()
    # compressed picture pixel density
    compic = data.copy()
    # initialize centroids randomly
    centroids= centInit(data,k,3)
    idx,history =  kmeans(data,centroids)
    # get the final converged points
    centroids = np.array(history[len(history)-1])

    # compress the picture, replace the sampe labeled pixels with its centroid
    for i in range(0,k):
        compic[idx == i] = centroids[i]

    fig, ax = plt.subplots(2)
    # The value for each component of MxNx3 and MxNx4 float arrays should be
    # in the range 0.0 to 1.0; MxN float arrays may be normalised.
    ax[0].imshow(orgpic.reshape((leng,wid,3)))
    ax[1].imshow(compic.reshape((leng,wid,3)))
    plt.show()
    
    


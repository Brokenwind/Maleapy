#!/usr/bin/python

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def imageToMatrix(filename):
    im = Image.open(filename)
    # im.show()  
    width,height = im.size
    # L model means the picture will be convert to grayscale picture. And the following formula is used to convert RGB color picture to grayscal picture.
    #  L = R * 299/1000 + G * 587/1000+ B * 114/1000
    im = im.convert("L") 
    data = im.getdata()
    data = np.matrix(data)
    new_data = np.reshape(data,(width,height))
    return new_data

def matrixToImage(data):
     return Image.fromarray(data.astype(np.uint8))

def showImage(data):
    """To show a grayscale picture with it's relative matrix
    """
    # use matplotlib to show image directly
    plt.imshow(data, cmap=plt.cm.gray, interpolation='nearest')
    plt.show()
    """
    img = matrixToImage(data)
    img.show()    
    """

def  randomShow():
    # x.txt stores 5000 training examples of handwritten digits, each digits has 400(20*20) pixels data
    data = np.loadtxt('x.txt')
    num = 25
    row = col = int(np.sqrt(num))
    fig,ax = plt.subplots(row,col)
    # random select num digits
    rnd = np.random.randint(1,5000,num)
    sels = data[rnd,:]
    for i in np.arange(0,num):
        r = i / row
        c = i % row
        pic = sels[i,:].reshape((20,20))
        ax[r,c].imshow(pic, cmap=plt.cm.gray, interpolation='nearest')
        # do not show any ticks
        ax[r,c].set_xticks([])
        ax[r,c].set_yticks([])
    # adjust the space between pictues to 0
    plt.subplots_adjust(wspace=0,hspace=0)
    plt.show()
    
if __name__ == '__main__':
    """
    filename = 'lena.jpg'
    data = imageToMatrix(filename)
    showImage(data)
    #img.save('lena_1.bmp')
    """
    randomShow()

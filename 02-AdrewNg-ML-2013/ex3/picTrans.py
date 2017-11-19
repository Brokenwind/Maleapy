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
    img = matrixToImage(data)
    img.show()    
    # use matplotlib to show image directly
    plt.imshow(data, cmap=plt.cm.gray, interpolation='nearest')
    plt.show()

if __name__ == '__main__':
    filename = 'lena.jpg'
    data = imageToMatrix(filename)
    showImage(data)
    #img.save('lena_1.bmp')

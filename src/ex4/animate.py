import numpy as np
from matplotlib import pyplot as plt     
from matplotlib import animation     
from one import *
from two import *

"""It will randomly select some test data and calculate its predicted result, then show original hand-written picture ,correct answer and predicted result at same time.
"""

# animation function.  this is called sequentially
def animate(i):  
    """
    i: means the corrent ith frame you will show
    """
    # show original hand-written digit
    pic = selx[i,:].reshape((20,20))
    pic0.set_data(pic)

    pic = selm[i,:].reshape((5,5))
    pic00.set_data(pic)
    
    # show correct digit
    res = int(sely[i])
    filename = "../ex3/digit/"+str(res)+".png"
    img = plt.imread(filename)  
    pic1.set_data(img)

    # show predicted digit
    res = int(selp[i])
    res = int(res)
    filename = "../ex3/digit/"+str(res)+".png"
    img = plt.imread(filename)  
    pic2.set_data(img)

    return pic0,pic00,pic1,pic2

if __name__ == '__main__':
    num=100
    path='../ex3/'
    # load and rearrange data
    x = np.loadtxt(path+'x.txt')
    #x = np.hstack((np.ones((np.size(x,0),1)),x))
    y = np.loadtxt(path+'y.txt')
    units = [400,25,10]
    theta1 = np.loadtxt(path+'theta1.txt')
    theta2 = np.loadtxt(path+'theta2.txt')

    fig,ax = plt.subplots(2,2)
    # random select num digits
    rnd = np.random.randint(1,5000,num)
    pred = predict([theta1,theta2],x,y,units)
    sely = y[rnd]
    selx = x[rnd,:]
    selp = pred[rnd]
    one = np.ones((num,1))
    tmp = np.hstack((one,selx))
    selm = sigmoid(tmp.dot(theta1.T))
    # original picture
    pic0 = ax[0,0].imshow(selx[0,:].reshape((20,20)),cmap=plt.cm.gray)
    # the hidden level of neural network
    pic00 = ax[1,0].imshow(selm[0,:].reshape((5,5)),cmap=plt.cm.gray)
    # the correct result
    pic1 = ax[0,1].imshow([[0]])
    # the predict result
    pic2 = ax[1,1].imshow([[0]])
    ax[0,0].set_title('Hand-Written')
    ax[1,0].set_title('Hidden')
    ax[0,1].set_title('Corret')
    ax[1,1].set_title('Predicted')
    plt.axis('off')
    anim1=animation.FuncAnimation(fig, animate, frames=num, interval=2000)
    plt.show()
    

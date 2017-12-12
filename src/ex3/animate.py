import numpy as np
from matplotlib import pyplot as plt     
from matplotlib import animation     
from one import *

"""It will randomly select some test data and calculate its predicted result, then show original hand-written picture ,correct answer and predicted result at same time.
"""

# animation function.  this is called sequentially
def animate(i):  
    """
    i: means the corrent ith frame you will show
    """
    # show original hand-written digit
    pic = sels[i,:].reshape((20,20))
    pic0.set_data(pic)

    # show correct digit
    res = int(sely[i])
    filename = "./digit/"+str(res)+".png"
    img = plt.imread(filename)  
    pic1.set_data(img)

    # show predicted digit
    res = predict(sels[i,:],theta)
    res = int(res)
    filename = "./digit/"+str(res)+".png"
    img = plt.imread(filename)  
    pic2.set_data(img)

    return pic0,pic1,pic2

if __name__ == '__main__':
    num=100
    x = np.loadtxt('x.txt')
    x = np.hstack((np.ones((np.size(x,0),1)),x))
    y = np.loadtxt('y.txt')
    theta = oneVsAll(x,y)

    fig,ax = plt.subplots(1,3)
    # random select num digits
    rnd = np.random.randint(1,5000,num)
    sely = y[rnd]
    sels = x[rnd,:]
    sels = np.delete(sels,0,axis=1)
    pic0 = ax[0].imshow(sels[0,:].reshape((20,20)),cmap=plt.cm.gray)
    pic1 = ax[1].imshow([[0]])
    pic2 = ax[2].imshow([[0]])
    ax[0].set_title('Hand-Written')
    ax[1].set_title('Corret')
    ax[2].set_title('Predicted')
    plt.axis('off')
    anim1=animation.FuncAnimation(fig, animate, frames=num, interval=3000)
    plt.show()
    

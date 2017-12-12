import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from recomsys import *

if __name__ == '__main__':
    ex8_movies = np.load('ex8_movies.npz')
    R = ex8_movies['R']
    Y = ex8_movies['Y']
    moviesParams = np.load('ex8_movieParams.npz')
    X = moviesParams['X']
    Theta = moviesParams['Theta']
    num_movie,num_user = R.shape
    num_feature = np.size(Theta,1)
    params = np.hstack((X.flatten(),Theta.flatten()))

    # use smaller dataset to test costFunc and gradient
    num_movie = 5
    num_user = 4
    num_feature = 3
    r = R[0:num_movie,0:num_user]
    y = Y[0:num_movie,0:num_user]
    x = X[0:num_movie,0:num_feature]
    theta = Theta[0:num_user,0:num_feature]
    params = np.hstack((x.flatten(),theta.flatten()))
    print ('Cost at loaded parameters: %f' % (costFunc(params,y,r,num_movie,num_user,num_feature,lamda=1.5)))
    print ('gradient at loaded parameters:')
    print gradient(params,y,r,num_movie,num_user,num_feature,lamda=1.5)
    checkGradient()

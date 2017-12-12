import re
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from recomsys import *

def loadMovies():
    data = open('movie_ids.txt')
    movies = []
    for line in data:
        movies.append(re.split('^\d+\s',line)[1])
    return movies

def getMyRatings(movies):
    my_ratings = np.zeros(len(movies))
    # Check the file movie_idx.txt for id of each movie in our dataset
    # For example, Toy Story (1995) has ID 1, so to rate it "4", you can set
    my_ratings[1] = 4
    # Or suppose did not enjoy Silence of the Lambs (1991), you can set
    my_ratings[98] = 2
    # We have selected a few movies we liked / did not like and the ratings we
    # gave are as follows:
    my_ratings[7] = 3
    my_ratings[12]= 5
    my_ratings[54]= 4
    my_ratings[64]= 5
    my_ratings[66]= 3
    my_ratings[69] = 5
    my_ratings[183] = 4
    my_ratings[226] = 5
    my_ratings[355]= 5

    print('New user ratings:')
    for i in range(0,len(my_ratings)):
        if my_ratings[i] > 0 :
            print( 'Rated %d for %s' % ( my_ratings[i], movies[i] ) )
    return my_ratings.reshape((len(my_ratings),1))
    

if __name__ == '__main__':
    ex8_movies = np.load('ex8_movies.npz')
    R = ex8_movies['R']
    Y = ex8_movies['Y']
    # Add our own ratings to the data matrix
    movies = loadMovies()
    my_ratings = getMyRatings(movies)    
    judge = (my_ratings > 0) + 0
    R = np.hstack((judge,R))
    Y = np.hstack((my_ratings,Y))
    num_movie,num_user = R.shape
    # 10 kinds of movie
    num_feature = 10
    #Set Initial Parameters (Theta, X)
    X = np.random.random((num_movie,num_feature))
    Theta = np.random.random((num_user,num_feature))
    params = np.hstack((X.flatten(),Theta.flatten()))

    Ynorm,Ymean = nomalizeRating(Y,R)
    status,params = optimSolve(params,Ynorm,R,num_movie,num_user,num_feature,lamda=10.0)
    print ('The learned features are:')
    X = params[0:num_movie*num_feature].reshape((num_movie,num_feature))
    print X
    print ('The learned parameters are:')
    Theta = params[num_movie*num_feature:].reshape((num_user,num_feature))
    print Theta

    #After training the model, you can now make recommendations by computing the predictions matrix.
    p = X.dot( Theta.T )
    my_pred = p[:,0] + Ymean
    for i in range(0,len(movies)):
        if my_ratings[i] > 0 :
            print( 'Original rating: %d, Predicted rating: %.1f , for %s' % ( my_ratings[i], my_pred[i],movies[i] ) )
    print Ymean

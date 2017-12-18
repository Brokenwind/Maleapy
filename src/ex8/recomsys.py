import numpy as np
import scipy.optimize as op

"""
This module provides common funtions about recommender systems  to calculate  cost function and its derivative(gradient), 
and find optimal value with gradient descend and other faster method.
"""
def costFunc(params,y,r,num_movie,num_user,num_feature,lamda=0.0):
    """
    params: it includes the x and theta parameters.
    y: the remark matrix
    r: the matrix indicating whether users remark the movies
    num_movie: number of movie
    num_user: number of user
    num_feature: number of feature
    lamda: regularization parameter
    """
    x = params[0:num_movie*num_feature].reshape((num_movie,num_feature))
    theta = params[num_movie*num_feature:].reshape((num_user,num_feature))
    # y is a matrix with size num_movie * num_user
    err = (x.dot(theta.T) - y)[r==1]
    J = 0.5*(np.sum(err ** 2 ))
    regSum = np.sum(theta * theta) + np.sum(x * x)
    J += 0.5*lamda*regSum
    return J

def gradient(params,y,r,num_movie,num_user,num_feature,lamda=0.0):
        """
    params: it includes the x and theta parameters.
    y: the remark matrix
    r: the matrix indicating whether users remark the movies
    num_movie: number of movie
    num_user: number of user
    num_feature: number of feature
    lamda: regularization parameter
    """
    lamda *= 1.0
    x = params[0:num_movie*num_feature].reshape((num_movie,num_feature))
    theta = params[num_movie*num_feature:].reshape((num_user,num_feature))
    err = (x.dot(theta.T) - y)*r
    gradtheta = (err.T).dot(x)
    gradtheta += lamda*theta
    gradx = err.dot(theta)
    gradx += lamda*x
    return np.hstack((gradx.flatten(),gradtheta.flatten()))

def numericalGradient(params,y,r,num_movie,num_user,num_feature,lamda=0.0):
    check = 1e-4
    m = params.size
    numegrad = np.zeros(m)
    for i in range(0,m):
        tmp = params[i]
        params[i] = tmp + check
        up = costFunc(params,y,r,num_movie,num_user,num_feature,lamda)
        params[i] = tmp - check
        down = costFunc(params,y,r,num_movie,num_user,num_feature,lamda)
        numegrad[i] = (up - down)/(2.0*check)
        # restore the params
        params[i] = tmp

    return numegrad

def norm(x,p=2):
    return np.sum(np.abs(x) ** p) ** (1.0/p)

def checkGradient():
    x = np.random.random((4,3))
    theta = np.random.random((5,3))
    y = x.dot(theta.T)
    y[np.random.random(y.shape) > 0.7] = 0
    r = np.ones(y.shape)
    r[y == 0] = 0

    lamda = 1.0
    num_movie,num_user = r.shape
    num_feature = np.size(theta,1)
    params = np.hstack((x.flatten(),theta.flatten()))

    fgra1 = gradient(params,y,r,num_movie,num_user,num_feature,lamda)
    fgra2 = numericalGradient(params,y,r,num_movie,num_user,num_feature,lamda)
    print ('The bellow two columns you get should be very similar:')
    print np.vstack((fgra1,fgra2)).T
    diff = norm(fgra1 - fgra2)/norm(fgra1 + fgra2)
    print ('Relative Difference: %e' % (diff))
    return norm(fgra1 - fgra2)/norm(fgra1 + fgra2)


def gradientSolve(params,y,r,num_movie,num_user,num_feature,alpha,lamda=0.0):
    """
    Use gradient decent method to compute parameters
    """
    lamda *= 1.0
    alpha *= 1.0
    jhistory = []
    while iters > 0:
        deri = gradient(params,y,r,num_movie,num_user,num_feature,lamda)
        params -= alpha*deri
        jhistory.append(costFunc(params,y,r,num_movie,num_user,num_feature,lamda))
        iters -= 1
    return theta,jhistory

def optimSolve(params,y,r,num_movie,num_user,num_feature,lamda=0.0):
    lamda *= 1.0
    res = op.minimize(fun = costFunc, x0 = params,args = (y,r,num_movie,num_user,num_feature,lamda),method = 'TNC',jac = gradient)
    # use BFGS minimization algorithm. but it will not work well because costFunc may return a NaN value
    #print op.fmin_bfgs(costFunc, initial_theta, args = (x,y), fprime=gradient)
    return res.success,res.x

def nomalizeRating(y,r):
    """
    normalized Y so that each movie has a rating of 0 on average, and returns the mean rating in Ymean.
    """
    m,n = y.shape
    ymean = np.zeros(m)
    ynorm = np.zeros(y.shape)
    sel = r == 1
    for i in range(0,m):
        idx = r[i] == 1
        ymean[i] = np.mean(y[i,idx])
        ynorm[i,idx] = y[i,idx] - ymean[i]
    return ynorm,ymean
    

"""
if __name__ == '__main__':
    data = np.loadtxt('ex2data1.txt',delimiter=',')
    x = np.delete(data,-1,axis=1)
    x = np.hstack((np.ones((x.shape[0],1)),x))
    m,n = x.shape
    y = data[:,-1].T
    initial_theta = np.zeros(n);
    print optimSolve(initial_theta,x,y)
    #print gradientSolve(initial_theta,x,y,0.0043,1000000)
"""

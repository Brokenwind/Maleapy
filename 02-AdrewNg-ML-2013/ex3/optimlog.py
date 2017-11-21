import numpy as np
import scipy.optimize as op

"""
This module provides common funtions about logistic regression to calculate  cost function and its derivative(gradient), 
and find optimal value with gradient descend and other faster method.
"""

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z));

def gradient(theta,x,y,reg=False,lamda=0.0):
    """
    theta: estimated  value of unknown parameter
    x: the input test data
        number of rows means the how many input datas
        number of cols means the how many features
    y: the label of relative x
    reg: if it is True, means using regularized logistic. Default False
    return: the derivative of parameter
    """
    m , n = x.shape
    theta = theta.reshape((n,1));
    y0 = y.reshape((m,1))
    y1 = sigmoid(x.dot(theta));
    grad = ((x.T).dot(y1-y0))/m;
    if reg:
        # the theta0 is not involed
        #theta[0,0] = 0.0  # this will change global theta
        grad += (lamda/m)*theta
        grad[0,0] -= (lamda/m)*theta[0,0]
    return grad.flatten();

def costFunc(theta,x,y,reg=False,lamda=0.0):
    """
    theta: estimated  value of unknown parameter
    x: the input test data
        number of rows means the how many input datas
        number of cols means the how many features
    y: the label of relative x
    reg: if it is True, means using regularized logistic. Default False
    """
    m,n = x.shape; 
    theta = theta.reshape((n,1));
    y0 = y.reshape((m,1));
    # when the y0 = 1, the cost function is as following:
    pos = np.log(sigmoid(x.dot(theta)));
    # when the y0 = 0, the cost function is as following:
    neg = np.log(1-sigmoid(x.dot(theta)));
    all = y0 * pos + (1 - y0) * neg;
    J = -((np.sum(all))/m);
    if reg:
        # the theta0 is not involed
        theta = theta[1:]
        regSum = np.sum(theta * theta)
        J += lamda/(2.0*m)*regSum
    return J;

def gradientSolve(theta,x,y,alpha,iters,reg=False,lamda=0.0):
    """
    Use gradient decent method to compute parameters
    """
    lamda *= 1.0
    alpha *= 1.0
    while iters > 0:
        deri = gradient(theta,x,y,reg,lamda)
        theta = theta - alpha*deri
        iters -= 1
    return theta

def optimSolve(theta,x,y,reg=False,lamda=0.0):
    lamda *= 1.0
    theta = theta.flatten()
    res = op.minimize(fun = costFunc, x0 = theta,args = (x, y,reg,lamda),method = 'TNC',jac = gradient);
    # use BFGS minimization algorithm. but it will not work well because costFunc may return a NaN value
    #print op.fmin_bfgs(costFunc, initial_theta, args = (x,y), fprime=gradient)
    return res.x

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

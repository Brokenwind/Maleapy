import numpy as np
import scipy.optimize as op

"""
This module provides common funtions about linear regression to calculate  cost function and its derivative(gradient), 
and find optimal value with gradient descend and other faster method.
"""

def gradient(theta,x,y,reg=False,lamda=0.0):
    """
    theta: estimated  value of unknown parameter
    x: the input test data
        number of rows means the how many input datas
        number of cols means the how many features
    y: the label of relative x
    reg: if it is True, means using regularized linear. Default False
    return: the derivative of parameter
    """
    m , n = x.shape
    theta = theta.reshape((n,1))
    y0 = y.reshape((m,1))
    y1 = x.dot(theta)
    grad = ((x.T).dot(y1-y0))/m
    if reg:
        # the theta0 is not involed
        theta = np.vstack((0,theta[1:]))
        grad += (lamda/m)*theta
    return grad.flatten();

def costFunc(theta,x,y,reg=False,lamda=0.0):
    """
    theta: estimated  value of unknown parameter
    x: the input test data
        number of rows means the how many input datas
        number of cols means the how many features
    y: the label of relative x
    reg: if it is True, means using regularized linear. Default False
    """
    m,n = x.shape; 
    theta = theta.reshape((n,1))
    y = y.reshape((m,1))
    err = x.dot(theta) - y
    J = 0.5*(np.sum(err * err))/m
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
    jhistory = []
    while iters > 0:
        deri = gradient(theta,x,y,reg,lamda)
        theta = theta - alpha*deri
        jhistory.append(costFunc(theta,x,y,reg,lamda))
        iters -= 1
    return theta,jhistory

def optimSolve(theta,x,y,reg=False,lamda=0.0):
    lamda *= 1.0
    theta = theta.flatten()
    res = op.minimize(fun = costFunc, x0 = theta,args = (x, y,reg,lamda),method = 'TNC',jac = gradient);
    # use BFGS minimization algorithm. but it will not work well because costFunc may return a NaN value
    #print op.fmin_bfgs(costFunc, initial_theta, args = (x,y), fprime=gradient)
    return res.x

# pridict the the value for the given x, according to the calculated  theta
def predict(theta,x):
    """
    x: the input value
    theta:   calculated parameters matrix
    """
    x = np.array(x)
    m = theta.size
    theta = theta.reshape((m,1))
    if x.ndim == 1:
        if x.size < m:
            x = np.hstack((1,x))
        if x.size == m:
            x = x.reshape((1,m))
        else:
            x = x.reshape((x.size,1))
    # when x.ndim == 2
    n = np.size(x,1)
    if n + 1 == m:
        one = np.ones((np.size(x,0),1))
        x = np.hstack((one,x))

    n = np.size(x,1)
    if n != m:
        print('You input data is error!!')
        return None

    return x.dot(theta)


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

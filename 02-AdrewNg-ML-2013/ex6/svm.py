import sys
sys.path.append('..')
import numpy as np
import scipy.optimize as op

def svmtrain(X, Y, C, kernelFunction, tol, iters):
    """[model] = svmtrain(X, Y, C, kernelFunction, tol, iters)
    svmtrain trains an SVM classifier using a simplified version of the SMO algorithm. 
    PARAMETERS:
    X is the matrix of training examples.  Each row is a training example, and the jth column holds the  jth feature.
    Y is a column matrix containing 1 for positive examples and 0 for negative examples.
    C is the standard SVM regularization parameter.  
    tol is a tolerance value used for determining equality of floating point numbers. 
    iters controls the number of iterations over the dataset (without changes to alpha) before the algorithm quits.
    RETURN:
        model is the result parameters
    """
    
    pass

if __name__ == '__main__':
    pass

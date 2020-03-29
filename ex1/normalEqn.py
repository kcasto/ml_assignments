import numpy as np

def normalEqn(X, y):
    return np.linalg.inv(X.T.dot(X)).dot(X.T.dot(y))

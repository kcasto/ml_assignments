import numpy as np
from computeCost import computeCost

def gradientDescent(X, y, theta, alpha, num_iters):
    m = y.size
    J_history = np.zeros(num_iters)

    for i in range(num_iters):
        grad = X.T.dot(X.dot(theta)-y)/m
        theta -= alpha*grad
        J_history[i] = computeCost(X, y, theta)

    return theta, J_history

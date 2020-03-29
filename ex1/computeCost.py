def computeCost(X, y, theta):
    m = y.size
    diff = X.dot(theta) - y
    return diff.dot(diff)/(2*m)

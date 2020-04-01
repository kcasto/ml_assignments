import numpy as np
import matplotlib as plt
import scipy.optimize as op
import scipy.io as io

def main():
    data = io.loadmat('ex3data1.mat')
    X = data['X']
    y = data['y'].ravel()
    m, n = X.shape
    num_labels = 10
    lamda = 0.1
    all_theta = np.zeros((num_labels, n+1))
    X = np.concatenate((np.ones((m,1)), X), axis=1)

    init_theta = np.zeros(n+1)
    for i in range(1,num_labels+1):
        Result = op.minimize(fun = costFunction, x0 = init_theta,
                    args = (X, y==i, lamda), method = 'CG', jac = gradient,
                    options = {'maxiter': 50})
        theta = Result.x
        print "Cost at", i, ":", Result.fun
        all_theta[i % num_labels, :] = theta

    all_probs = sigmoid(X.dot(all_theta.T))
    pred = np.argmax(all_probs, axis=1)
    score = (pred == y % 10).mean() * 100
    print "Training set accuracy:", score

    weights = io.loadmat('ex3weights.mat')
    Theta1 = weights['Theta1']
    Theta2 = weights['Theta2']
    a2 = np.concatenate((np.ones((m,1)), sigmoid(X.dot(Theta1.T))), axis=1)
    a3 = sigmoid(a2.dot(Theta2.T))
    pred = np.argmax(a3, axis=1)
    score = (pred + 1 == y).mean() * 100
    print "Neural net accuracy:", score
    
def costFunction(theta, X, y, lamda):
    m = y.size
    J = (-y.dot(np.log(sigmoid(X.dot(theta))))
            - (1 - y).dot(np.log(1 - sigmoid(X.dot(theta))))
            + lamda/2*theta[1:].dot(theta[1:]))/m
    return J

def gradient(theta, X, y, lamda):
    m = y.size
    theta_no_first = np.append(0,theta[1:])
    grad = (X.T.dot(sigmoid(X.dot(theta)) - y) + lamda*theta_no_first)/m
    return grad

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


if __name__ == '__main__':
    main()

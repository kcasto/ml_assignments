import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op

def main():
    data = np.genfromtxt('ex2data1.txt', delimiter=',')
    X = data[:, 0:2]
    y = data[:, 2]
    m, n = X.shape
    print "Plotting data..."
    plt.ion()
    plotData(X, y)
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend(["Admitted", "Not admitted"])
    raw_input("Press enter to continue: ")

    X = np.concatenate((np.ones((m,1)), X), axis=1)
    init_theta = np.zeros(n+1)
    cost = costFunction(init_theta, X, y, 0)
    grad = gradient(init_theta, X, y, 0)
    print "Cost at initial theta (zeros):", cost
    print "Gradient at initial theta:", grad

    Result = op.minimize(fun = costFunction, x0 = init_theta,
                        args = (X, y, 0), method = 'TNC', jac = gradient)
    theta = Result.x
    cost = Result.fun
    print "Optimal theta:", theta
    print "Cost at theta:", cost

    plotDecisionBoundary(theta, X, y)

    prob = sigmoid(np.array([1, 45, 85]).dot(theta))
    print "For a student with scores 45 and 85, admission has probability", prob

    p = predict(theta, X)
    score = (p == (y==1)).mean() * 100
    print "Train accuracy: ", score


    data = np.genfromtxt('ex2data2.txt', delimiter=',')
    X = data[:, 0:2]
    y = data[:, 2]
    plt.figure()
    plotData(X, y)
    plt.xlabel("Microchip test 1")
    plt.ylabel("Microchip test 2")
    raw_input("Press enter: ")

    X = mapFeature(X[:,0], X[:,1], 6)
    init_theta = np.zeros(X.shape[1])
    lamda = 1

    Result = op.minimize(fun = costFunction, x0 = init_theta,
                        args = (X, y, lamda), method = 'TNC', jac = gradient)
    theta = Result.x
    cost = Result.fun
    plotDecisionBoundary2(theta, X, y)
    plt.xlabel("Microchip test 1")
    plt.ylabel("Microchip test 2")
    plt.legend(["y = 1", "y = 0", "Decision boundary"])
    plt.show()

    p = predict(theta, X)
    score = (p == (y==1)).mean() * 100
    print "Train accuracy: ", score
    raw_input("Press enter: ")

def plotData(X, y):
    pos = y==1
    neg = y==0
    plt.plot(X[pos, 0], X[pos, 1], 'k+', markersize=10)
    plt.plot(X[neg, 0], X[neg, 1], 'ko', markersize=10, fillstyle='none')
    plt.draw()

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

def plotDecisionBoundary(theta, X, y):
    plot_x = np.array([X[:,1].min()-2, X[:,1].max()-2])
    plot_y = -(theta[1]*plot_x + theta[0])/theta[2]
    plt.plot(plot_x, plot_y)
    plt.axis([30, 100, 30, 100])
    plt.show()
    raw_input("Press enter: ")

def plotDecisionBoundary2(theta, X, y):
    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)
    U, V = np.meshgrid(u, v)
    Z = np.zeros(U.shape)
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            Z[i,j] = mapFeature(np.array([U[i,j]]),
                        np.array([V[i,j]]), 6).dot(theta)
    plt.contour(U, V, Z, [0], linewidths=2)

def predict(theta, X):
    return sigmoid(X.dot(theta)) >= 0.5

def mapFeature(X1, X2, degree):
    out = np.zeros((X1.shape[0],(degree+1)*(degree+2)/2))
    k = 0
    for i in range(degree+1):
        for j in range(i+1):
            out[:,k] = (X1**(i-j) * X2**j)
            k += 1
    return out

if __name__ == '__main__':
    main()

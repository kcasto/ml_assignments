import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op

def main():
    data = np.genfromtxt('ex2data1.txt', delimiter=',')
    X = data[:, 0:2]
    y = data[:, 2]
    m, n = X.shape
    logReg1 = LogisticRegression(X, y)
    print "Plotting data..."
    plt.ion()
    logReg1.plotData()
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend(["Admitted", "Not admitted"])
    raw_input("Press enter to continue: ")

    init_theta = logReg1.initialTheta()
    cost = logReg1.costFunction(init_theta, 0)
    grad = logReg1.gradient(init_theta, 0)
    print "Cost at initial theta (zeros):", cost
    print "Gradient at initial theta:", grad

    Result = op.minimize(fun = logReg1.costFunction, x0 = init_theta,
                        args = (0), method = 'TNC', jac = logReg1.gradient)
    theta = Result.x
    cost = Result.fun
    print "Optimal theta:", theta
    print "Cost at theta:", cost

    logReg1.plotDecisionBoundary(theta)
    raw_input("Press enter: ")

    prob = sigmoid(np.array([1, 45, 85]).dot(theta))
    print "For a student with scores 45 and 85, admission has probability", prob
    print "Train accuracy: ", logReg1.score(theta)

    data = np.genfromtxt('ex2data2.txt', delimiter=',')
    X = data[:, 0:2]
    y = data[:, 2]
    logReg2 = LogisticRegression(X, y, featureDegree=6)
    plt.figure()
    logReg2.plotData()
    plt.xlabel("Microchip test 1")
    plt.ylabel("Microchip test 2")
    raw_input("Press enter: ")

    init_theta = logReg2.initialTheta()
    lamda = 1

    Result = op.minimize(fun = logReg2.costFunction, x0 = init_theta,
                        args = (lamda), method = 'TNC', jac = logReg2.gradient)
    theta = Result.x
    logReg2.plotDecisionBoundary(theta)
    plt.xlabel("Microchip test 1")
    plt.ylabel("Microchip test 2")
    plt.legend(["y = 1", "y = 0", "Decision boundary"])
    plt.show()

    print "Train accuracy: ", logReg2.score(theta)
    raw_input("Press enter: ")

class LogisticRegression:
    def __init__(self, X, y, featureDegree=1):
        self.rawX = X
        self.y = y
        self.featureDegree = featureDegree
        self.X = mapFeature(X[:,0], X[:,1], featureDegree)
        self.m, self.n = self.X.shape

    def initialTheta(self):
        return np.zeros(self.n)

    def costFunction(self, theta, lamda):
        return (-self.y.dot(np.log(sigmoid(self.X.dot(theta))))
            - (1 - self.y).dot(np.log(1 - sigmoid(self.X.dot(theta))))
            + lamda/2*theta[1:].dot(theta[1:]))/self.m

    def gradient(self, theta, lamda):
        theta_no_first = np.append(0,theta[1:])
        return (self.X.T.dot(sigmoid(self.X.dot(theta)) - self.y)
                + lamda*theta_no_first)/self.m

    def plotData(self):
        pos = (self.y == 1)
        neg = (self.y == 0)
        plt.plot(self.rawX[pos, 0], self.rawX[pos, 1], 'k+', markersize=10)
        plt.plot(self.rawX[neg, 0], self.rawX[neg, 1], 'ko', markersize=10,
                fillstyle='none')
        plt.draw()

    def plotDecisionBoundary(self, theta):
        if self.featureDegree == 1:
            plot_x = np.array([self.X[:,1].min()-2, self.X[:,1].max()-2])
            plot_y = -(theta[1]*plot_x + theta[0])/theta[2]
            plt.plot(plot_x, plot_y)
            plt.axis([30, 100, 30, 100])
            plt.show()
        else:
            u = np.linspace(-1, 1.5, 50)
            v = np.linspace(-1, 1.5, 50)
            U, V = np.meshgrid(u, v)
            Z = np.zeros(U.shape)
            for i in range(U.shape[0]):
                for j in range(U.shape[1]):
                    Z[i,j] = mapFeature(np.array([U[i,j]]),
                            np.array([V[i,j]]), self.featureDegree).dot(theta)
            plt.contour(U, V, Z, [0], linewidths=2)

    def score(self, theta):
        p = sigmoid(self.X.dot(theta)) >= 0.5
        return (p == (self.y == 1)).mean()*100


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

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

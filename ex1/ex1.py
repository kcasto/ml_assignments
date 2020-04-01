import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def main():
    print "Plotting data..."
    data = np.genfromtxt('ex1data1.txt', delimiter=',')
    X = data[:,0]
    y = data[:,1]

    #Plotting data
    plt.ion()
    plt.plot(X,y,'rx',markersize=10)
    plt.title("Training Data")
    plt.xlabel("Population")
    plt.ylabel("Profit")
    plt.draw()
    plt.pause(0.001)
    raw_input("Press enter to continue: ")

    linReg1 = LinearRegression(X, y)
    
    print("Running gradient descent...")

    iterations = 9000
    alpha = 0.01

    print "Initial cost: ", linReg1.computeCost() #theta default is 0 vector

    theta,J_history = linReg1.gradientDescent(alpha, iterations)

    print "Theta found by gradient descent: ", theta

    theta2 = linReg1.normalEqn()
    print "Theta found by normal equation: ", theta2

    plt.plot(linReg1.X[:,1], linReg1.X.dot(theta), '-')
    plt.legend(["Training data", "Linear regression"])
    plt.draw()
    plt.pause(0.001)
    raw_input("Press enter to continue: ")

    predict1 = np.array([1, 3.5]).dot(theta)*10000
    print "For population of 35,000, we predict a profit of", predict1
    predict2 = np.array([1, 7]).dot(theta)*10000
    print "For population of 70,000, we predict a profit of", predict2

    print "Visualizing J(theta)..."
    
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    t0 = np.linspace(-10,10,100)
    t1 = np.linspace(-1,4,100)
    T0, T1 = np.meshgrid(t0,t1)
    J = np.zeros(T0.shape)
    for i in range(J.shape[0]):
        for j in range(J.shape[1]):
            J[i,j] = linReg1.computeCost([T0[i,j],T1[i,j]])
    ax.plot_surface(T0,T1,J)
    ax.set_xlabel("Theta_0")
    ax.set_ylabel("Theta_1")
    ax.set_zlabel("Cost")
    plt.show()
    raw_input("Contour plot of cost...")
    fig, ax = plt.subplots(1,1)
    ax.contourf(T0, T1, J, np.logspace(-2, 3, 20))
    plt.show()

    raw_input("Multilinear regression: ")
    plt.close('all')
    
    data2 = np.genfromtxt('ex1data2.txt',delimiter=',')
    X2 = data2[:, 0:2]
    y2 = data2[:, 2]
    linReg2 = LinearRegression(X2, y2, normalize=True)

    alpha = 0.02
    num_iters = 4000
    theta, J_history = linReg2.gradientDescent(alpha,num_iters)

    plt.plot(range(J_history.size), J_history, '-b', linewidth=2)
    plt.show()
    print "Theta compute from gradient descent: ", theta
    t_denorm = linReg2.deNormalize(theta)
    print "De-normalized theta computed from gradient descent: ", t_denorm
    price = t_denorm.dot([1, 1650, 3])
    print "Predicted price on 1650 sqft 3br house w gradient descent: ", price
    print "Solving with normal equations..."
    t_exact = linReg2.normalEqn(denormalize=True)
    print "Theta computed from normal equation: ", t_exact
    price2 = t_exact.dot([1, 1650, 3])
    print "Price on 1650 sqft 3br house from normal equation: ", price2

class LinearRegression:
    def __init__(self, X, y, normalize=False):
        if X.ndim == 1:
            X = X[:,None]
        self.X = X
        self.y = y
        self.m = y.size
        if normalize:
            self.mu = self.X.mean(axis=0)
            self.sigma = self.X.std(axis=0,ddof=1)
            self.denormX = np.concatenate((np.ones((self.m,1)), self.X), axis=1)
            self.X = (self.X-self.mu)/self.sigma
        self.normalized = normalize
        self.X = np.concatenate((np.ones((self.m,1)), self.X), axis=1)

    def computeCost(self, theta=None):
        if theta is None:
            theta = np.zeros(self.X.shape[1])
        diff = self.X.dot(theta) - self.y
        return diff.dot(diff)/(2*self.m)

    def gradientDescent(self, alpha, num_iters, theta=None):
        if theta is None:
            theta = np.zeros(self.X.shape[1])
        J_history = np.zeros(num_iters)

        for i in range(num_iters):
            grad = self.X.T.dot(self.X.dot(theta)-self.y)/self.m
            theta -= alpha*grad
            J_history[i] = self.computeCost(theta)

        return theta, J_history

    def normalEqn(self, denormalize=False):
        if self.normalized and denormalize:
            X = self.denormX
        else:
            X = self.X
        return np.linalg.inv(X.T.dot(X)).dot(X.T.dot(self.y))

    def deNormalize(self, theta):
        #Given theta used with a normalized X,
        #returns the equivalent theta for a de-normalized X
        return np.array([theta[0] - theta[1]*self.mu[0]/self.sigma[0]
                - theta[2]*self.mu[1]/self.sigma[1],
                theta[1]/self.sigma[0], theta[2]/self.sigma[1]])

if __name__ == '__main__':
    main()

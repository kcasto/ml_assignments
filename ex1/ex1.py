import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def main():
    print "Plotting data..."
    data = np.genfromtxt('ex1data1.txt', delimiter=',')
    X = data[:,0]
    y = data[:,1]
    m = data.shape[0]

    #Plotting data
    plt.ion()
    plt.plot(X,y,'rx',markersize=10)
    plt.title("Training Data")
    plt.xlabel("Population")
    plt.ylabel("Profit")
    plt.draw()
    plt.pause(0.001)
    raw_input("Press enter to continue: ")
    

    print("Running gradient descent...")
    X = np.concatenate((np.ones((m,1)), X[:,None]), axis=1)
    theta = np.zeros(2)

    iterations = 9000
    alpha = 0.01

    print "Initial cost: ", computeCost(X, y, theta)

    theta,J_history = gradientDescent(X, y, theta, alpha, iterations)

    print "Theta found by gradient descent: ", theta

    theta2 = normalEqn(X,y)
    print "Theta found by normal equation: ", theta2

    plt.plot(X[:,1], X.dot(theta), '-')
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
            J[i,j] = computeCost(X,y,[T0[i,j],T1[i,j]])
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
    
    data = np.genfromtxt('ex1data2.txt',delimiter=',')
    X = data[:, 0:2]
    y = data[:, 2]
    m = data.shape[0]
    X, mu, sigma = featureNormalize(X)
    X = np.concatenate((np.ones((m,1)), X), axis=1)

    alpha = 0.02
    num_iters = 4000
    theta = np.zeros(3)
    theta, J_history = gradientDescent(X,y,theta,alpha,num_iters)

    plt.plot(range(J_history.size), J_history, '-b', linewidth=2)
    plt.show()
    print "Theta compute from gradient descent: ", theta
    t_denorm = np.array([theta[0] - theta[1]*mu[0]/sigma[0]
        - theta[2]*mu[1]/sigma[1], theta[1]/sigma[0], theta[2]/sigma[1]])
    print "De-normalized theta computed from gradient descent: ", t_denorm
    price = t_denorm.dot([1, 1650, 3])
    print "Predicted price on 1650 sqft 3br house w gradient descent: ", price
    print "Solving with normal equations..."
    X = data[:, 0:2]
    X = np.concatenate((np.ones((m,1)), X), axis=1)
    t_exact = normalEqn(X,y)
    print "Theta computed from normal equation: ", t_exact
    price2 = t_exact.dot([1, 1650, 3])
    print "Price on 1650 sqft 3br house from normal equation: ", price2


def computeCost(X, y, theta):
    m = y.size
    diff = X.dot(theta) - y
    return diff.dot(diff)/(2*m)

def gradientDescent(X, y, theta, alpha, num_iters):
    m = y.size
    J_history = np.zeros(num_iters)

    for i in range(num_iters):
        grad = X.T.dot(X.dot(theta)-y)/m
        theta -= alpha*grad
        J_history[i] = computeCost(X, y, theta)

    return theta, J_history

def normalEqn(X, y):
    return np.linalg.inv(X.T.dot(X)).dot(X.T.dot(y))

def featureNormalize(X):
    mu, sigma = X.mean(axis=0), X.std(axis=0,ddof=1)
    return (X-mu)/sigma, mu, sigma

if __name__ == '__main__':
    main()

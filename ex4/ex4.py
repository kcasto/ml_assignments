import numpy as np
import matplotlib as plt
import scipy.optimize as op
import scipy.io as io

def main():
    data = io.loadmat('ex4data1.mat')
    X = data['X']
    y = data['y']
    m = X.shape[0]
    in_size = 400
    hid_size = 25
    out_size = 10
    lamda = 1.3
    X = np.concatenate((np.ones((m,1)), X), axis=1)
    weights = io.loadmat('ex4weights.mat')
    Theta1 = weights['Theta1']
    Theta2 = weights['Theta2']
    thetas = np.concatenate((Theta1.ravel(), Theta2.ravel()))
    #can only optimize over 1d arrays, so have to flatten thetas
    print costFunction(thetas, in_size, hid_size, out_size, X, y, 0)
    print costFunction(thetas, in_size, hid_size, out_size, X, y, 1)

    init_theta1 = initializeWeights(in_size, hid_size)
    init_theta2 = initializeWeights(hid_size, out_size)
    init_thetas = np.concatenate((init_theta1.ravel(), init_theta2.ravel()))

    checkGradients(1)
    print costFunction(thetas, in_size, hid_size, out_size, X, y, 3)

    lamda = 1.3
    Result = op.minimize(fun = costFunction, x0 = init_thetas,
                args = (in_size, hid_size, out_size, X, y, lamda),
                method = 'CG', jac = gradient, options = {'maxiter': 200})
    thetas = Result.x
    Theta1 = thetas[:hid_size*(in_size+1)].reshape(hid_size, in_size+1)
    Theta2 = thetas[hid_size*(in_size+1):].reshape(out_size, hid_size+1)
    a2 = np.concatenate((np.ones((m,1)), sigmoid(X.dot(Theta1.T))), axis=1)
    a3 = sigmoid(a2.dot(Theta2.T))
    pred = np.argmax(a3, axis=1)
    score = (pred + 1 == y.ravel()).mean() * 100
    print "Neural net accuracy:", score
    

def costFunction(thetas, in_size, hid_size, out_size, X, y, lamda):
    Theta1 = thetas[:hid_size*(in_size+1)].reshape(hid_size, in_size+1)
    Theta2 = thetas[hid_size*(in_size+1):].reshape(out_size, hid_size+1)
    m = X.shape[0]

    z2 = X.dot(Theta1.T)
    a2 = np.concatenate((np.ones((m,1)), sigmoid(z2)), axis=1)
    a3 = sigmoid(a2.dot(Theta2.T))

    y_mat = range(1,out_size+1) == y
    return (-y_mat.ravel().dot(np.log(a3.ravel())) - (1 - y_mat.ravel()).dot(
        np.log(1 - a3.ravel()))
        + lamda/2.*(Theta1[:,1:].ravel().dot(Theta1[:,1:].ravel()) + 
            Theta2[:,1:].ravel().dot(Theta2[:,1:].ravel())))/m

def gradient(thetas, in_size, hid_size, out_size, X, y, lamda):
    Theta1 = thetas[:hid_size*(in_size+1)].reshape(hid_size, in_size+1)
    Theta2 = thetas[hid_size*(in_size+1):].reshape(out_size, hid_size+1)
    m = X.shape[0]

    z2 = X.dot(Theta1.T)
    a2 = np.concatenate((np.ones((m,1)), sigmoid(z2)), axis=1)
    a3 = sigmoid(a2.dot(Theta2.T))

    y_mat = range(1,out_size+1) == y
    T1_no_first = np.concatenate((np.zeros((hid_size,1)), Theta1[:,1:]),axis=1)
    T2_no_first = np.concatenate((np.zeros((out_size,1)), Theta2[:,1:]),axis=1)

    delta3 = a3 - y_mat
    delta2 = delta3.dot(Theta2)[:,1:] * sigmoidGradient(z2)

    Theta1_grad = (delta2.T.dot(X) + lamda*T1_no_first)/m
    Theta2_grad = (delta3.T.dot(a2) + lamda*T2_no_first)/m

    return np.concatenate((Theta1_grad.ravel(), Theta2_grad.ravel()))

def checkGradients(lamda):
    in_size = 3
    hid_size = 5
    out_size = 3
    m = 5

    T1 = np.sin(range(hid_size*(in_size+1))).reshape(hid_size,in_size+1)/10
    T2 = np.sin(range(out_size*(hid_size+1))).reshape(out_size,hid_size+1)/10
    X = np.sin(range(m*in_size)).reshape(m, in_size)/10
    X = np.concatenate((np.ones((m,1)), X), axis=1)
    y = np.mod(range(m), out_size)[:,None]

    thetas = np.concatenate((T1.ravel(), T2.ravel()))
    grad = gradient(thetas, in_size, hid_size, out_size, X, y, lamda)
    fun = lambda p: costFunction(p, in_size, hid_size, out_size, X, y, lamda)
    numgrad = numericalGradient(fun, thetas)
    print "Numerical gradient:", numgrad
    print "Analytical gradient:", grad


def numericalGradient(J, theta):
    numgrad = np.zeros(theta.size)
    perturb = np.zeros(theta.size)
    e = 1e-4
    for p in range(theta.size):
        perturb[p] = e
        numgrad[p] = (J(theta + perturb) - J(theta - perturb))/(2*e)
        perturb[p] = 0
    return numgrad


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoidGradient(z):
    return sigmoid(z) * (1 - sigmoid(z))

def initializeWeights(l_in, l_out):
    eps = 0.12
    return (2*np.random.rand(l_out, l_in + 1)-1)*eps

if __name__ == '__main__':
    main()

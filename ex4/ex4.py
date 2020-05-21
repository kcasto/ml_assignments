import numpy as np
import matplotlib as plt
import scipy.optimize as op
import scipy.io as io

def main():
    data = io.loadmat('ex4data1.mat')
    X = data['X']
    y = data['y']
    in_size = 400
    hid_size = 25
    out_size = 10
    neuralnet = NeuralNet(in_size, hid_size, out_size, X, y)
    lamda = 1.3
    weights = io.loadmat('ex4weights.mat')
    Theta1 = weights['Theta1']
    Theta2 = weights['Theta2']
    thetas = np.concatenate((Theta1.ravel(), Theta2.ravel()))
    #can only optimize over 1d arrays, so have to flatten thetas
    print neuralnet.costFunction(thetas, 0)
    print neuralnet.costFunction(thetas, 1)
    init_thetas = neuralnet.randInitialThetas()
    checkGradients(1)
    print neuralnet.costFunction(thetas, 3)

    lamda = 1.3
    Result = op.minimize(fun = neuralnet.costFunction, x0 = init_thetas,
                args = (lamda), method = 'CG',
                jac = neuralnet.gradient, options = {'maxiter': 200})
    thetas = Result.x
    print "Neural net accuracy:", neuralnet.score(thetas)

class NeuralNet:
    def __init__(self, in_size, hid_size, out_size, X, y):
        self.m = X.shape[0]
        self.X = padLeftColumn(X, 1)
        self.y = y
        self.y_mat = range(1,out_size+1) == y
        self.in_size = in_size
        self.hid_size = hid_size
        self.out_size = out_size

    def costFunction(self, thetas, lamda):
        Theta1, Theta2 = self.reshapeThetas(thetas)
        z2 = self.X.dot(Theta1.T)
        a2 = padLeftColumn(sigmoid(z2), 1)
        a3 = sigmoid(a2.dot(Theta2.T))

        return (-self.y_mat.ravel().dot(np.log(a3.ravel())) - 
                (1 - self.y_mat.ravel()).dot(np.log(1 - a3.ravel()))
                + lamda/2.*(Theta1[:,1:].ravel().dot(Theta1[:,1:].ravel())
                + Theta2[:,1:].ravel().dot(Theta2[:,1:].ravel())))/self.m

    def gradient(self, thetas, lamda):
        Theta1, Theta2 = self.reshapeThetas(thetas)
        z2 = self.X.dot(Theta1.T)
        a2 = padLeftColumn(sigmoid(z2), 1)
        a3 = sigmoid(a2.dot(Theta2.T))

        T1_no_first = padLeftColumn(Theta1[:,1:], 0)
        T2_no_first = padLeftColumn(Theta2[:,1:], 0)

        delta3 = a3 - self.y_mat
        delta2 = delta3.dot(Theta2)[:,1:] * sigmoidGradient(z2)

        Theta1_grad = (delta2.T.dot(self.X) + lamda*T1_no_first)/self.m
        Theta2_grad = (delta3.T.dot(a2) + lamda*T2_no_first)/self.m

        return np.concatenate((Theta1_grad.ravel(), Theta2_grad.ravel()))
    
    def reshapeThetas(self, thetas):
        Theta1 = thetas[:self.hid_size*(self.in_size+1)].reshape(
                    self.hid_size, self.in_size+1)
        Theta2 = thetas[self.hid_size*(self.in_size+1):].reshape(
                    self.out_size, self.hid_size+1)
        return Theta1, Theta2

    def randInitialThetas(self):
        eps = 0.12
        init_t1 = (2*np.random.rand(self.hid_size, self.in_size+1) - 1)*eps
        init_t2 = (2*np.random.rand(self.out_size, self.hid_size+1) - 1)*eps
        return np.concatenate((init_t1.ravel(), init_t2.ravel()))

    def score(self, thetas):
        Theta1, Theta2 = self.reshapeThetas(thetas)
        a2 = padLeftColumn(sigmoid(self.X.dot(Theta1.T)), 1)
        a3 = sigmoid(a2.dot(Theta2.T))
        pred = np.argmax(a3, axis=1)
        return (pred + 1 == self.y.ravel()).mean() * 100


def padLeftColumn(mat, fill_value):
    return np.concatenate((np.full((mat.shape[0],1), fill_value), mat), axis=1)

def checkGradients(lamda):
    in_size = 3
    hid_size = 5
    out_size = 3
    m = 5

    T1 = np.sin(range(hid_size*(in_size+1))).reshape(hid_size,in_size+1)/10
    T2 = np.sin(range(out_size*(hid_size+1))).reshape(out_size,hid_size+1)/10
    X = np.sin(range(m*in_size)).reshape(m, in_size)/10
    y = np.mod(range(m), out_size)[:,None]

    testNN = NeuralNet(in_size, hid_size, out_size, X, y)

    thetas = np.concatenate((T1.ravel(), T2.ravel()))
    grad = testNN.gradient(thetas, lamda)
    fun = lambda p: testNN.costFunction(p, lamda)
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

if __name__ == '__main__':
    main()

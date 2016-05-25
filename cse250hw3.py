import numpy as np
import random
import math
import matplotlib.pyplot as plt

### project funtion
# project to d-dimension cube
def cubeProjection(point):
    for i in xrange(len(point)):
        if np.sign(point[i]) >= 0: point[i] = 1
        else: point[i] = -1
    return point

# project to d-dimension unit ball
def ballProjection(point):
    return point / np.linalg.norm(point)
    
### generate samples
#   input:  n sample size
#           sigma: variance of Gaussian
#           scenario =1: project to cube, =0: project to ball
def pointGen(n, sigma, scenario = 0):
    y = np.zeros((n, 1))
    x = np.zeros((n, 4))
    for i in xrange(n):
        if random.random() >= 0.5:
            y[i] = 1
            mu = np.array([0.25, 0.25, 0.25, 0.25])
            px = np.random.multivariate_normal(mu, np.eye(4) * sigma * sigma)
        else: 
            y[i] = -1
            mu = -np.array([0.25, 0.25, 0.25, 0.25])
            px = np.random.multivariate_normal(mu, np.eye(4) * sigma * sigma)
        if scenario == 1: x[i,:] = cubeProjection(px)
        elif scenario == 2: x[i,:] = ballProjection(px)
    return x, y        

### Gradient of logistic loss function
def deltaLoss(w, x, y):
    x = np.reshape(x, (1, 4))
    one = np.ones((1, 1))
    x = np.concatenate((x, one), axis = 1)
    return - y * x / (1 + math.exp(y * np.dot(x, w)))

### logistic loss function
def logisticLoss(w, x, y):
    x = np.reshape(x, (1, 4))
    one = np.ones((1, 1))
    x = np.concatenate((x, one), axis = 1)
    return math.log(1 + math.exp(- y * np.dot(x, w)))

### binary classification error
def binaryLoss(w, x, y):
    x = np.reshape(x, (1, 4))
    one = np.ones((1, 1))
    x = np.concatenate((x, one), axis = 1)
    if np.sign(np.dot(x, w)) != y: return 1
    else: return 0

### calculate average error on test data samples
#   l=0: binary classification error
#   l=1: logisitic loss
def check(x, y, w, l):
    err = 0
    for i in xrange(len(y)):
        if l == 0: err += binaryLoss(w, x[i,:], y[i])
        elif l == 1: err += logisticLoss(w, x[i,:], y[i])
    return err * 1.0 / len(y)
        
### Stochastic Gradient Descent Algorithm
#   input:
#       scenario = 1: cube, =2: ball
#       loss = 0: binary error, =1: logistic error
def sgd(x, y, scenario, loss):
    w = np.zeros((len(y) + 1, x.shape[1] + 1))
    wt = np.zeros((x.shape[1] + 1, 1))
    for i in xrange(len(y)):
        g = deltaLoss(wt, x[i], y[i])
        if scenario == 1: wt = cubeProjection(np.subtract(wt, g.T / math.sqrt(i + 1)))
        elif scenario == 2: wt = ballProjection(np.subtract(wt, g.T / math.sqrt(i + 1) / 2))
        w[i + 1,:] = wt.T[:]
    return np.sum(w, axis = 0) / len(w)

### experiment function
#   scenario: =1: cube, =2: ball
#   loss: =0: binary error, =1: logistic error
def test(n, sigma, scenario, loss):
    testx, testy = pointGen(400, sigma, scenario)
    # run sgd for 20 times, store error of each time
    err = np.zeros((20, 1))
    for i in xrange(20):
        x, y = pointGen(n, sigma, scenario)
        w = sgd(x, y, scenario, loss)
        err[i] = check(testx, testy, w, loss)
    return np.mean(err), np.std(err)

def experiment(loss):
    sigma = np.array([0.05, 0.25])
    n = np.array([50, 100, 500, 1000])
    plt.figure()
    if loss == 1:
        plt.title("Logistic loss result on different scenarios")
        axes = plt.gca()
        axes.set_xlim([25, 1025])
        axes.set_ylim([0, 1])
    elif loss == 0:
        plt.title("Binary Classfication error on different scenarios")
        axes = plt.gca()
        axes.set_xlim([25,1025])
        axes.set_ylim([-0.1,0.6])
    for i in xrange(len(sigma)):
        for scenario in xrange(1, 3):
            error = np.zeros((2, len(n)))
            for k in xrange(len(n)):
                error[0][k], error[1][k] = test(n[k], sigma[i], scenario, loss)
            print error
            print "Scenario: {}, sigma: {}".format(scenario, sigma[i])
            print "Average: {}".format(sum(error[0])/len(error[0]))
            if i == 0 and scenario == 1:
                plt.errorbar(n, error[0], error[1], ls="--", label = "sigma = 0.05,cube")
            elif i == 1 and scenario == 1:
                plt.errorbar(n, error[0], error[1], ls="-", label = "sigma = 0.25,cube")
            elif i == 0 and scenario == 2:
                plt.errorbar(n, error[0], error[1], ls="-.", label = "sigma = 0.05,ball")
            elif i == 1 and scenario == 2:
                plt.errorbar(n, error[0], error[1], ls=":", label = "sigma = 0.25,ball")
    plt.legend()
    plt.show()

experiment(1)
experiment(0)

import numpy as np
import random
import math

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
def test(sigma, scenario, loss):
    testx, testy = pointGen(400, sigma, scenario)
    n = np.array([50, 100, 500, 1000])
    ex = np.zeros((len(n), 1))
    std = np.zeros((len(n), 1))
    for i in xrange(len(n)):
        # run sgd for 20 times, store error of each time
        err = np.zeros((20, 1))
        for j in xrange(20):
            x, y = pointGen(n[i], sigma, scenario)
            w = sgd(x, y, scenario, loss)
            err[j] = check(testx, testy, w, loss)
        ex[i] = np.mean(err)
        std[i] = np.std(err)
    return ex, std

f = open('data.txt', 'w')
sigma = np.array([0.05, 0.25])
for i in xrange(len(sigma)):
    for scenario in xrange(1, 3):
        f.write("%d" % scenario)
        f.write("\n")
        f.write("%f" % sigma[i])
        f.write("\n")
        lossEx, lossStd = test(sigma[i], scenario, 1)
        lossEx.tofile(f, sep = " ")
        f.write("\n")
        lossStd.tofile(f, sep = " ")
        f.write("\n")
        binEx, binStd = test(sigma[i], scenario, 0)
        binEx.tofile(f, sep = " ")
        f.write("\n")
        binStd.tofile(f, sep = " ")
        f.write("\n")
f.close()

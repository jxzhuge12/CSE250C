import numpy as np
import random
import math
import matplotlib.pyplot as plt

### functions from existing packages
#   sign function from numpy
#   input:  int 
#   output: -1 if input < 0
#           0 if input == 0
#           1 if input > 0
#
#   norm function from numpy.linalg
#   input:  array
#   output: euclidean distance of input
#
#   multivariate_normal function from numpy.random
#   using Gaussian distribution represented by mean and cov to generate a n-dimension point 
#   input:  mean
#           cov
#   output: n-dimension array
#
#   mean function from numpy
#   input:  array
#   output: mean of the array
#
#   std function from numpy
#   input:  array
#   output: standard deviation of the array
#
#   random function from random
#   generate a random number from [0, 1)
#   output: a number from [0, 1)

### project funtion
#   project to d-dimension cube
def cubeProjection(point):
    # for each entry of the point, project it into [-1, 1] if the point locates outside [-1, 1]
    for i in xrange(len(point)):
        if point[i] > 1: point[i] = 1
        elif point[i] < -1: point[i] = -1
    return point

#   project to d-dimension unit ball
def ballProjection(point):
    # divide each entry of the point by the euclidean distance of the point if the point locates outside the unit ball
    if np.linalg.norm(point) > 1:
        point = point / np.linalg.norm(point)
    return point 
    
### generate samples
#   input:  n: sample size
#           sigma: variance of Gaussian function
#           scenario == 1: project to cube, == 2: project to ball
def pointGen(n, sigma, scenario = 0):
    y = np.zeros((n, 1))
    x = np.zeros((n, 4))
    for i in xrange(n):
        # generate a random number from [0, 1)
        # if it is smaller than 0.5, generate a point labeled -1
        # else generate a point labeled 1
        if random.random() >= 0.5:
            y[i] = 1
            # set mean as (0.25, 0.25, 0.25, 0.25)
            mu = np.array([0.25, 0.25, 0.25, 0.25])
            # generate point based on mean and cov
            px = np.random.multivariate_normal(mu, np.eye(4) * sigma * sigma)
        else: 
            y[i] = -1
            # set mean as (-0.25, -0.25, -0.25, -0.25)
            mu = np.array([-0.25, -0.25, -0.25, -0.25])
            # generate point based on mean and cov
            px = np.random.multivariate_normal(mu, np.eye(4) * sigma * sigma)
        # make projection based on different scenario
        if scenario == 1: x[i,:] = cubeProjection(px)
        elif scenario == 2: x[i,:] = ballProjection(px)
    # return n samples x and their labels y
    return x, y        

### gradient of logistic loss function
def deltaLoss(w, x, y):
    # append x with 1
    x = np.reshape(x, (1, 4))
    one = np.ones((1, 1))
    x = np.concatenate((x, one), axis = 1)
    # return gradient of logistic loss function based on w, x and y
    return - y * x / (1 + math.exp(y * np.dot(x, w)))

### logistic loss function
def logisticLoss(w, x, y):
    # append x with 1
    x = np.reshape(x, (1, 4))
    one = np.ones((1, 1))
    x = np.concatenate((x, one), axis = 1)
    # return logisitic loss based on w, x and y
    return math.log(1 + math.exp(- y * np.dot(x, w)))

### binary classification error
def binaryLoss(w, x, y):
    # append x with 1
    x = np.reshape(x, (1, 4))
    one = np.ones((1, 1))
    x = np.concatenate((x, one), axis = 1)
    # return binary loss based on w, x and y
    if np.sign(np.dot(x, w)) != y: return 1
    else: return 0

### calculate average error on test data samples
#   x, y: test points
#   l == 0: binary classification error
#   l == 1: logisitic loss
def check(x, y, w, l):
    err = 0
    # for each test point, check whether it is classified correctly
    if l == 0: 
        for i in xrange(len(y)): err += binaryLoss(w, x[i,:], y[i])
    elif l == 1: 
        for i in xrange(len(y)): err += logisticLoss(w, x[i,:], y[i])
    # return error rate
    return err * 1.0 / len(y)
        
### Stochastic Gradient Descent Algorithm
#   input:
#       scenario == 1: cube, == 2: ball
#       for cube and ball have different learning rate
def sgd(x, y, scenario):
    # store all of w
    w = np.zeros((len(y) + 1, x.shape[1] + 1))
    # current w
    wt = np.zeros((x.shape[1] + 1, 1))
    # calculate w based on each training point and store it
    for i in xrange(len(y)):
        g = deltaLoss(wt, x[i], y[i])
        if scenario == 1: wt = cubeProjection(np.subtract(wt, g.T / math.sqrt(i + 1)))
        elif scenario == 2: wt = ballProjection(np.subtract(wt, g.T / math.sqrt(i + 1) / math.sqrt(2)))
        w[i + 1,:] = wt.T[:]
    return np.sum(w, axis = 0) / len(w)


### test performance of sgd on test set 
#   input:
#       n: training sample size
#       sigma: variance
#       scenario == 1: cube, == 2: ball
#   output: 
#       bianry error mean, std
#       logistic loss mean, std
def test(testx, testy, n, sigma, scenario):    
    # run sgd for 20 times, store test error into err
    err0 = np.zeros((20, 1))
    err1 = np.zeros((20, 1))
    for i in xrange(20):
        # generate n points
        x, y = pointGen(n, sigma, scenario)
        # calculate w using sgd method
        w = sgd(x, y, scenario)
        # check error rate on test points using w
        err0[i] = check(testx, testy, w, 0)
        err1[i] = check(testx, testy, w, 1)
    # return mean and standard deviation of error rates
    return np.mean(err0), np.std(err0), np.mean(err1), np.std(err1)
    


### do experiment 
#   We will do following experiments:
#   Scenario ==1, cube:
#       case 1: sigma = 0.05, logistic loss(loss == 0)
#       case 2: sigma = 0.05, binary error(loss == 1)
#       case 3: sigma = 0.25, logistic loss(loss == 0)
#       case 4: sigma = 0.25, binary error(loss == 1)
#   Scenario == 2, ball:
#       case 1: sigma = 0.05, logistic loss(loss == 0)
#       case 2: sigma = 0.05, binary error(loss == 1)
#       case 3: sigma = 0.25, logistic loss(loss == 0)
#       case 4: sigma = 0.25, binary error(loss == 1)
#   Return: errorAvg, errorStd in the following cases
def experiment():
    # for each cases, run SGD algorithm on different size
    n = np.array([50, 100, 500, 1000])
    sigma = np.array([0.05, 0.25])
    errorAvg = np.zeros((8, 4))
    errorStd = np.zeros((8, 4))
    idx = 0
    # do experiments on two scenarios: cube and ball:
    for scenario in xrange(1, 3):
        # for each case, do experiments on different sigma:
        for i in xrange(len(sigma)):
            # generate test sample points
            testx, testy = pointGen(400, sigma[i], scenario)
            # run SGD algorithm on different training size
            for k in xrange(len(n)):
                errorAvg[idx][k],errorStd[idx][k],errorAvg[idx+1][k],errorStd[idx+1][k]=test(testx, testy, n[k], sigma[i], scenario)
            idx += 2
    return errorAvg, errorStd

errorAvg, errorStd = experiment()
print errorAvg
print errorStd

### draw and save images
n = np.array([50, 100, 500, 1000])
plt.figure(1)
plt.subplot(221)
axes = plt.gca()
axes.set_xlim([25, 1025])
plt.errorbar(n,errorAvg[0], errorStd[0], ls='--', label = "0.05,binary")
plt.legend()
plt.subplot(222)
axes = plt.gca()
axes.set_xlim([25, 1025])
plt.errorbar(n,errorAvg[1], errorStd[1], ls='--', label = "0.05,logistic")
plt.legend()
plt.subplot(223)
axes = plt.gca()
axes.set_xlim([25, 1025])
plt.errorbar(n,errorAvg[2], errorStd[2],ls='--', label = "0.25,binary")
plt.legend()
plt.subplot(224)
axes = plt.gca()
axes.set_xlim([25, 1025])
plt.errorbar(n,errorAvg[3], errorStd[3],ls='--', label = "0.25,logistic")
plt.legend()
plt.savefig("result_scenario1.png")

plt.figure(2)
plt.subplot(221)
axes = plt.gca()
axes.set_xlim([25, 1025])
plt.errorbar(n,errorAvg[4], errorStd[4], ls='--', label = "0.05,binary")
plt.legend()
plt.subplot(222)
axes = plt.gca()
axes.set_xlim([25, 1025])
plt.errorbar(n,errorAvg[5], errorStd[5], ls='--', label = "0.05,logistic")
plt.legend()
plt.subplot(223)
axes = plt.gca()
axes.set_xlim([25, 1025])
plt.errorbar(n,errorAvg[6], errorStd[6],ls='--', label = "0.25,binary")
plt.legend()
plt.subplot(224)
axes = plt.gca()
axes.set_xlim([25, 1025])
plt.errorbar(n,errorAvg[7], errorStd[7],ls='--', label = "0.25,logistic")
plt.legend()
plt.savefig("result_scenario2.png")

plt.figure(3)
axes = plt.gca()
axes.set_xlim([25, 1025])
plt.errorbar(n, errorAvg[1], errorStd[1], ls='--', label="cube, 0.05, logistic")
plt.errorbar(n, errorAvg[3], errorStd[3], ls='--', label="cube, 0.25, logistic")
plt.errorbar(n, errorAvg[5], errorStd[5], ls='--', label="ball, 0.05, logistic")
plt.errorbar(n, errorAvg[7], errorStd[7], ls='--', label="ball, 0.25, logistic")
plt.legend()
plt.title("SGD result comparision under logistic loss")
plt.savefig("comparision_logistic")

plt.figure(4)
axes = plt.gca()
axes.set_xlim([25, 1025])
plt.errorbar(n, errorAvg[0], errorStd[0], ls='--', label="cube, 0.05, binary")
plt.errorbar(n, errorAvg[2], errorStd[2], ls='--', label="cube, 0.25, binary")
plt.errorbar(n, errorAvg[4], errorStd[4], ls='--', label="ball, 0.05, binary")
plt.errorbar(n, errorAvg[6], errorStd[6], ls='--', label="ball, 0.25, binary")
plt.legend()
plt.title("SGD result comparision under binary loss")
plt.savefig("comparision_binary")


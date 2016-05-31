import numpy as np
from matplotlib import pyplot as pl
from scipy import stats

# Sphere can be represented by position of center and radius.
# c: center, represented by an array
# r: radius
class sphere:
    def __init__(self, c, r):
        self.c = c
        self.r = r

# Axis-aligned cube can be represented by position of center and edge length.
# c: center, represented by an array
# a: half of the edge length
class cube:
    def __init__(self, c, a):
        self.c = c
        self.a = np.abs(a)

# Given a set X and a point p, return the projection.
# p: the point, represented by an array
# X: the set, either a sphere or a cube
def projection(p, X):
    if type(X) == sphere:
        # np.linalg.norm: return the 2-norm of the vector p - X.c
        n = np.linalg.norm(p - X.c)
        if n <= X.r:
            return p
        else:
            return X.c + (p - X.c) / n * X.r
    if type(X) == cube:
        # clip: truncate each component of the vector p - X.c to [-X.a, X.a]
        return X.c + (p - X.c).clip(-X.a, X.a)

# Distribution described in the handout can be determined by two parameters:
# sigma and X. Given sigma and X, return a sample from the distribution.
# sigma: sigma used in the Gaussian distribution
# X: the set of all possible features, either a sphere or a cube
def oracle(sigma, X):
    d = len(X.c)
    # np.random.choice: choose a element from distribution of [0.5, 0.5] over [True, False]
    if np.random.choice([True, False], p = [0.5, 0.5]):
        # np.array: convert the list to an 1-d array
        # np.identity: produce a d-dimentional identity matrix
        return projection(np.random.multivariate_normal(\
            np.array([-0.25] * d), sigma ** 2 * np.identity(d)), X), -1
    else:
        # np.array: convert the list to an 1-d array
        # np.identity: produce a d-dimentional identity matrix
        return projection(np.random.multivariate_normal(\
            np.array([0.25] * d), sigma ** 2 * np.identity(d)), X), 1

# The gradient of the logistic loss function, represented by an array.
# w: w in the loss function
# x: x in the loss function
# y: y in the loss function
def grad(w, x, y):
    # np.exp: compute the exponent
    # np.dot: compute the inner product of w and x
    return -y * x / (1 + np.exp(y * np.dot(w, x)))

# Given an oracle, use SGD to train a classifier.
# M: the bound of C
# rho: loss function is rho-Lipschitz
# n: iterates for n times
# sigma: sigma for the oracle
# X: X for the oracle
# C: the set of all possible w, either a sphere or a cube
def sgd(M, rho, n, sigma, X, C):
    w = np.array(C.c)
    s = np.array(w)
    for t in range(1, n + 1):
        # np.sqrt: compute the square root of t
        alpha = M / rho / np.sqrt(t)
        x, y = oracle(sigma, X)
        # np.resize: add an extra dimention to x
        x = np.resize(x, len(w))
        x[-1] = 1
        w -= alpha * grad(w, x, y)
        w = projection(w, C)
        s += w
    return s / (n + 1)

# Given a classifier and test data, return the average risk.
# w: the classifier, represented by an array
# data: test data, list of tuples of features and labels
def avgRisk(w, data):
    s = .0
    for x, y in data:
        # np.resize: add an extra dimention to x
        x = np.resize(x, len(w))
        x[-1] = 1
        # np.log: compute the logarithm
        # np.exp: compute the exponent
        # np.dot: compute the inner product of w and x
        s += np.log(1 + np.exp(-y * np.dot(w, x)))
    return s / len(data)

# Given a classifier and test data, return the average error.
# w: the classifier, represented by an array
# data: test data, list of tuples of features and labels
def avgError(w, data):
    s = .0
    for x, y in data:
        # np.resize: add an extra dimention to x
        x = np.resize(x, len(w))
        x[-1] = 1
        # np.sign: 1 for positive numbers, -1 for negative numbers, 0 for 0
        # np.dot: compute the inner product of w and x
        s += 0 if y * (np.sign(np.dot(w, x)) + 0.1) > 0 else 1
    return s / len(data)

# set the parameters
sigmaList = [0.05, 0.25]
nList = [50, 100, 500, 1000]
N = 400
nTrain = 30
scenario = ["cube", "sphere"]
result = {}
# start the experiment

# TODO: run 20 times for each parameter set

for s in scenario:
    # set parameters related to the scenario
    # np.array: convert the list to an 1-d array
    # np.sqrt: compute the square root
    if s == "cube":
        X = cube(np.array([.0] * 4), 1)
        C = cube(np.array([.0] * 5), 1)
        M = np.sqrt(5)
        rho = np.sqrt(5)
    else:
        X = sphere(np.array([.0] * 4), 1)
        C = sphere(np.array([.0] * 5), 1)
        M = 1
        rho = np.sqrt(2)
    result[s] = {}
    for sigma in sigmaList:
        result[s][sigma] = {"risk": {"mean": [], "std": []}, "error": {"mean": [], "std": []}}
        # generate test data
        test = [oracle(sigma, X) for i in range(N)]
        # test the outputs of SGD on the test data
        for n in nList:
            # collect risks and errors
            risk = []
            error = []
            for i in range(nTrain):
                w = sgd(M, rho, n, sigma, X, C)
                risk.append(avgRisk(w, test))
                error.append(avgError(w, test))
            # get means and standard deviations
            # np.mean: return the mean value of data
            # np.std: return the standard error of data
            result[s][sigma]["risk"]["mean"].append(np.mean(risk))
            result[s][sigma]["risk"]["std"].append(np.std(risk))
            result[s][sigma]["error"]["mean"].append(np.mean(error))
            result[s][sigma]["error"]["std"].append(np.std(error))

# print and plot the results
for f in ["risk", "error"]:
    for s in scenario:
        for sigma in sigmaList:
            print(s, sigma, f)
            print(result[s][sigma][f]["mean"], result[s][sigma][f]["std"])
            if f == 'risk':
                y = []
                x = [np.log(nList[i])
                        for i in range(len(result[s][sigma][f]["mean"])-1)]
                for i in range(len(result[s][sigma][f]["mean"])-1):
                    # np.log: compute the logarithm
                    y.append(np.log(result[s][sigma][f]["mean"][i]
                                      - result[s][sigma][f]["mean"][-1]))
                # stats.linregress: linear regression of data (x, y).
                slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
                print('slope: ' + str(slope))
            pl.errorbar(nList,\
                        result[s][sigma][f]["mean"],\
                        yerr = result[s][sigma][f]["std"],\
                        label = s + '-' + str(sigma) + '-' + f)
    pl.ylabel(f)
    pl.xlabel('Size of Train Set')
    pl.legend()
    pl.show()
# theoretical risk in R^d: 0.4745 sigma = 0.05; 0.4833 sigma = 0.25
# theoretical error:  0 sigma = 0.05; 0.0228 sigma = 0.25

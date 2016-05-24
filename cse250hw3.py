import numpy as np
import random

def sce1Proj(point):
    for i in xrange(len(point)):
        if np.sign(point[i]) >= 0: point[i] = 1
        else: point[i] = -1
    return point

def sce2Proj(point):
    return point / np.linalg.norm(point)
    
def pointGen(n, sigma, scenario = 0):
    y = np.zeros((n, 1))
    x = np.zeros((n, 4))
    for i in xrange(n):
        py = -1
        if random.random() >= 0.5: py = 1
        y[i] = py
        mu = np.array([0.25, 0.25, 0.25, 0.25])
        if py == -1: mu = -mu
        px = np.random.multivariate_normal(mu, np.eye(4) * sigma * sigma)
        if scenario == 1: px = sce1Proj(px)
        elif scenario == 2: px = sce2Proj(px)
        x[i,:] = px[:]
    return x, y        

x, y = pointGen(10, 0.05, 1)
for i in xrange(len(x)):
    print"{}:{}".format(x[i], y[i])

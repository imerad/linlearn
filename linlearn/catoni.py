import numpy as np


catoni = np.frompyfunc(lambda x : np.log(1 + x + x*x/2) if x > 0 else -np.log(1 - x + x*x/2), 1, 1)
khi = np.frompyfunc(lambda x : 1 - 0.34 - 1/(1 + x*x), 1, 1)
gud = np.frompyfunc(lambda x : 2*np.arctan(np.exp(x)) - np.pi/2 if x < 10 else np.pi/2, 1, 1)


def estimate_sigma(x, eps=0.001):
    sigma = 1
    array = np.array(x)
    avg = array.mean()
    diff = 1
    khi0 = khi(0)
    niter = 0
    while diff > eps:
        tmp = sigma * np.sqrt(1 - (khi((array - avg)/sigma)).mean()/khi0)
        diff = np.abs(tmp - sigma)
        sigma = tmp
        niter += 1
    #print(niter)
    return sigma


def Holland_catoni_estimator(x, eps=0.001):
    s = estimate_sigma(x)
    m = 0
    diff = 1
    niter = 0
    while diff > eps:
        tmp = m + s * gud((x - m)/s).mean()
        diff = np.abs(tmp - m)
        m = tmp
        niter += 1
    #print(niter)
    return m

from scipy.optimize import brentq

def standard_catoni_estimator(x, eps=0.001):
    s = estimate_sigma(x)
    res = brentq(lambda u : s * catoni((x - u)/s).mean(), np.min(x), np.max(x))
    return res
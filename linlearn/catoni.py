import numpy as np
from numba import njit, vectorize, float64

#@njit
@vectorize([float64(float64)])
def catoni(x):
    return np.sign(x)*np.log(1 + np.sign(x)*x + x*x/2)# if x > 0 else -np.log(1 - x + x*x/2)

#@njit
@vectorize([float64(float64)])
def khi(x):
    return 1 - 0.34 - 1/(1 + x*x)#np.log(0.5 + x*x)#

#@njit
@vectorize([float64(float64)])
def gud(x):
    return 2*np.arctan(np.exp(x)) - np.pi/2 if x < 12 else np.pi/2


@njit
def estimate_sigma(x, eps=0.001):
    sigma = 1
    ar = x#np.array(x)
    avg = ar.mean()
    diff = 1
    khi0 = khi(0)
    niter = 0
    while diff > eps:
        tmp = sigma * np.sqrt(1 - (khi((ar - avg)/sigma)).mean()/khi0)
        diff = np.abs(tmp - sigma)
        sigma = tmp
        niter += 1
    #print(niter)
    return sigma

@njit
def Holland_catoni_estimator(x, eps=0.001):
    #if the array is constant, do not try to estimate scale
    # the following condition is supposed to reproduce np.allclose() behavior
    if (np.abs(x[0] - x) <= ((1e-8) + (1e-5) * np.abs(x[0]))).all():
        return x[0]

    s = estimate_sigma(x)*np.sqrt(len(x)/np.log(1/eps))
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
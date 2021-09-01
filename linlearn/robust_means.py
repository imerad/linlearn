import numpy as np
from numba import njit, vectorize, float64
from math import ceil
from numpy.random import permutation

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
def Holland_catoni_estimator(x, eps=0.001, max_it=30):
    #if the array is constant, do not try to estimate scale
    # the following condition is supposed to reproduce np.allclose() behavior
    if (np.abs(x[0] - x) <= ((1e-8) + (1e-5) * np.abs(x[0]))).all():
        return x[0]
    s = estimate_sigma(x)*np.sqrt(len(x)/np.log(1/eps))
    m = 0
    diff = 1
    niter = 0
    while diff > eps and niter < max_it:
        tmp = m + s * gud((x - m)/s).mean()
        diff = np.abs(tmp - m)
        m = tmp
        niter += 1
    #print(niter)
    return m

from scipy.optimize import brentq

def standard_catoni_estimator(x, eps=0.001):
    if (np.abs(x[0] - x) <= ((1e-8) + (1e-5) * np.abs(x[0]))).all():
        return x[0]
    s = estimate_sigma(x)
    res = brentq(lambda u : s * catoni((x - u)/s).mean(), np.min(x), np.max(x))
    return res


C5 = 0.01
C = lambda p: C5
print("WARNING : importing implementation of outlier robust gradient by (Prasad et al.) with arbitrary constant C(p)=%.2f"%C5)

def SSI(samples, subset_cardinality):
    """original name of this function is smallest_subset_interval"""
    if subset_cardinality < 2:
        raise ValueError("subset_cardinality must be at least 2")
    sorted_array = np.sort(samples)
    differences = sorted_array[subset_cardinality - 1:] - sorted_array[:-subset_cardinality + 1]
    argmin = np.argmin(differences)
    return sorted_array[argmin:argmin + subset_cardinality]


def alg2(X, eps, delta=0.001):
    # from Prasad et al. 2018
    X_tilde = alg4(X, eps, delta)

    n, p = X_tilde.shape

    if p == 1:
        return np.mean(X_tilde)

    S = np.cov(X.T)
    _, V = np.linalg.eigh(S)
    PW = V[:, :p // 2] @ V[:, :p // 2].T

    est1 = np.mean(X_tilde @ PW, axis=0, keepdims=True)

    QV = V[:, p // 2:]
    est2 = alg2(X_tilde @ QV, eps, delta)
    est2 = QV.dot(est2.T)
    est2 = est2.reshape((1, p))
    est = est1 + est2

    return est


def alg4(X, eps, delta=0.001):
    # from Prasad et al. 2018
    n, p = X.shape
    if p == 1:
        X_tilde = SSI(X.flatten(), max(2, ceil(n * (1 - eps - C5 * np.sqrt(np.log(n / delta) / n)) * (1 - eps))))
        return X_tilde[:, np.newaxis]

    a = np.array([alg2(X[:, i:i + 1], eps, delta / p) for i in range(p)])
    dists = ((X - a.reshape((1, p))) ** 2).sum(axis=1)
    asort = np.argsort(dists)
    X_tilde = X[asort[:ceil(n * (1 - eps - C(p) * np.sqrt(np.log(n / (p * delta)) * p / n)) * (1 - eps))], :]
    return X_tilde

@njit
def gmom_njit(xs, tol=1e-7):
    # from Vardi and Zhang 2000
    y = np.zeros(xs.shape[1])
    for i in range(xs.shape[0]):
        y += xs[i]
    y /= xs.shape[0]
    eps = 1e-10
    delta = 1
    niter = 0
    while delta > tol:
        xsy = xs - y
        dists = np.zeros(xsy.shape[0])
        for i in range(xsy.shape[1]):
            dists += xsy[:,i] ** 2 #np.linalg.norm(xsy, axis=1)
        dists = np.sqrt(dists)
        inv_dists = 1 / dists
        mask = dists < eps
        inv_dists[mask] = 0
        nb_too_close = (mask).sum()
        ry = np.sqrt(np.sum(np.dot(inv_dists, xsy)**2))#np.linalg.norm(np.dot(inv_dists, xsy))
        cst = nb_too_close / ry
        y_new = max(0, 1 - cst) * np.dot(inv_dists, xs)/np.sum(inv_dists) + min(1, cst) * y
        delta = np.sqrt(np.sum((y - y_new)**2))#np.linalg.norm(y - y_new)
        y = y_new
        niter += 1
    # print(niter)
    return y

def gmom(xs, tol=1e-7, max_it=30):
    # from Vardi and Zhang 2000
    y = np.average(xs, axis=0)
    eps = 1e-10
    delta = 1
    niter = 0
    while delta > tol and niter < max_it:
        xsy = xs - y
        dists = np.linalg.norm(xsy, axis=1)
        inv_dists = 1 / dists
        mask = dists < eps
        inv_dists[mask] = 0
        nb_too_close = (mask).sum()
        ry = np.linalg.norm(np.dot(inv_dists, xsy))
        cst = nb_too_close / ry
        y_new = max(0, 1 - cst) * np.average(xs, axis=0, weights=inv_dists) + min(1, cst) * y
        delta = np.linalg.norm(y - y_new)
        y = y_new
        niter += 1
    # print(niter)
    return y

@njit
def median_of_means(x, block_size):
    n = x.shape[0]
    n_blocks = int(n // block_size)
    last_block_size = n % block_size
    if last_block_size == 0:
        block_means = np.empty(n_blocks, dtype=x.dtype)
    else:
        block_means = np.empty(n_blocks + 1, dtype=x.dtype)

    # TODO:instanciates in the closure
    # This shuffle or the indexes to get different blocks each time
    permuted_indices = permutation(n)
    sum_block = 0.0
    n_block = 0
    for i in range(n):
        idx = permuted_indices[i]
        # Update current sum in the block
        sum_block += x[idx]
        if (i != 0) and ((i + 1) % block_size == 0):
            # It's the end of the block, save its mean
            block_means[n_block] = sum_block / block_size
            n_block += 1
            sum_block = 0.0

    if last_block_size != 0:
        block_means[n_blocks] = sum_block / last_block_size

    mom = np.median(block_means)
    return mom#, blocks_means



from numpy.random.mtrand import multivariate_normal
import pytest

from linlearn import MOMRegressor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

n_samples = 500
n_features = 5

rng = np.random.RandomState(42)
X = rng.multivariate_normal(np.zeros(n_features), np.eye(n_features), size=n_samples)

w_star = rng.multivariate_normal(np.zeros(n_features), np.eye(n_features)).reshape(n_features)

sigma = 0.5

y = X @ w_star + sigma * rng.rand(n_samples)
X_df = pd.DataFrame(X)
reg = MOMRegressor(tol=1e-17, max_iter=200, fit_intercept=False, strategy="mom")
#print(reg.estimate_grad_coord_variances(X, y, np.zeros(n_features)))

reg.fit(X_df, y)

#print(reg.predict(X))
print("finished in : %d iterations" % reg.n_iter_)
print(reg.mse(X, y))

print("w star is ")
print(w_star)
print("found optimum")
print(reg.get_params())
a, b = reg.get_params()
#print(reg.estimate_grad_coord_variances(X, y, np.concatenate((b, a.flatten()))))
#print(reg.estimate_grad_coord_variances(X, y, a.flatten()))
#w_star0 = np.zeros(n_features+1)
#w_star0[1:] = w_star
#print(reg.estimate_grad_coord_variances(X, y, w_star))


X_test = rng.multivariate_normal(np.zeros(n_features), np.eye(n_features), size=n_samples//5)

y_test = X_test @ w_star + sigma * rng.rand(n_samples//5)
print(reg.mse(X_test, y_test))

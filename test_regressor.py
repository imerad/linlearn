from numpy.random.mtrand import multivariate_normal
import pytest

from linlearn import Regressor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

n_samples = 200
n_features = 5

rng = np.random.RandomState(42)
X = rng.multivariate_normal(np.zeros(n_features), np.eye(n_features), size=n_samples)

w_star = rng.multivariate_normal(np.zeros(n_features), np.eye(n_features)).reshape(n_features)

sigma = 0.1

y = X @ w_star + sigma * rng.rand(n_samples)
X_df = pd.DataFrame(X)

reg = Regressor(tol=1e-17, max_iter=200).fit(X_df, y)

print(reg.predict(X))

print(reg.mse(X, y))

print("w star is ")
print(w_star)
print("found optimum")
print(reg.get_params())
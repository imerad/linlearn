import numpy as np
from numba import njit, prange
from collections import namedtuple
from .robust_means import Holland_catoni_estimator, median_of_means
from math import ceil

# TODO: definir ici une strategy

Strategy = namedtuple("Strategy", ["grad_coordinate", "n_samples_in_block", "name"])


# @njit
# def decision_function(X, fit_intercept, w, out):
#     if fit_intercept:
#         # TODO: use out= in dot and + z[0] at the same time with parallelize ?
#         out[:] = X.dot(w[1:,:]) + w[0,:]
#     else:
#         out[:] = X.dot(w)
#     return out

# @njit
# def decision_function_with_intercept(X, w, out):
#     # TODO: use out= in dot and + z[0] at the same time with parallelize ?
#     out[:] = X.dot(w[1]) + w[0]
#     return out
#
# @njit
# def decision_function_no_intercept(X, w, out):
#     # TODO: use out= in dot and + z[0] at the same time with parallelize ?
#     out[:] = X.dot(w)
#     return out


@njit
def decision_function(X, coef, intercept, out):
    # TODO: use out= in dot and + z[0] at the same time with parallelize ?

    out[:] = X.dot(coef) + intercept
    return out


# @njit
# def loss_sample(model, i, w):
#     z = inner_prod(model.X, model.fit_intercept, i, w)
#     if model.sample_weight.size == 0:
#         return model.value(model.y[i], z)
#     else:
#         return model.sample_weight[i] * model.value(model.y[i], z)
#
#
# @njit
# def loss_batch(model, w):
#     out = 0.0
#     # TODO: allocate this in fit
#     n_samples = model.y.shape[0]
#     Xw = np.empty(n_samples)
#     # TODO: inner_prods or for loop ? No need for Xw
#     Xw = inner_prods(model.X, model.fit_intercept, w, Xw)
#     if model.sample_weight.size == 0:
#         for i in range(n_samples):
#             out += model.loss(model.y[i], Xw[i]) / n_samples
#     else:
#         for i in range(n_samples):
#             out += model.sample_weight[i] * model.loss(model.y[i], Xw[i]) / n_samples
#     return out
#
#
# @njit
# def grad_sample_coef(model, i, w):
#     z = inner_prod(model.X, model.fit_intercept, i, w)
#     if model.sample_weight.size == 0:
#         return model.derivative(model.y[i], z)
#     else:
#         return model.sample_weight[i] * model.derivative(model.y[i], z)
#
#
# @njit
# def grad_sample(model, i, w, out):
#     c = grad_sample_coef(model, i, w)
#     if model.fit_intercept:
#         out[0] = c
#         out[1:] = c * model.X[i]
#     else:
#         out[:] = c * model.X[i]
#     return out


import numpy as np
from numba import njit


@njit
def trimmed_mean(x, delta=0.01):
    x.sort()
    n_excluded = ceil(5*np.log(8/delta))
    return np.mean(x[n_excluded:-n_excluded])

# x = np.arange(0, 12).astype("float64")
# x = np.random.permutation(x)
# x[4] = -1200
#
# n = x.shape[0]
# block_size = 2
# n_blocks = n // block_size
#
# if n % block_size == 0:
#     blocks_means = np.empty(n_blocks)
# else:
#     blocks_means = np.empty(n_blocks + 1)
#
# print(x)
# mom, blocks_means = median_of_means(x, block_size, blocks_means)
#
# print(blocks_means)
#
# print("mom: ", mom)

@njit
def grad_coordinate_per_sample(
    loss_derivative,
    j,
    X,
    y,
    inner_products,
    fit_intercept,
):
    n_samples = inner_products.shape[0]
    # TODO: parallel ?
    # TODO: sparse matrix ?

    place_holder = np.empty(n_samples, dtype=X.dtype)
    if fit_intercept:
        if j[0] == 0:
            # In this case it's the derivative w.r.t the intercept
            for idx in range(n_samples):
                place_holder[idx] = loss_derivative(y[idx], inner_products[idx], j[1])

        else:
            for idx in range(n_samples):
                place_holder[idx] = loss_derivative(y[idx], inner_products[idx], j[1]) * X[idx, j[0] - 1]

    else:
        # There is no intercept
        for idx in range(n_samples):
            place_holder[idx] = loss_derivative(y[idx], inner_products[idx], j[1]) * X[idx, j[0]]
    return place_holder

@njit
def grad_coordinate_erm(loss_derivative, j, X, y, inner_products, fit_intercept):
    """Computation of the derivative of the loss with respect to a coordinate using the
    empirical risk minimization (erm) stategy."""

    return np.mean(grad_coordinate_per_sample(loss_derivative, j, X, y, inner_products, fit_intercept))


def erm_strategy_factory(loss, fit_intercept, **kwargs):
    @njit
    def grad_coordinate(X, y, j, inner_products):
        return grad_coordinate_erm(
            loss.derivative, j, X, y, inner_products, fit_intercept
        )

    return Strategy(grad_coordinate=grad_coordinate, n_samples_in_block=None, name="erm")


# TODO: overlapping blocks in MOM ???


@njit
def grad_coordinate_mom(
    loss_derivative,
    j,
    X,
    y,
    inner_products,
    fit_intercept,
    n_samples_in_block,
):
    """Computation of the derivative of the loss with respect to a coordinate using the
    median of means (mom) stategy."""
    # TODO: parallel ?
    # TODO: sparse matrix ?
    return median_of_means(grad_coordinate_per_sample(loss_derivative, j, X, y, inner_products, fit_intercept),n_samples_in_block)

    # n_samples = inner_products.shape[0]
    # n_blocks = n_samples // n_samples_in_block
    # last_block_size = n_samples % n_samples_in_block
    #
    # if n_samples % n_samples_in_block == 0:
    #     grad_means_in_blocks = np.empty(n_blocks, dtype=X.dtype)
    # else:
    #     grad_means_in_blocks = np.empty(n_blocks + 1, dtype=X.dtype)
    #
    # # TODO:instanciates in the closure
    # # This shuffle or the indexes to get different blocks each time
    # idx_samples = permutation(n_samples)
    # #permutation(idx_samples)
    #
    # # Cumulative sum in the block
    # grad_block = 0.0
    # # Block counter
    # n_block = 0
    #
    # #print(j)
    # if fit_intercept:
    #     if j[0] == 0:
    #         # In this case it's the derivative w.r.t the intercept
    #         for idx in range(n_samples):
    #             i = idx_samples[idx]
    #             # Update current sum in the block
    #             # print(sum_block, "+=", x[i])
    #             grad_block += loss_derivative(y[i], inner_products[i], j[1])
    #             # sum_block += x[i]
    #             if (idx != 0) and ((idx + 1) % n_samples_in_block == 0):
    #                 # It's the end of the block, we need to save its mean
    #                 # print("sum_block: ", sum_block)
    #                 grad_means_in_blocks[n_block] = grad_block / n_samples_in_block
    #                 n_block += 1
    #                 grad_block = 0.0
    #
    #         if last_block_size != 0:
    #             grad_means_in_blocks[n_blocks] = grad_block / last_block_size
    #
    #         grad_mom = np.median(grad_means_in_blocks)
    #         return grad_mom
    #     else:
    #         for idx in range(n_samples):
    #             i = idx_samples[idx]
    #             # Update current sum in the block
    #             # print(sum_block, "+=", x[i])
    #             grad_block += loss_derivative(y[i], inner_products[i], j[1]) * X[i, j[0] - 1]
    #             # sum_block += x[i]
    #             if (idx != 0) and ((idx + 1) % n_samples_in_block == 0):
    #                 # It's the end of the block, we need to save its mean
    #                 # print("sum_block: ", sum_block)
    #                 grad_means_in_blocks[n_block] = grad_block / n_samples_in_block
    #                 n_block += 1
    #                 grad_block = 0.0
    #
    #         if last_block_size != 0:
    #             grad_means_in_blocks[n_blocks] = grad_block / last_block_size
    #
    #         grad_mom = np.median(grad_means_in_blocks)
    #         return grad_mom
    # else:
    #     # There is no intercept
    #     for idx in range(n_samples):
    #         i = idx_samples[idx]
    #         # Update current sum in the block
    #         # print(sum_block, "+=", x[i])
    #         grad_block += loss_derivative(y[i], inner_products[i], j[1]) * X[i, j[0]]
    #         # sum_block += x[i]
    #         if (idx != 0) and ((idx + 1) % n_samples_in_block == 0):
    #             # It's the end of the block, we need to save its mean
    #             # print("sum_block: ", sum_block)
    #             grad_means_in_blocks[n_block] = grad_block / n_samples_in_block
    #             n_block += 1
    #             grad_block = 0.0
    #
    #     if last_block_size != 0:
    #         grad_means_in_blocks[n_blocks] = grad_block / last_block_size
    #
    #     grad_mom = np.median(grad_means_in_blocks)
    #     return grad_mom


# def erm_strategy_factory(loss, X, y, fit_intercept):
#     @njit
#     def grad_coordinate(j, inner_products):
#         return grad_coordinate_erm(
#             loss.derivative, j, X, y, inner_products, fit_intercept
#         )
#
#     return Strategy(grad_coordinate=grad_coordinate)


def mom_strategy_factory(loss, fit_intercept, n_samples_in_block, **kwargs):
    @njit
    def grad_coordinate(X, y, j, inner_products):
        return grad_coordinate_mom(
            loss.derivative, j, X, y, inner_products, fit_intercept, n_samples_in_block
        )

    return Strategy(
        grad_coordinate=grad_coordinate, n_samples_in_block=n_samples_in_block, name="mom"
    )

@njit
def grad_coordinate_catoni(
    loss_derivative,
    j,
    X,
    y,
    inner_products,
    fit_intercept,
):
    """Computation of the derivative of the loss with respect to a coordinate using the
    catoni stategy."""
    return Holland_catoni_estimator(grad_coordinate_per_sample(loss_derivative, j, X, y, inner_products, fit_intercept))

    # # grad = 0.0
    # # TODO: parallel ?
    # # TODO: sparse matrix ?
    # n_samples = inner_products.shape[0]
    #
    # # TODO:instanciates in the closure
    #
    # place_holder = np.empty(n_samples, dtype=X.dtype)
    # if fit_intercept:
    #     if j[0] == 0:
    #         # In this case it's the derivative w.r.t the intercept
    #         for idx in range(n_samples):
    #             place_holder[idx] = loss_derivative(y[idx], inner_products[idx], j[1])
    #
    #     else:
    #         for idx in range(n_samples):
    #             place_holder[idx] = loss_derivative(y[idx], inner_products[idx], j[1]) * X[idx, j[0] - 1]
    #
    # else:
    #     # There is no intercept
    #     for idx in range(n_samples):
    #         place_holder[idx] = loss_derivative(y[idx], inner_products[idx], j[1]) * X[idx, j[0]]


def catoni_strategy_factory(loss, fit_intercept, **kwargs):
    @njit
    def grad_coordinate(X, y, j, inner_products):
        return grad_coordinate_catoni(
            loss.derivative, j, X, y, inner_products, fit_intercept
        )

    return Strategy(
        grad_coordinate=grad_coordinate, n_samples_in_block=None, name="catoni"
    )

@njit
def grad_coordinate_tmean(
    loss_derivative,
    j,
    X,
    y,
    inner_products,
    fit_intercept,
):
    return trimmed_mean(grad_coordinate_per_sample(loss_derivative, j, X, y, inner_products, fit_intercept))


def tmean_strategy_factory(loss, fit_intercept, **kwargs):
    @njit
    def grad_coordinate(X, y, j, inner_products):
        return grad_coordinate_tmean(
            loss.derivative, j, X, y, inner_products, fit_intercept
        )

    return Strategy(
        grad_coordinate=grad_coordinate, n_samples_in_block=None, name="tmean"
    )



strategies_factory = {"erm": erm_strategy_factory, "mom": mom_strategy_factory, "catoni": catoni_strategy_factory, "tmean": tmean_strategy_factory}

# erm_strategy = Strategy(grad_coordinate=grad_coordinate_erm)

# mom_strategy = TrainingStrategy(grad_coordinate=grad_coordinate_mom)

# erm_strategy_dense
# erm_strategy_sparse


@njit(parallel=True)
def row_squared_norm_dense(model):
    n_samples, n_features = model.X.shape
    if model.fit_intercept:
        norms_squared = np.ones(n_samples, dtype=model.X.dtype)
    else:
        norms_squared = np.zeros(n_samples, dtype=model.X.dtype)
    for i in prange(n_samples):
        for j in range(n_features):
            norms_squared[i] += model.X[i, j] * model.X[i, j]
    return norms_squared


def row_squared_norm(model):
    # TODO: for C and F order with aliasing
    return row_squared_norm_dense(model.no_python)


@njit(parallel=True)
def col_squared_norm_dense(X, fit_intercept):
    n_samples, n_features = X.shape
    if fit_intercept:
        norms_squared = np.zeros(n_features + 1, dtype=X.dtype)
        # First squared norm is n_samples
        norms_squared[0] = n_samples
        for j in prange(1, n_features + 1):
            for i in range(n_samples):
                norms_squared[j] += X[i, j - 1] ** 2
    else:
        norms_squared = np.zeros(n_features, dtype=X.dtype)
        for j in prange(n_features):
            for i in range(n_samples):
                norms_squared[j] += X[i, j] * X[i, j]
    return norms_squared


def col_squared_norm(model):
    # TODO: for C and F order with aliasing
    return col_squared_norm_dense(model.no_python)


#
# @njit
# def grad_batch(model, w, out):
#     out.fill(0)
#     if model.fit_intercept:
#         for i in range(model.n_samples):
#             c = grad_sample_coef(model, i, w) / model.n_samples
#             out[1:] += c * model.X[i]
#             out[0] += c
#     else:
#         for i in range(model.n_samples):
#             c = grad_sample_coef(model, i, w) / model.n_samples
#             out[:] += c * model.X[i]
#     return out
#
#
# @njit(parallel=True)
# def row_squared_norm_dense(model):
#     n_samples, n_features = model.X.shape
#     if model.fit_intercept:
#         norms_squared = np.ones(n_samples, dtype=model.X.dtype)
#     else:
#         norms_squared = np.zeros(n_samples, dtype=model.X.dtype)
#     for i in prange(n_samples):
#         for j in range(n_features):
#             norms_squared[i] += model.X[i, j] * model.X[i, j]
#     return norms_squared
#
#
# def row_squared_norm(model):
#     # TODO: for C and F order with aliasing
#     return row_squared_norm_dense(model.no_python)
#
#
# @njit(parallel=True)
# def col_squared_norm_dense(model):
#     n_samples, n_features = model.X.shape
#     if model.fit_intercept:
#         norms_squared = np.zeros(n_features + 1, dtype=model.X.dtype)
#         # First squared norm is n_samples
#         norms_squared[0] = n_samples
#         for j in prange(1, n_features + 1):
#             for i in range(n_samples):
#                 norms_squared[j] += model.X[i, j - 1] * model.X[i, j - 1]
#     else:
#         norms_squared = np.zeros(n_features, dtype=model.X.dtype)
#         for j in prange(n_features):
#             for i in range(n_samples):
#                 norms_squared[j] += model.X[i, j] * model.X[i, j]
#     return norms_squared
#
#
# def col_squared_norm(model):
#     # TODO: for C and F order with aliasing
#     return col_squared_norm_dense(model.no_python)

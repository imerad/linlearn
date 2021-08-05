from linlearn import MOMRegressor
from linlearn.catoni import Holland_catoni_estimator, standard_catoni_estimator, estimate_sigma
import numpy as np
import logging
import pickle
from datetime import datetime
from scipy.optimize import minimize
from scipy.spatial import distance_matrix
from scipy.stats import fisk
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

file_handler = logging.FileHandler(filename='exp_archives/linreg2_exp.log')
stdout_handler = logging.StreamHandler(sys.stdout)
handlers = [file_handler, stdout_handler]

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S", handlers=handlers
)

save_results = False

logging.info(128*"=")
logging.info("Running new experiment session")
logging.info(128*"=")

if not save_results:
    logging.info("WARNING : results will NOT be saved at the end of this session")

n_repeats = 10

train_sample_sizes = [50 +10*i for i in range(5)]#[50, 100, 200, 500][:2]
n_samples_test = 500
n_features = 5

n_outliers = 10
outliers = False

mom_thresholding = True
mom_thresholding = False
#mom_K = 20
MOMreg_block_size = 0.07
adamom_K_init = 20

catoni_thresholding = True
catoni_thresholding = False

random_seed = 42

noise_sigma = 3
noises = ["gaussian", "lognormal", "tri_s", "loglogistic", "pareto"][:-1]
pareto = 3
fisk_c = 3

rng = np.random.RandomState(random_seed) ## Global random generator

def gen_w_star(d, dist="normal"):
    if dist =="normal":
        return rng.multivariate_normal(np.zeros(d), np.eye(d)).reshape(d)
    elif dist == "uniform":
        return 10 * rng.uniform(size=d) - 5
    else:
        raise Exception("Unknown w_star distribution")


def noise_gen(noise, sigma, n_samples):
    if noise == "gaussian":
        return sigma*rng.normal(size=n_samples)
    elif noise =="lognormal":
        return rng.lognormal(0, np.sqrt(np.log((1 + np.sqrt(1 + 4*sigma**2))/2)), n_samples)
    elif noise =="pareto":
        return sigma * rng.pareto(pareto, n_samples)/(pareto/(((pareto-1)**2)*(pareto-2)))
    elif noise == "loglogistic":
        return sigma * fisk.rvs(fisk_c, size=n_samples, random_state=rng)/np.sqrt(fisk.stats(fisk_c, moments="v"))
    elif noise =="tri_s":
        return sigma*rng.triangular(-1, 0, 1, size=n_samples)/0.4
    else:
        raise ValueError("Unknown noise type")

Sigma_X = np.diag(np.arange(1, n_features+1))
mu_X = np.ones(n_features)

w_star_dist = "normal"

step_size = 0.01
T = 100

eps = 1e-10

def gmom(xs, tol=1e-7):
    y = np.average(xs, axis=0)

    delta = 1
    niter = 0
    while delta > tol:
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


def ERM(X, y):
    n = X.shape[0]
    n_features = X.shape[1]
    XXT = X.T @ X
    Xy = X.T @ y
    empirical_risk = lambda w : ((X @ w - y)**2).mean()/2

    def empirical_gradient(w):
        return (XXT @ w - Xy)/n

    opt = minimize(empirical_risk, np.zeros(n_features), jac=empirical_gradient)
    return opt.x


def minsker_regression(X, y, block_size=10):
    estimators = []
    n = X.shape[0]
    permut = np.random.permutation(n)
    n_blocks = n // block_size
    last_block_size = n % block_size

    b = 0
    while b < n_blocks:
        block_indices = permut[b*block_size:(b+1)*block_size]
        estimators.append(ERM(X[block_indices, :], y[block_indices]))
        b += 1
    if last_block_size > 0:
        block_indices = permut[b*block_size:]
        estimators.append(ERM(X[block_indices, :], y[block_indices]))

    return gmom(np.vstack(estimators))


def Hsu_Sabato_regression(X, y, block_size=10):
    estimators = []
    n = X.shape[0]
    permut = np.random.permutation(n)
    n_blocks = n // block_size
    last_block_size = n % block_size

    b = 0
    while b < n_blocks:
        block_indices = permut[b*block_size:(b+1)*block_size]
        estimators.append(ERM(X[block_indices, :], y[block_indices]))
        b += 1
    if last_block_size > 0:
        block_indices = permut[b*block_size:]
        estimators.append(ERM(X[block_indices, :], y[block_indices]))
    estimators = np.vstack(estimators)
    n_est = estimators.shape[0]
    n2 = n_est//2 + n_est % 2
    distances = distance_matrix(estimators, estimators)
    distances.sort(axis=1)
    return estimators[np.argmin(distances[:,n2]),:]

def LAD_regression(X, y):

    empirical_risk = lambda w : np.abs(X @ w - y).mean()

    def empirical_gradient(w):
        return (np.sign(X @ w - y) @ X)/(X.shape[0])

    opt = minimize(empirical_risk, np.zeros(X.shape[1]), jac=empirical_gradient)
    return opt.x

def MSE(X, y, w):
    return ((X @ w - y)**2).mean()/2


def gradient_descent(x0, grad, step_size, T):
    """run gradient descent for given gradient grad and step size for T steps"""
    x = x0
    for t in range(T):
        grad_x = grad(x)
        x -= step_size * grad_x
    return x

def Holland_regression(X, y, step_size = 0.01, T=100):
    def Holland_gradient(w):
        """compute the grandient used by Holland et al."""
        sample_gradients = np.multiply((X @ w - y)[:,np.newaxis], X)
        catoni_avg_grad = np.zeros_like(w)
        for i in range(w.shape[0]):
            catoni_avg_grad[i] = Holland_catoni_estimator(sample_gradients[:,i])
        return catoni_avg_grad
    return gradient_descent(np.zeros(X.shape[1]), Holland_gradient, step_size, T)

def MOM_cgd(X, y, step_size = 0.01, T=100, strategy="mom"):

    MOM_regressor = MOMRegressor(tol=1e-17, max_iter=T, fit_intercept=False, strategy=strategy,
                                 thresholding=mom_thresholding, step_size=step_size, block_size=MOMreg_block_size)
    MOM_regressor.fit(X, y)
    return MOM_regressor.coef_.flatten()

def catoni_cgd(X, y, step_size = 0.01, T=100):
    return MOM_cgd(X, y, step_size=step_size, T=T, strategy="catoni")

def mom_func(x, block_size):
    """compute median of means of a sequence of numbers x with given block size"""
    n = x.shape[0]
    n_blocks = int(n // block_size)
    last_block_size = n % block_size
    blocks_means = np.zeros(int(n_blocks + int(last_block_size > 0)))
    sum_block = 0.0
    n_block = 0
    for i in range(n):
        # Update current sum in the block
        # print(sum_block, "+=", x[i])
        sum_block += x[i]
        if (i != 0) and ((i + 1) % block_size == 0):
            # It's the end of the block, save its mean
            # print("sum_block: ", sum_block)
            blocks_means[n_block] = sum_block / block_size
            n_block += 1
            sum_block = 0.0

    if last_block_size != 0:
        blocks_means[n_blocks] = sum_block / last_block_size

    mom = np.median(blocks_means)
    return mom#, blocks_means


def adaptive_mom_cgd(X, y, step_size=0.01, T=100, K_init=adamom_K_init):
    """a mom cgd algorithm that adapts block sizes coordinatewise according to changes in coord. gradient variance estimates"""
    n_features = X.shape[1]
    x = np.zeros(n_features)
    MOM_regressor = MOMRegressor(tol=1e-17, max_iter=1, fit_intercept=False, strategy="mom")
    MOM_regressor.fit(X, y)
    steps = MOM_regressor._steps
    K = K_init * np.ones(n_features)
    sample_gradients = np.multiply((X @ x - y)[:, np.newaxis], X)
    n_samples = X.shape[0]
    vars_init = np.array([estimate_sigma(sample_gradients[:,co]) for co in range(n_features)])
    vars = vars_init.copy()

    for t in range(T):

        for i in np.random.permutation(n_features):
            sample_gradients = np.multiply((X @ x - y)[:, np.newaxis], X)
            vars[i] = estimate_sigma(sample_gradients[:,i])
            K[i] = int(min(n_samples//2, max(K_init*((vars_init[i]/vars[i])**2) , K_init)))
            grad_i = mom_func(sample_gradients[:,i], n_samples // K[i])
            #K[i] = int(min(n_samples//2, max(0.1*n_samples/(vars[i]**2) , K_init)))
            #grad_i = mom_func(sample_gradients[:, i], n_samples // (K_init))

            x[i] -= step_size * steps[i] * grad_i

    return x

col_try, col_noise, col_algo, col_val, col_n_samples = [], [], [], [], []

algorithms = [ERM, LAD_regression, minsker_regression, Hsu_Sabato_regression, adaptive_mom_cgd, Holland_regression,
              MOM_cgd, catoni_cgd, adaptive_mom_cgd]

for rep in range(n_repeats):
    if not save_results:
        logging.info("WARNING : results will NOT be saved at the end of this session")
    logging.info(64*'-')
    logging.info("repeat : %d" % (rep+1))
    logging.info(64*'-')

    for n_samples_train in train_sample_sizes:
        logging.info("Sample size %d ..." % n_samples_train)
        for noise in noises:
            logging.info("performing regression with noise : %s" % noise)
            #logging.info("generating train data ...")

            X_train = rng.multivariate_normal(mu_X, Sigma_X, size=n_samples_train)
            w_star = gen_w_star(n_features, dist=w_star_dist)
            y_train = X_train @ w_star + noise_gen(noise, noise_sigma, n_samples_train)

            #outliers
            if outliers:
                X_train = np.concatenate((X_train, np.ones((n_outliers, n_features))), axis=0)
                y_train = np.concatenate((y_train, 10*np.ones(n_outliers)*np.max(np.abs(y_train))))

            #logging.info("generating test data ...")

            X_test = rng.multivariate_normal(mu_X, Sigma_X, size=n_samples_train)
            y_test = X_test @ w_star + noise_gen(noise, noise_sigma, n_samples_train)

            outputs = {}

            #logging.info("Running regression algorithms ...")

            for algo in algorithms:
                col_val.append(MSE(X_test, y_test, algo(X_train, y_train)))
                col_algo.append(algo.__name__)
                col_noise.append(noise)
                col_try.append(rep)
                col_n_samples.append(n_samples_train)


data = pd.DataFrame({"repeat":col_try, "noise": col_noise, "algorithm":col_algo, "MSE":col_val, "sample_size":col_n_samples})

if save_results:
    logging.info("Saving results ...")
    now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

    filename = "linreg2_results_" + now + ".pickle"
    with open("exp_archives/linreg2/" + filename, "wb") as f:
        pickle.dump({"datetime": now, "results": data}, f)

    logging.info("Saved results in file %s" % filename)

# fig, axes = plt.subplots(1, len(noises), sharey=True)
#
# for i in range(len(noises)):
#     sns.lineplot(ax=axes[i], data=data.query("noise == '%s'" % noises[i]), x="sample_size", y="MSE", hue="algorithm", legend = False).set_title(noises[i])
#
# handles, labels = axes[-1].get_legend_handles_labels()
# fig.tight_layout()
# fig.legend(handles, labels, loc='upper center')
#
# plt.show()


g = sns.FacetGrid(
    data, col="noise", col_wrap=len(noises), height=4, sharex=True, sharey=False, legend_out=True
)
g.map(
    sns.lineplot,
    "sample_size",
    "MSE",
    "algorithm",
    ci=False,
    markers=True,
    #lw=4,
)#.set(yscale="log", xlabel="", ylabel="")

g.set_titles(col_template="{col_name}")

axes = g.axes.flatten()

# for i, dataset in enumerate(df["dataset"].unique()):
#     axes[i].set_xticklabels([0, 1, 2, 5, 10, 20, 50], fontsize=14)
#     axes[i].set_title(dataset, fontsize=18)


plt.legend(
    list(data["algorithm"].unique()),
    #bbox_to_anchor=(0.3, 0.7, 1.0, 0.0),
    loc="upper center",
    #ncol=1,
    #borderaxespad=0.0,
    #fontsize=14,
)
plt.show()
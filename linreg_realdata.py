from linlearn import MOMRegressor
from linlearn.robust_means import (
    Holland_catoni_estimator,
    estimate_sigma,
    alg2,
    gmom,
    median_of_means,
)
from linlearn.strategy import median_of_means as mom_func
import numpy as np
import logging
import pickle
from datetime import datetime
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import sys, os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import time
import itertools

file_handler = logging.FileHandler(filename="exp_archives/linreg_realdata.log")
stdout_handler = logging.StreamHandler(sys.stdout)
handlers = [file_handler, stdout_handler]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=handlers,
)


def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


save_results = False
save_fig = True

logging.info(128 * "=")
logging.info("Running new experiment session")
logging.info(128 * "=")

if not save_results:
    logging.info("WARNING : results will NOT be saved at the end of this session")

n_repeats = 5

fit_intercept = False

n_outliers = 10
outliers = False

mom_thresholding = True
mom_thresholding = False
MOMreg_block_size = 0.02

catoni_thresholding = True
catoni_thresholding = False

prasad_delta = 0.01

random_seed = 42
test_size = 0.3

dataset = "Diabetes"#"Boston"#"CaliforniaHousing"#

def load_data_boston():
    from sklearn.datasets import load_boston

    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        shuffle=True,
        random_state=random_seed,
    )
    return X_train, X_test, y_train, y_test


def load_california_housing():
    from sklearn.datasets import fetch_california_housing

    X, y = fetch_california_housing(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        shuffle=True,
        random_state=random_seed,
    )
    return X_train, X_test, y_train, y_test

def fetch_diabetes():
    from sklearn.datasets import load_diabetes

    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        shuffle=True,
        random_state=random_seed,
    )
    return X_train, X_test, y_train, y_test

if dataset == "Boston":
    X_train, X_test, y_train, y_test = load_data_boston()
elif dataset == "CaliforniaHousing":
    X_train, X_test, y_train, y_test = load_california_housing()
elif dataset == "Diabetes":
    X_train, X_test, y_train, y_test = fetch_diabetes()
else:
    raise ValueError("Unknown dataset")

for dat in [X_train, X_test, y_train, y_test]:
    dat = np.ascontiguousarray(dat)

Lip = np.max(
    [
        median_of_means(X_train[:, j] ** 2, int(MOMreg_block_size * X_train.shape[0]))
        for j in range(X_train.shape[1])
    ]
)
n_samples, n_features = X_train.shape

step_size = 0.05
max_iter = 100


def risk(X, y, w, fit_intercept=fit_intercept):
    if fit_intercept:
        w0 = w[0]
        w1 = w[1:]
    else:
        w0 = 0
        w1 = w
    return 0.5 * ((X @ w1 + w0 - y)**2).mean()

def train_risk(w, algo_name=""):
    return risk(X_train, y_train, w, fit_intercept=fit_intercept)
def test_risk(w, algo_name=""):
    return risk(X_test, y_test, w, fit_intercept=fit_intercept)


logging.info(
    "Lauching experiment with parameters : \n dataset : %s, n_repeats = %d , n_samples = %d , n_features = %d , outliers = %r"
    % (dataset, n_repeats, n_samples, n_features, outliers)
)
if outliers:
    logging.info("n_outliers = %d" % n_outliers)
logging.info("block size for MOM_CGD is %f" % MOMreg_block_size)
logging.info(
    "mom_thresholding = %r , random_seed = %d " % (mom_thresholding, random_seed)
)

logging.info("step_size = %f , max_iter = %d" % (step_size, max_iter))

# rng = np.random.RandomState(random_seed)  ## Global random generator

class Record(object):
    def __init__(self, shape, capacity):
        self.record = np.zeros(capacity) if shape == 1 else np.zeros(tuple([capacity] + list(shape)))
        self.cursor = 0
    def update(self, value):
        self.record[self.cursor] = value
        self.cursor += 1
    def __len__(self):
        return self.record.shape[0]

# class Record(object):
#     def __init__(self, shape, capacity):
#         self.record = (
#             np.zeros(capacity)
#             if shape == 1
#             else np.zeros(tuple([capacity] + list(shape)))
#         )
#         self.cursor = 0
#
#     def update(self, value):
#         self.record[self.cursor] = value
#         self.cursor += 1


def Holland_gradient(w):
    """compute the gradient used by Holland et al."""
    if fit_intercept:
        w0 = w[0]
        w1 = w[1:]
    else:
        w0 = 0
        w1 = w
    derivatives = X_train @ w1 + w0 - y_train
    sample_gradients = np.multiply(derivatives[:, np.newaxis], X_train)
    catoni_avg_grad = np.zeros_like(w)
    int_fit_intercept = int(fit_intercept)
    for i in range(int_fit_intercept, w.shape[0]):
        catoni_avg_grad[i] = Holland_catoni_estimator(sample_gradients[:, i-int_fit_intercept])
    if fit_intercept:
        catoni_avg_grad[0] = Holland_catoni_estimator(derivatives)
    return catoni_avg_grad


def Lecue_gradient(w, n_blocks=1+int(1 / MOMreg_block_size)):  # n_blocks must be uneven
    assert n_blocks % 2 == 1
    if fit_intercept:
        w0 = w[0]
        w1 = w[1:]
    else:
        w0 = 0
        w1 = w

    def argmedian(x):
        return np.argpartition(x, len(x) // 2)[len(x) // 2]

    block_size = n_samples // n_blocks
    perm = np.random.permutation(n_samples)
    risks = ((X_train.dot(w1) + w0 - y_train) ** 2) / 2
    means = [
        np.mean(risks[perm[i * block_size : (i + 1) * block_size]])
        for i in range(n_blocks)
    ]
    argmed = argmedian(means)
    indices = perm[argmed * block_size : (argmed + 1) * block_size]
    X_subset, y_subset = X_train[indices, :], y_train[indices]
    derivatives = X_subset @ w1 + w0 - y_subset
    grad = X_subset.T @ derivatives / len(indices)
    if fit_intercept:
        return np.append(grad[::-1], np.mean(derivatives))[::-1]
    else:
        return grad


def empirical_gradient(w):
    if fit_intercept:
        w0 = w[0]
        w1 = w[1:]
    else:
        w0 = 0
        w1 = w
    derivatives = X_train @ w1 + w0 - y_train
    sample_gradients = np.multiply(derivatives[:, np.newaxis], X_train)
    if fit_intercept:
        sample_gradients = np.hstack((derivatives[:, np.newaxis], sample_gradients))
    return sample_gradients.mean(axis=0)


def Prasad_HeavyTail_gradient(w, delta=prasad_delta):
    if fit_intercept:
        w0 = w[0]
        w1 = w[1:]
    else:
        w0 = 0
        w1 = w
    n_blocks = 1 + int(3.5 * np.log(1 / delta))
    block_size = n_samples // n_blocks
    derivatives = X_train @ w1 + w0 - y_train
    sample_gradients = np.multiply(derivatives[:, np.newaxis], X_train)
    if fit_intercept:
        sample_gradients = np.hstack((derivatives[:, np.newaxis], sample_gradients))
    permutation = np.random.permutation(n_samples)
    block_means = []
    for i in range(n_blocks):
        block_means.append(
            np.mean(
                sample_gradients[permutation[i * block_size : (i + 1) * block_size], :],
                axis=0,
            )
        )
    return gmom(np.array(block_means))


def Prasad_outliers_gradient(w, eps=0.05, delta=prasad_delta):
    if fit_intercept:
        w0 = w[0]
        w1 = w[1:]
    else:
        w0 = 0
        w1 = w

    derivatives = X_train @ w1 + w0 - y_train
    sample_gradients = np.multiply(derivatives[:, np.newaxis], X_train)
    if fit_intercept:
        sample_gradients = np.hstack((derivatives[:, np.newaxis], sample_gradients))

    return alg2(sample_gradients, eps, delta)[0]


def linlearn_cgd(X, y, step_size, strategy="mom"):
    logging.info("running %s" % strategy)
    MOM_regressor = MOMRegressor(
        tol=1e-17,
        max_iter=max_iter,
        fit_intercept=fit_intercept,
        strategy=strategy,
        thresholding=mom_thresholding,
        step_size=step_size,
        block_size=MOMreg_block_size,
    )
    param_record = Record((n_features + int(fit_intercept),), max_iter)
    time_record = Record(1, max_iter)

    MOM_regressor.fit(
        X,
        y,
        trackers=[
            lambda w: param_record.update(np.vstack(w).flatten()),
            lambda _: time_record.update(time.time()),
        ],
    )

    return param_record, time_record


def gradient_descent(x0, grad, step_size, max_iter=max_iter):
    """run gradient descent for given gradient grad and step size for max_iter steps"""
    x = x0
    param_record = Record((X_train.shape[1] + int(fit_intercept),), max_iter)
    time_record = Record(1, max_iter)
    for t in tqdm(range(max_iter), desc=grad.__name__):
        x -= step_size * grad(x)
        param_record.update(x)
        time_record.update(time.time())
    return param_record, time_record



metrics = [train_risk, test_risk]  # , "gradient_error"]


def run_repetition(rep):
    if not save_results:
        logging.info("WARNING : results will NOT be saved at the end of this session")

    logging.info(64 * "-")
    logging.info("repeat : %d" % (rep + 1))
    logging.info(64 * "-")

    col_try, col_time, col_algo, col_metric, col_val = [], [], [], [], []

    outputs = {}

    logging.info("Running algorithms ...")

    for strategy in ["mom"]:#, "erm", "catoni"]:
        outputs[strategy+"_cgd"] = linlearn_cgd(X_train, y_train, step_size, strategy=strategy)

    for gradient in [
        empirical_gradient,
        Holland_gradient,
        Prasad_HeavyTail_gradient,
        Lecue_gradient,
        Prasad_outliers_gradient,
    ]:
        logging.info("running %s" % gradient.__name__)
        outputs[gradient.__name__] = gradient_descent(
            np.zeros(n_features+int(fit_intercept)),
            gradient,
            step_size / Lip,
            max_iter,
        )

    logging.info("computing objective history")
    for alg in outputs.keys():
        for ind_metric, metric in enumerate(metrics):
            for i in range(max_iter):
                col_try.append(rep)
                col_algo.append(alg)
                col_metric.append(metric.__name__)
                col_val.append(metric(outputs[alg][0].record[i]))
                col_time.append(outputs[alg][1].record[i] - outputs[alg][1].record[0])
    logging.info("repetition done")
    return col_try, col_algo, col_metric, col_val, col_time

results = [run_repetition(rep) for rep in range(n_repeats)]

col_try = list(itertools.chain.from_iterable([x[0] for x in results]))
col_algo = list(itertools.chain.from_iterable([x[1] for x in results]))
col_metric = list(itertools.chain.from_iterable([x[2] for x in results]))
col_val = list(itertools.chain.from_iterable([x[3] for x in results]))
col_time = list(itertools.chain.from_iterable([x[4] for x in results]))

logging.info("Creating pandas DataFrame")
data = pd.DataFrame(
    {
        "time": col_time,
        "repeat": col_try,
        "algo": col_algo,
        "metric": col_metric,
        "value": col_val,
    }
)

if save_results:
    now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

    import subprocess

    # Get the commit number as a string
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
    commit = commit.decode("utf-8").strip()

    filename = "linreg_realdata_results_%s" + now + ".pickle" % dataset
    ensure_directory("exp_archives/linreg_real_data/")
    with open("exp_archives/linreg_real_data/" + filename, "wb") as f:
        pickle.dump({"datetime": now, "commit": commit, "results": data}, f)

    logging.info("Saved results in file %s" % filename)

logging.info("Plotting ...")

line_width = 1.0

g = sns.FacetGrid(data, col="metric", height=4, legend_out=True)
g.map(sns.lineplot, "time", "value", "algo", lw=line_width).set(xlabel="", ylabel="")

# g.set_titles(col_template="{col_name}")

axes = g.axes.flatten()
axes[0].set_title("Train risk")
axes[1].set_title("Test risk")

color_palette = []
for line in axes[0].get_lines():
    color_palette.append(line.get_c())
color_palette = color_palette[:7]

code_names = {
    "erm_cgd": "ermcgd",
    "catoni_cgd" : "catoni_cgd",
    "empirical_gradient" : "erm",
    "Holland_gradient": "holland",
    "true_gradient": "oracle",
    "mom_cgd": "momcgd",
    "Prasad_HeavyTail_gradient": "gmom_grad",
    "Lecue_gradient": "implicit",
    "catoni_cgd": "catoni_cgd",
    "tmean_cgd": "tmean_cgd",
    "Prasad_outliers_gradient": "Prasad_outliers_gradient",
}

# plt.legend(
axes[0].legend(
    [code_names[name] for name in data["algo"].unique()],
    # bbox_to_anchor=(0.3, 0.7, 1.0, 0.0),
    loc="lower left",
    ncol=2,
    borderaxespad=0.2,
    columnspacing=1.0,
    fontsize=10,
)

plt.tight_layout()
plt.show()

if save_fig:
    now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    specs = "%s_block_size=%.2f" % (
        dataset,
        MOMreg_block_size,
    )
    fig_file_name = "exp_archives/linreg_realdata/" + specs + now + ".pdf"
    ensure_directory("exp_archives/linreg_realdata/")
    g.fig.savefig(fname=fig_file_name, bbox_inches="tight")
    logging.info("Saved figure into file : %s" % fig_file_name)

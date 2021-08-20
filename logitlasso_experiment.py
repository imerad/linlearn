from linlearn import BinaryClassifier
from linlearn.robust_means import Holland_catoni_estimator, gmom
import numpy as np
import logging
import pickle
from datetime import datetime
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import itertools
from tqdm import tqdm
import joblib
import time
from numba import njit
from numba import (
    int64,
    float64,
)
from numba.experimental import jitclass

def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

ensure_directory('exp_archives/')

file_handler = logging.FileHandler(filename='exp_archives/logitlasso_exp.log')
stdout_handler = logging.StreamHandler(sys.stdout)
handlers = [file_handler, stdout_handler]

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S", handlers=handlers
)

save_results = False
save_fig= True

logging.info(64*"=")
logging.info("Running new experiment session")
logging.info(64*"=")

step_size = 0.01

max_iter = 5
fit_intercept = True

n_samples = None
n_repeats = 5

logging.info("Parameters are : n_repeats = %d , n_samples = %d , max_ter = %d , fit_intercept=%r" % (n_repeats, n_samples or 0, max_iter, fit_intercept))

if not save_results:
    logging.info("WARNING : results will NOT be saved at the end of this session")

logging.info("loading data ...")

dataset = "Adult"

with open("pickled_adult.pickle", "rb") as f:
    X_train, X_test, y_train, y_test = pickle.load(f)
    y_train = 2 * y_train - 1
    y_test = 2 * y_test - 1
    for dat in [X_train, X_test, y_train, y_test]:
        data = np.ascontiguousarray(dat)
    f.close()

# record_type = [
#     ("record", float64[:]),
#     ("cursory_pred", int64),
# ]
#@jitclass(record_type)
class Record(object):
    def __init__(self, shape, capacity):
        self.record = np.zeros(capacity) if shape == 1 else np.zeros(tuple([capacity] + list(shape)))
        self.cursor = 0
    def update(self, value):
        self.record[self.cursor] = value
        self.cursor += 1
    def __len__(self):
        return self.record.shape[0]

#@vectorize([float64(float64)])
def logit(x):
    if x > 0:
        return np.log(1 + np.exp(-x))
    else:
        return -x + np.log(1 + np.exp(x))
vec_logit = np.vectorize(logit)

#@vectorize([float64(float64)])
def sigmoid(z):
    if z > 0:
        return 1 / (1 + np.exp(-z))
    else:
        exp_z = np.exp(z)
        return exp_z / (1 + exp_z)
vec_sigmoid = np.vectorize(sigmoid)

def l1_apply_single(x, t):
    if x > t:
        return x - t
    elif x < -t:
        return x + t
    else:
        return 0.0

def sample_objectives(X, y, w, fit_intercept=fit_intercept, lnlearn=False):
    if fit_intercept:
        w0 = w[0] if lnlearn else w[0]
        w1 = w[1] if lnlearn else w[1:]
    else:
        w0 = 0
        w1 = w
    scores = X @ w1 + w0
    return vec_logit(y*scores)

def objective(X, y, w, fit_intercept=fit_intercept, lnlearn=False):
    return sample_objectives(X, y, w, fit_intercept=fit_intercept, lnlearn=lnlearn).mean()

def sample_gradients(X, y, w, fit_intercept=fit_intercept):
    scores = X @ w[1:] + w[0] if fit_intercept else X @ w

    derivatives = -y * vec_sigmoid(-y * scores)
    if fit_intercept:
        return np.hstack((derivatives[:,np.newaxis], np.einsum("i,ij->ij", derivatives, X)))
    else:
        return np.einsum("i,ij->ij", derivatives, X)

def gradient(X, y, w, fit_intercept=fit_intercept):
    return sample_gradients(X, y, w, fit_intercept=fit_intercept).mean(axis=0)

linlearn_algorithms = ["mom_cgd", "catoni_cgd", "tmean_cgd"]

def train_loss(w, algo_name=""):
    return objective(X_train, y_train, w, fit_intercept=fit_intercept, lnlearn=algo_name in linlearn_algorithms)
def test_loss(w, algo_name=""):
    return objective(X_test, y_test, w, fit_intercept=fit_intercept, lnlearn=algo_name in linlearn_algorithms)


def mom_cgd(X_train, y_train, l1_penalty=1):
    mom_logreg = BinaryClassifier(tol=1e-17, max_iter=max_iter, strategy="mom", fit_intercept=fit_intercept, penalty="l1", C=l1_penalty,
                              step_size=step_size, loss="logistic")
    param_record = Record((X_train.shape[1]+int(fit_intercept),), max_iter)
    time_record = Record(1, max_iter)
    mom_logreg.fit(X_train, y_train, trackers=[lambda w: param_record.update(np.vstack(w).flatten()), lambda _:time_record.update(time.time())])

    # n_iter = len(mom_logreg.optimization_result_.tracked_funs[0])
    # n_batches = 1 if batch_size == 0 else X_train.shape[0] // batch_size + int(X_train.shape[0] % batch_size > 0)
    # gradient_counts = [(i // n_batches) * X_train.shape[0] + (i % n_batches) * batch_size for i in
    #                    range(n_iter)]

    return param_record, time_record

#@njit
def SAGA(X, y, w0=None, fit_intercept=fit_intercept, l1_penalty=0):
    if w0 is None:
        wt = np.zeros(X.shape[1] + int(fit_intercept))
    else:
        wt = w0

    param_record = Record((X_train.shape[1]+int(fit_intercept),), max_iter)
    time_record = Record(1, max_iter)

    table = X @ wt[1:] + wt[0] if fit_intercept else X @ wt
    table = -y * vec_sigmoid(-y * table)
    if fit_intercept:
        mean_grad = np.hstack((table[:,np.newaxis], np.einsum("i,ij->ij", table, X))).mean(axis=0)
    else:
        mean_grad = (table * X)

    step = step_size/X.shape[0]

    for i in tqdm(range(max_iter), desc="SAGA"):
        for sam in range(X.shape[0]):
            j = np.random.randint(X.shape[0])
            new_j_score = X[j,:].dot(wt[1:]) + wt[0] if fit_intercept else X[j,:].dot(wt)
            new_j_der = -y[j] * sigmoid(-y[j] * new_j_score)
            Xj  = (np.concatenate((np.array([1]), X[j,:])) if fit_intercept else X[j,:])
            j_grad_diff = (new_j_der - table[j]) * Xj
            wt -= step * (j_grad_diff + mean_grad)
            mean_grad += j_grad_diff*Xj/X.shape[0]
            table[j] = new_j_der

            for k in range(int(fit_intercept), len(wt)):
                wt[k] = l1_apply_single(wt[k], l1_penalty*step)
        param_record.update(wt)
        time_record.update(time.time())

    return param_record, time_record


metrics = [train_loss, test_loss]

def run_repetition(rep):
    col_try, col_algo, col_metric, col_val, col_time = [], [], [], [], []

    outputs = {}
    def announce(x):
        logging.info(str(rep)+" : "+x+" done")
    outputs["SAGA"] = SAGA(X_train, y_train)
    announce("SAGA")
    outputs["mom_cgd"] = mom_cgd(X_train, y_train)
    announce("mom_cgd")

    logging.info("computing objective history")
    for alg in outputs.keys():
        for ind_metric, metric in enumerate(metrics):
            for i in range(max_iter):
                col_try.append(rep)
                col_algo.append(alg)
                col_metric.append(metric.__name__)
                col_val.append(metric(outputs[alg][0].record[i]))
                col_time.append(outputs[alg][1].record[i] - outputs[alg][1].record[0])
    print("repetition done")
    return col_try, col_algo, col_metric, col_val, col_time

if os.cpu_count() > 8:
    #logging.info("precompiling linlearn code")
    logging.info("running parallel repetitions")
    results = joblib.Parallel(n_jobs=-1)(joblib.delayed(run_repetition)(rep) for rep in range(n_repeats))
else:
    results = [run_repetition(rep) for rep in range(n_repeats)]

col_try = list(itertools.chain.from_iterable([x[0] for x in results]))
col_algo = list(itertools.chain.from_iterable([x[1] for x in results]))
col_metric = list(itertools.chain.from_iterable([x[2] for x in results]))
col_val = list(itertools.chain.from_iterable([x[3] for x in results]))
col_time = list(itertools.chain.from_iterable([x[4] for x in results]))


data = pd.DataFrame({"repeat":col_try, "algorithm":col_algo, "metric":col_metric, "value":col_val, "time": col_time})

if save_results:
    logging.info("Saving results ...")
    now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

    filename = "logitlasso_"+dataset+"_results_" + now + ".pickle"
    ensure_directory("exp_archives/logitlasso/")
    with open("exp_archives/logitlasso/" + filename, "wb") as f:
        pickle.dump({"datetime": now, "results": data}, f)

    logging.info("Saved results in file %s" % filename)


g = sns.FacetGrid(
    data, col="metric", height=4, legend_out=True
)
g.map(
    sns.lineplot,
    "time",
    "value",
    "algorithm",
    #lw=4,
)#.set(yscale="log")#, xlabel="", ylabel="")

#g.set_titles(col_template="{col_name}")

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
#g.fig.subplots_adjust(top=0.9)
#g.fig.suptitle('n=%d , noise=%s , $\\sigma$ = %.2f, block_size=%.2f, w_star_dist=%s' % (n_samples, noise_dist, noise_sigma[noise_dist], MOMreg_block_size, w_star_dist))

plt.show()


if save_fig:
    now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    ensure_directory("exp_archives/logitlasso/")
    #specs = 'n%d_%s%.2f_block_size=%.2f_w_dist=%s' % (n_samples, noise_dist, noise_sigma[noise_dist], MOMreg_block_size, w_star_dist)
    fig_file_name = "exp_archives/logitlasso/" + dataset + now + ".pdf"
    # g.fig.savefig(fname=fig_file_name)#, bbox_inches='tight')
    logging.info("Saved figure into file : %s" % fig_file_name)

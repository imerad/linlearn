from linlearn import BinaryClassifier, MultiClassifier
from linlearn.robust_means import Holland_catoni_estimator, gmom, alg2
import numpy as np
import gzip
import logging
import pickle
from datetime import datetime
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import logsumexp, softmax
import os
import itertools
from tqdm import tqdm
import joblib
import time

def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

ensure_directory('exp_archives/')

file_handler = logging.FileHandler(filename='exp_archives/classif_exp.log')
stdout_handler = logging.StreamHandler(sys.stdout)
handlers = [file_handler, stdout_handler]

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S", handlers=handlers
)

save_results = False
save_fig= True

dataset="MNIST"

logging.info(64*"=")
logging.info("Running new experiment session ON GPU with dataset : %s" % dataset)
logging.info(64*"=")

m_SVRG = 50
step_size = 0.01

max_iter = 10
fit_intercept = True

n_samples = 1000
n_repeats = 2

logging.info("Parameters are : n_repeats = %d , n_samples = %d , max_ter = %d , fit_intercept=%r , m_SVRG = %d" % (n_repeats, n_samples or 0, max_iter, fit_intercept, m_SVRG))

if not save_results:
    logging.info("WARNING : results will NOT be saved at the end of this session")

def _images(path):
    """Return images loaded locally."""
    with gzip.open(path) as f:
        # First 16 bytes are magic_number, n_imgs, n_rows, n_cols
        pixels = np.frombuffer(f.read(), 'B', offset=16)
    return pixels.reshape(-1, 784).astype('float64') / 255


def _labels(path):
    """Return labels loaded locally."""
    with gzip.open(path) as f:
        # First 8 bytes are magic_number, n_labels
        integer_labels = np.frombuffer(f.read(), 'B', offset=8)

    def _onehot(integer_labels):
        """Return matrix whose rows are onehot encodings of integers."""
        n_rows = len(integer_labels)
        n_cols = integer_labels.max() + 1
        onehot = np.zeros((n_rows, n_cols), dtype='uint8')
        onehot[np.arange(n_rows), integer_labels] = 1
        return onehot

    return _onehot(integer_labels)

mnist_train_images_file = "mnist_data/train-images-idx3-ubyte.gz"
mnist_train_labels_file = "mnist_data/train-labels-idx1-ubyte.gz"

mnist_test_images_file = "mnist_data/t10k-images-idx3-ubyte.gz"
mnist_test_labels_file = "mnist_data/t10k-labels-idx1-ubyte.gz"

logging.info("loading data ...")
X_train = _images(mnist_train_images_file)[:n_samples]
y_train = _labels(mnist_train_labels_file)[:n_samples]

X_test = _images(mnist_test_images_file)
y_test = _labels(mnist_test_labels_file)

def l1_apply_single(x, t):
    if x > t:
        return x - t
    elif x < -t:
        return x + t
    else:
        return 0.0


def sample_objectives(X, y, w, fit_intercept=fit_intercept, lnlearn=False):
    if fit_intercept:
        w0 = w[0] if lnlearn else w[0,:]
        w1 = w[1] if lnlearn else w[1:,:]
    else:
        w0 = 0
        w1 = w
    scores = X @ w1 + w0
    scores = np.hstack((scores, np.zeros((X.shape[0], 1))))
    obj = (-scores[np.arange(X.shape[0]), np.argmax(y, axis=1)] + logsumexp(scores, axis=1))
    return obj

def objective(X, y, w, fit_intercept=fit_intercept, lnlearn=False):
    return sample_objectives(X, y, w, fit_intercept=fit_intercept, lnlearn=lnlearn).mean()


def gradient(X, y, w, fit_intercept=fit_intercept):
    scores = X @ w[1:,:] + w[0,:] if fit_intercept else X @ w
    scores = np.hstack((scores, np.zeros((X.shape[0], 1))))
    sftmax = softmax(scores, axis=1) - y#np.hstack((y, np.zeros((X.shape[0], 1))))
    if fit_intercept:
        return np.vstack((sftmax[:,:-1].sum(axis=0), X.T @ sftmax[:,:-1]))/X.shape[0]
    else:
        return (X.T @ sftmax[:,:-1])/X.shape[0] # np.vstack((np.ones((X.shape[0], 1)) @ sftmax[:,:-1], X.T @ sftmax[:,:-1]

def sample_gradients(X, y, w, fit_intercept=fit_intercept):
    scores = X @ w[1:,:] + w[0,:] if fit_intercept else X @ w
    scores = np.hstack((scores, np.zeros((X.shape[0], 1))))
    sftmax = softmax(scores, axis=1) - y#np.hstack((y, np.zeros((X.shape[0], 1))))
    if fit_intercept:
        return np.concatenate(
            (sftmax[:,np.newaxis,:-1], np.einsum("ij, ik->ijk", X, sftmax[:,:-1])), axis=1)
    else:
        return np.einsum("ij, ik->ijk", X, sftmax[:,:-1])

# def train_loss(w): return objective(X_train, y_train, w, fit_intercept=fit_intercept)
# def test_loss(w): return objective(X_test, y_test, w, fit_intercept=fit_intercept)
#
# def linlearn_train_loss(w): return objective(X_train, y_train, w, fit_intercept=fit_intercept, lnlearn=True)
# def linlearn_test_loss(w): return objective(X_test, y_test, w, fit_intercept=fit_intercept, lnlearn=True)
# linlearn_tracked_funs = [linlearn_train_loss, linlearn_test_loss]
linlearn_algorithms = ["mom_cgd", "catoni_cgd", "tmean_cgd"]

def train_loss(w, algo_name=""):
    return objective(X_train, y_train, w, fit_intercept=fit_intercept, lnlearn=algo_name in linlearn_algorithms)
def test_loss(w, algo_name=""):
    return objective(X_test, y_test, w, fit_intercept=fit_intercept, lnlearn=algo_name in linlearn_algorithms)
tracked_funs = [train_loss, test_loss]


class Record(object):
    def __init__(self, shape, capacity):
        self.record = np.zeros(capacity) if shape == 1 else np.zeros(tuple([capacity] + list(shape)))
        self.cursor = 0
    def update(self, value):
        self.record[self.cursor] = value
        self.cursor += 1
    def __len__(self):
        return self.record.shape[0]


def tmean_cgd(X_train, y_train, batch_size=500):
    mom_logreg = MultiClassifier(tol=1e-17, max_iter=max_iter, strategy="tmean", fit_intercept=fit_intercept,
                              thresholding=False, step_size=step_size*batch_size/1000, loss="multilogistic", batch_size=batch_size)
    mom_logreg.fit(X_train, y_train, tracked_funs=linlearn_tracked_funs)

    n_iter = len(mom_logreg.optimization_result_.tracked_funs[0])
    n_batches = X_train.shape[0] // batch_size + int(X_train.shape[0] % batch_size > 0)
    gradient_counts = [(i // n_batches)*X_train.shape[0] + (i % n_batches)*batch_size for i in range(n_iter)]

    return mom_logreg.optimization_result_.tracked_funs + [gradient_counts]


# def catoni_cgd(X_train, y_train, batch_size=500):
#     mom_logreg = MultiClassifier(tol=1e-17, max_iter=max_iter, strategy="catoni", fit_intercept=fit_intercept,
#                               thresholding=False, step_size=step_size*batch_size/1000, loss="multilogistic", batch_size=batch_size)
#     mom_logreg.fit(X_train, y_train, tracked_funs=linlearn_tracked_funs)
#
#     n_iter = len(mom_logreg.optimization_result_.tracked_funs[0])
#     n_batches = X_train.shape[0] // batch_size + int(X_train.shape[0] % batch_size > 0)
#     gradient_counts = [(i // n_batches)*X_train.shape[0] + (i % n_batches)*batch_size for i in range(n_iter)]
#
#     return mom_logreg.optimization_result_.tracked_funs + [gradient_counts]

def catoni_cgd(X_train, y_train, l1_penalty=1):
    mom_logreg = MultiClassifier(tol=1e-17, max_iter=max_iter, strategy="catoni", fit_intercept=fit_intercept, penalty="l1", C=l1_penalty,
                              step_size=step_size, loss="multilogistic")
    param_record = Record((X_train.shape[1]+int(fit_intercept), y_train.shape[1]-1), max_iter)
    time_record = Record(1, max_iter)

    if fit_intercept:
        param_tracker = lambda w : param_record.update(np.vstack(w))
    else:
        param_tracker = lambda w : param_record.update(w)

    mom_logreg.fit(X_train, y_train, trackers=[param_tracker, lambda _:time_record.update(time.time())])

    return param_record, time_record



# def mom_cgd(X_train, y_train, batch_size=500):
#     mom_logreg = MultiClassifier(tol=1e-17, max_iter=max_iter, strategy="mom", fit_intercept=fit_intercept,
#                               thresholding=False, step_size=step_size*batch_size/1000, loss="multilogistic", batch_size=batch_size)
#     mom_logreg.fit(X_train, y_train, tracked_funs=linlearn_tracked_funs)
#
#     n_iter = len(mom_logreg.optimization_result_.tracked_funs[0])
#     n_batches = X_train.shape[0] // batch_size + int(X_train.shape[0] % batch_size > 0)
#     gradient_counts = [(i // n_batches) * X_train.shape[0] + (i % n_batches) * batch_size for i in
#                        range(n_iter)]
#
#     return mom_logreg.optimization_result_.tracked_funs + [gradient_counts]

def mom_cgd(X_train, y_train, l1_penalty=1):
    mom_logreg = MultiClassifier(tol=1e-17, max_iter=max_iter, strategy="mom", fit_intercept=fit_intercept, penalty="l1", C=l1_penalty,
                              step_size=step_size, loss="multilogistic")
    param_record = Record((X_train.shape[1]+int(fit_intercept), y_train.shape[1]-1), max_iter)
    time_record = Record(1, max_iter)

    if fit_intercept:
        param_tracker = lambda w : param_record.update(np.vstack(w))
    else:
        param_tracker = lambda w : param_record.update(w)

    mom_logreg.fit(X_train, y_train, trackers=[param_tracker, lambda _:time_record.update(time.time())])

    return param_record, time_record


# def SVRG(X, y, grad, m, w0=None, T=max_iter, fit_intercept=fit_intercept, tracked_funs=tracked_funs):
#     if w0 is None:
#         w0 = np.zeros((X.shape[1] + int(fit_intercept), y.shape[1]-1))
#     w_tilde = w0
#     wt = w0
#     step = step_size*(X.shape[0]/m + 2)/1000
#     tracks = [[obj(w0)] for obj in tracked_funs] + [[0]]
#     for i in tqdm(range((T*500)//(X.shape[0] + 2*m) + 1), desc="SVRG"):
#         mu = grad(X, y, w_tilde, fit_intercept=fit_intercept)
#         additional_gradients = X.shape[0]
#         for j in range(m):
#             ind = np.random.randint(X.shape[0])
#             X_ind, y_ind = X[ind:ind+1,:], y[ind:ind+1,:]
#             wt -= step*(grad(X_ind, y_ind, wt, fit_intercept=fit_intercept) - grad(X_ind, y_ind, w_tilde, fit_intercept=fit_intercept) + mu)
#             additional_gradients += 2
#             for idx, obj in enumerate(tracked_funs):
#                 tracks[idx].append(obj(wt))
#             tracks[-1].append(tracks[-1][-1] + additional_gradients)
#             additional_gradients = 0
#         w_tilde = wt
#     return tracks

def SVRG(X, y, w0=None, T=max_iter, fit_intercept=fit_intercept):
    if w0 is None:
        w0 = np.zeros((X.shape[1] + int(fit_intercept), y.shape[1]-1))
    w_tilde = w0
    wt = w0
    step = step_size/(X.shape[0])
    m = X.shape[0]

    param_record = Record((X.shape[1]+int(fit_intercept), y.shape[1]-1), max_iter)
    time_record = Record(1, max_iter)

    for i in tqdm(range(T), desc="SVRG"):
        mu = gradient(X, y, w_tilde, fit_intercept=fit_intercept)

        for j in range(m):
            ind = np.random.randint(X.shape[0])
            X_ind, y_ind = X[ind:ind+1,:], y[ind:ind+1]
            wt -= step*(gradient(X_ind, y_ind, wt, fit_intercept=fit_intercept) - gradient(X_ind, y_ind, w_tilde, fit_intercept=fit_intercept) + mu)
        w_tilde = wt
        param_record.update(wt)
        time_record.update(time.time())

    return param_record, time_record


def SGD(X, y, w0=None, T=max_iter, fit_intercept=fit_intercept):
    if w0 is None:
        w0 = np.zeros((X.shape[1] + int(fit_intercept), y.shape[1]-1))
    wt = w0

    step = step_size/(X.shape[0])

    param_record = Record((X.shape[1]+int(fit_intercept), y.shape[1]-1), max_iter)
    time_record = Record(1, max_iter)

    for i in tqdm(range(T), desc="SGD"):
        index = np.random.randint(X.shape[0])
        wt -= step * gradient(X[index:index+1,:], y[index:index+1,:], wt, fit_intercept=fit_intercept)
        param_record.update(wt)
        time_record.update(time.time())

    return param_record, time_record

def Holland_gd(X, y, w0=None, T=max_iter, fit_intercept=fit_intercept):
    if w0 is None:
        w0 = np.zeros((X.shape[1] + int(fit_intercept), y.shape[1]-1))
    wt = w0
    param_record = Record((X.shape[1]+int(fit_intercept), y.shape[1]-1), max_iter)
    time_record = Record(1, max_iter)

    for i in tqdm(range(T), desc="Holland"):
        gradients = sample_gradients(X, y, wt, fit_intercept=fit_intercept)
        catoni_avg_grad = np.zeros_like(wt)
        for k in range(wt.shape[0]):
            for l in range(wt.shape[1]):
                catoni_avg_grad[k,l] = Holland_catoni_estimator(gradients[:, k, l])

        wt -= step_size * catoni_avg_grad

        param_record.update(wt)
        time_record.update(time.time())

    return param_record, time_record


# def Prasad_heavyTails_gd(X, y, w0=None, T=max_iter, fit_intercept=fit_intercept, delta=0.01):
#     if w0 is None:
#         w0 = np.zeros((X.shape[1] + int(fit_intercept), y.shape[1]-1))
#     n_blocks = 1 + int(3.5 * np.log(1 / delta))
#     block_size = batch_size // n_blocks
#     wt = w0
#     step = step_size*batch_size/1000
#     tracks = [[obj(wt)] for obj in tracked_funs] + [[0]]
#     for i in tqdm(range(T), desc = "prasad_heavy_tail"):
#         indices = np.random.randint(X.shape[0], size=batch_size)
#         gradients = sample_gradients(X[indices,:], y[indices,:], wt, fit_intercept=fit_intercept)
#         permutation = np.random.permutation(batch_size)
#         block_means = []
#         for j in range(n_blocks):
#             block_means.append(np.mean(gradients[permutation[j * block_size:(j + 1) * block_size], :], axis=0).reshape(-1))
#         grad = gmom(np.array(block_means)).reshape(wt.shape)
#         wt -= step * grad
#
#         for idx, obj in enumerate(tracked_funs):
#             tracks[idx].append(obj(wt))
#         tracks[-1].append(tracks[-1][-1] + batch_size)
#
#     return tracks

def Prasad_heavyTails_gd(X, y, w0=None, T=max_iter, fit_intercept=fit_intercept, delta=0.01, l1_penalty=0):
    if w0 is None:
        w0 = np.zeros((X.shape[1] + int(fit_intercept), y.shape[1] - 1))
    n_blocks = 1 + int(3.5 * np.log(1 / delta))
    block_size = X.shape[0] // n_blocks
    wt = w0

    param_record = Record((X.shape[1]+int(fit_intercept), y.shape[1] - 1), max_iter)
    time_record = Record(1, max_iter)


    for i in tqdm(range(T), desc = "prasad_heavy_tail"):
        gradients = sample_gradients(X, y, wt, fit_intercept=fit_intercept)
        permutation = np.random.permutation(X.shape[0])
        block_means = []
        for j in range(n_blocks):
            block_means.append(np.mean(gradients[permutation[j * block_size:(j + 1) * block_size], :], axis=0).reshape(-1))
        grad = gmom(np.array(block_means)).reshape(wt.shape)
        wt -= step_size * grad
        for k in range(int(fit_intercept), wt.shape[0]):
            for l in range(wt.shape[1]):
                wt[k,l] = l1_apply_single(wt[k,l], l1_penalty * step_size)

        param_record.update(wt)
        time_record.update(time.time())

    return param_record, time_record


# def Lecue_gd(X, y, w0=None, T=max_iter, batch_size=500, fit_intercept=fit_intercept, tracked_funs=tracked_funs, n_blocks=21):
#     if w0 is None:
#         w0 = np.zeros((X.shape[1] + int(fit_intercept), y.shape[1]-1))
#
#     def argmedian(x):
#         return np.argpartition(x, len(x) // 2)[len(x) // 2]
#     block_size = batch_size // n_blocks
#     wt = w0
#     step = step_size*batch_size/1000
#     tracks = [[obj(wt)] for obj in tracked_funs] + [[0]]
#     for i in tqdm(range(T), desc = "Lecue"):
#         indices = np.random.randint(X.shape[0], size=batch_size)
#         objectives = sample_objectives(X[indices,:], y[indices,:], wt, fit_intercept=fit_intercept)
#
#         perm = np.random.permutation(len(indices))
#         means = [
#             np.mean(objectives[perm[j * block_size: (j + 1) * block_size]])
#             for j in range(n_blocks)
#         ]
#         argmed = argmedian(means)
#         indices = perm[argmed * block_size: (argmed + 1) * block_size]
#         X_subset, y_subset = X[indices, :], y[indices]
#
#
#         grad = gradient(X_subset, y_subset, wt, fit_intercept=fit_intercept)
#         wt -= step * grad
#
#         for idx, obj in enumerate(tracked_funs):
#             tracks[idx].append(obj(wt))
#         tracks[-1].append(tracks[-1][-1] + batch_size)
#
#     return tracks

def Lecue_gd(X, y, w0=None, T=max_iter, fit_intercept=fit_intercept, n_blocks=21):
    """n_blocks must be uneven"""
    if w0 is None:
        w0 = np.zeros((X.shape[1] + int(fit_intercept), y.shape[1]-1))

    def argmedian(x):
        return np.argpartition(x, len(x) // 2)[len(x) // 2]
    block_size = X.shape[0] // n_blocks
    wt = w0
    param_record = Record((X.shape[1]+int(fit_intercept), y.shape[1]-1), max_iter)
    time_record = Record(1, max_iter)


    for i in tqdm(range(T), desc = "Lecue"):

        objectives = sample_objectives(X, y, wt, fit_intercept=fit_intercept)

        perm = np.random.permutation(X.shape[0])
        means = [
            np.mean(objectives[perm[j * block_size: (j + 1) * block_size]])
            for j in range(n_blocks)
        ]
        argmed = argmedian(means)
        indices = perm[argmed * block_size: (argmed + 1) * block_size]
        X_subset, y_subset = X[indices, :], y[indices]

        grad = gradient(X_subset, y_subset, wt, fit_intercept=fit_intercept)
        wt -= step_size * grad
        param_record.update(wt)
        time_record.update(time.time())

    return param_record, time_record


# This one is too computationally heavy

# def Prasad_outliers_gd(X, y, w0=None, step=0.01, T=max_iter, batch_size=100, fit_intercept=fit_intercept, tracked_funs=tracked_funs, eps = 0.01, delta=0.01):
#     if w0 is None:
#         w0 = np.zeros((X.shape[1] + int(fit_intercept), y.shape[1]-1))
#     wt = w0
#     tracks = [[obj(wt)] for obj in tracked_funs] + [[0]]
#     for i in range(T):
#         print("iter")
#         indices = np.random.randint(X.shape[0], size=batch_size)
#         gradients = sample_gradients(X[indices,:], y[indices,:], wt, fit_intercept=fit_intercept)
#         grad = alg2(gradients.reshape((batch_size, -1)), eps, delta)[0]
#         wt -= step * grad.reshape(wt.shape)
#
#         for idx, obj in enumerate(tracked_funs):
#             tracks[idx].append(obj(wt))
#         tracks[-1].append(tracks[-1][-1] + batch_size)
#
#     return tracks




metrics = [train_loss, test_loss]

def run_repetition(rep):
    col_try, col_algo, col_metric, col_time, col_val = [], [], [], [], []

    outputs = {}
    def announce(x):
        print(str(rep)+" : "+x+" done")

    outputs["SGD"] = SGD(X_train, y_train)
    announce("SGD")

    outputs["SVRG"] = SVRG(X_train, y_train)#, gradient, m_SVRG)
    announce("SVRG")
    outputs["lecue_gd"] = Lecue_gd(X_train, y_train)
    announce("lecue_gd")
    outputs["Prasad_heavytails"] = Prasad_heavyTails_gd(X_train, y_train)
    announce("gmom_gd")

    outputs["mom_cgd"] = mom_cgd(X_train, y_train)
    announce("mom_cgd")

    outputs["Holland_gd"] = Holland_gd(X_train, y_train)
    announce("Holland_gd")

    outputs["catoni_cgd"] = catoni_cgd(X_train, y_train)
    announce("catoni_cgd")

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


data = pd.DataFrame({"repeat":col_try, "algorithm":col_algo, "metric":col_metric, "value":col_val, "time" : col_time})

if save_results:
    logging.info("Saving results ...")
    now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

    filename = "classif_"+dataset+"_results_" + now + ".pickle"
    ensure_directory("exp_archives/classif/")
    with open("exp_archives/classif/" + filename, "wb") as f:
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
    ensure_directory("exp_archives/classif/")
    #specs = 'n%d_%s%.2f_block_size=%.2f_w_dist=%s' % (n_samples, noise_dist, noise_sigma[noise_dist], MOMreg_block_size, w_star_dist)
    fig_file_name = "exp_archives/classif/" + dataset + now + ".pdf"
    g.fig.savefig(fname=fig_file_name, bbox_inches='tight')
    logging.info("Saved figure into file : %s" % fig_file_name)

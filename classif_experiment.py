from linlearn import BinaryClassifier, MultiClassifier
from linlearn.catoni import Holland_catoni_estimator, standard_catoni_estimator, estimate_sigma
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


file_handler = logging.FileHandler(filename='exp_archives/classif_exp.log')
stdout_handler = logging.StreamHandler(sys.stdout)
handlers = [file_handler, stdout_handler]

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S", handlers=handlers
)

save_results = False

logging.info(64*"=")
logging.info("Running new experiment session")
logging.info(64*"=")

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

# mnist_train_images = _images(mnist_train_images_file)
# mnist_train_labels = _labels(mnist_train_labels_file)
X_train = _images(mnist_train_images_file)
y_train = _labels(mnist_train_labels_file)

# mnist_test_images = _images(mnist_test_images_file)
# mnist_test_labels = _labels(mnist_test_labels_file)
X_test = _images(mnist_test_images_file)
y_test = _labels(mnist_test_labels_file)

save_results = False
max_iter = 150
fit_intercept = True

logging.info(128*"=")
logging.info("Running new experiment session")
logging.info(128*"=")

if not save_results:
    logging.info("WARNING : results will NOT be saved at the end of this session")

n_repeats = 10


def catoni_cgd(X_train, y_train, step_size=0.1):
    mom_logreg = MultiClassifier(tol=1e-17, max_iter=max_iter, strategy="catoni", fit_intercept=fit_intercept,
                              thresholding=False, step_size=step_size, loss="multilogistic", penalty="none")
    mom_logreg.fit(X_train, y_train, tracked_funs=linlearn_tracked_funs)
    return mom_logreg.optimization_result_.tracked_funs


def mom_cgd(X_train, y_train, step_size=0.1):
    mom_logreg = MultiClassifier(tol=1e-17, max_iter=max_iter, strategy="mom", fit_intercept=fit_intercept,
                              thresholding=False, step_size=step_size, loss="multilogistic", penalty="none")
    mom_logreg.fit(X_train, y_train, tracked_funs=linlearn_tracked_funs)
    return mom_logreg.optimization_result_.tracked_funs


def SVRG(X, y, obj, grad, w0, step, m, T, fit_intercept=False):
    w_tilde = w0
    wt = w0
    track_objective = [obj(X, y, w0, fit_intercept=fit_intercept)]
    for i in range(T//m):
        mu = grad(X, y, w_tilde, fit_intercept=fit_intercept)
        for j in range(m):
            ind = np.random.randint(X.shape[0])
            wt -= step*(grad(X[ind:ind+1,:], y[ind:ind+1,:], wt, fit_intercept=fit_intercept) - grad(X[ind:ind+1,:], y[ind:ind+1,:], w_tilde, fit_intercept=fit_intercept) + mu)
            track_objective.append(obj(X, y, wt, fit_intercept=fit_intercept))
        w_tilde = wt
    return w_tilde, track_objective

def SGD(X, y, obj, grad, w0, step, m, T, fit_intercept=False):
    wt = w0
    track_objective = [obj(X, y, w0, fit_intercept=fit_intercept)]
    for i in range(T):
        ind = np.random.randint(X.shape[0])
        wt -= step * grad(X[ind:ind+1,:], y[ind:ind+1,:], wt, fit_intercept=fit_intercept)
        track_objective.append(obj(X, y, wt, fit_intercept=fit_intercept))
        w_tilde = wt
    return w_tilde, track_objective

def objective(X, y, w, fit_intercept=False, linlearn=False):
    if fit_intercept:
        w0 = w[0] if linlearn else w[0,:]
        w1 = w[1] if linlearn else w[1:,:]
    else:
        w0 = 0
        w1 = w
    scores = X @ w1 + w0
    scores = np.hstack((scores, np.zeros((X.shape[0], 1))))
    obj = (-scores[np.arange(X.shape[0]), np.argmax(y, axis=1)] + logsumexp(scores, axis=1)).mean()
    return obj

def gradient(X, y, w, fit_intercept=False):
    scores = X @ w[1:,:] + w[0,:] if fit_intercept else X @ w
    scores = np.hstack((scores, np.zeros((X.shape[0], 1))))
    sftmax = softmax(scores, axis=1) - y#np.hstack((y, np.zeros((X.shape[0], 1))))
    if fit_intercept:
        return np.vstack((sftmax[:,:-1].sum(axis=0), X.T @ sftmax[:,:-1]))/X.shape[0]
    else:
        return (X.T @ sftmax[:,:-1])/X.shape[0] # np.vstack((np.ones((X.shape[0], 1)) @ sftmax[:,:-1], X.T @ sftmax[:,:-1]

algorithms = [mom_cgd, catoni_cgd, SVRG, SGD]

def linlearn_train_loss(w): return objective(X_train, y_train, w, fit_intercept=fit_intercept, linlearn=True)
def linlearn_test_loss(w): return objective(X_test, y_test, w, fit_intercept=fit_intercept, linlearn=True)
linlearn_tracked_funs = [linlearn_train_loss, linlearn_test_loss]

col_try, col_algo, col_train_err, col_test_err = [], [], [], []

for rep in range(n_repeats):
    if not save_results:
        logging.info("WARNING : results will NOT be saved at the end of this session")
    logging.info(64*'-')
    logging.info("repeat : %d" % (rep+1))
    logging.info(64*'-')


    outputs = {}
    w, track = SVRG(X, y, objective, gradient, np.zeros((X.shape[1]+int(fit_intercept), 9)), 0.01, 10, max_iter, fit_intercept=fit_intercept)

    #logging.info("Running regression algorithms ...")

    for algo in algorithms:
        col_val.append()
        col_algo.append(algo.__name__)
        col_try.append(rep)


data = pd.DataFrame({"repeat":col_try, "algorithm":col_algo, "train_err":col_train_err, "test_err":col_test_err})

if save_results:
    logging.info("Saving results ...")
    now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

    filename = "classif_results_" + now + ".pickle"
    with open("exp_archives/classif/" + filename, "wb") as f:
        pickle.dump({"datetime": now, "results": data}, f)

    logging.info("Saved results in file %s" % filename)

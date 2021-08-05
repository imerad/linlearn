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

from tick.linear_model import ModelLogReg
from tick.solver import SVRG
from tick.prox import ProxL1
from tick.plot import plot_history

# file_handler = logging.FileHandler(filename='exp_archives/classif_exp.log')
# stdout_handler = logging.StreamHandler(sys.stdout)
# handlers = [file_handler, stdout_handler]
#
# logging.basicConfig(
#     level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S", handlers=handlers
# )
#
# save_results = False
#
# logging.info(128*"=")
# logging.info("Running new experiment session")
# logging.info(128*"=")

# if not save_results:
#     logging.info("WARNING : results will NOT be saved at the end of this session")

def _images(path):
    """Return images loaded locally."""
    with gzip.open(path) as f:
        # First 16 bytes are magic_number, n_imgs, n_rows, n_cols
        pixels = np.frombuffer(f.read(), 'B', offset=16)
    return pixels.reshape(-1, 784).astype('float32') / 255


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

mnist_train_images = _images(mnist_train_images_file)
mnist_train_labels = _labels(mnist_train_labels_file)

mnist_test_images = _images(mnist_test_images_file)
mnist_test_labels = _labels(mnist_test_labels_file)


save_results = False
max_iter = 150

logging.info(128*"=")
logging.info("Running new experiment session")
logging.info(128*"=")

if not save_results:
    logging.info("WARNING : results will NOT be saved at the end of this session")

n_repeats = 10

def MOM_classifier():
    pass


def SVRG(X, y, obj, grad, w0, step, m, T):
    w_tilde = w0
    wt = w0
    track_objective = [obj(X, y, w0)]
    for i in range(T//m):
        mu = grad(X, y, w_tilde)
        for j in range(m):
            ind = np.random.randint(X.shape[0])
            wt -= step*(grad(X[ind:ind+1,:], y[ind:ind+1,:], wt) - grad(X[ind:ind+1,:], y[ind:ind+1,:], w_tilde) + mu)
            track_objective.append(obj(X, y, wt))
        w_tilde = wt
    return w_tilde, track_objective

def objective(X, y, w, fit_intercept=False):
    scores = X @ w[1:,:] + w[0,:] if fit_intercept else X @ w
    scores = np.hstack((scores, np.zeros((X.shape[0], 1))))
    obj = (-scores[np.arange(X.shape[0]), np.argmax(y, axis=1)] + logsumexp(scores, axis=1)).mean()
    return obj

def gradient(X, y, w):
    scores = X @ w # X @ w[1:,:] + w[0,:]
    scores = np.hstack((scores, np.zeros((X.shape[0], 1))))
    sftmax = softmax(scores, axis=1) - np.hstack((y, np.zeros(X.shape[0])))
    return (X.T @ sftmax[:,:-1])/X.shape[0] # np.vstack((np.ones((X.shape[0], 1)) @ sftmax[:,:-1], X.T @ sftmax[:,:-1]

algorithms = [MOM_classifier]

col_try, col_noise, col_algo, col_val, col_n_samples = [], [], [], [], []
#
# for rep in range(n_repeats):
#     if not save_results:
#         logging.info("WARNING : results will NOT be saved at the end of this session")
#     logging.info(128*'-')
#     logging.info("repeat : %d" % (rep+1))
#     logging.info(128*'-')
#
#
#     outputs = {}
#
#     #logging.info("Running regression algorithms ...")
#
#     for algo in algorithms:
#         col_val.append()
#         col_algo.append(algo.__name__)
#         col_noise.append(noise)
#         col_try.append(rep)
#
#
# data = pd.DataFrame({"repeat":col_try, "noise": col_noise, "algorithm":col_algo, "MSE":col_val, "sample_size":col_n_samples})
#
# if save_results:
#     logging.info("Saving results ...")
#     now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
#
#     filename = "linreg2_results_" + now + ".pickle"
#     with open("exp_archives/linreg2/" + filename, "wb") as f:
#         pickle.dump({"datetime": now, "results": data}, f)
#
#     logging.info("Saved results in file %s" % filename)

mom_reg = MultiClassifier(tol=1e-17, max_iter=max_iter, strategy="mom",
                             thresholding=False, step_size=0.1, loss="multilogistic", penalty="none")

n_samples = 100
X = mnist_train_images[:n_samples,:]
y = mnist_train_labels[:n_samples,:]

mom_reg.fit(X, y)
mom_pred = mom_reg.predict(mnist_train_images[:n_samples,:])


#############################@
model = ModelLogReg(fit_intercept=False).fit(X, np.argmax(y, axis=1))
#prox = ProxElasticNet(strength=1e-3, ratio=0.5, range=(0, X.shape[1]))

solver_params = {'max_iter': max_iter, 'tol': 0., 'verbose': True}
x0 = np.zeros(model.n_coeffs)
print(model.n_coeffs)
prox = ProxL1(strength=0.001)
svrg = SVRG(**solver_params).set_model(model).set_prox(prox)
svrg.solve(x0, step=1 / model.get_lip_max())

#print(svrg.solution.shape)
#plot_history([svrg], log_scale=True, dist_min=True)


correct = 0
# for i in range(n_samples):
#     if pred[i] == np.argmax(mnist_train_labels[i, :]):
#         correct += 1
# print("mom accuracy : %f" % (correct/n_samples))
svrg_pred = np.argmax(X.dot(svrg.solution), axis=1)

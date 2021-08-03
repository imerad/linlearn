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

logging.info(128*"=")
logging.info("Running new experiment session")
logging.info(128*"=")

if not save_results:
    logging.info("WARNING : results will NOT be saved at the end of this session")

n_repeats = 10

def MOM_classifier():
    pass

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

mom_reg = MultiClassifier(tol=1e-17, max_iter=10, fit_intercept=False, strategy="mom",
                             thresholding=False, step_size=0.01, loss="multilogistic", penalty="none")

mom_reg.fit(mnist_train_images[:100,:], mnist_train_labels[:100,:])

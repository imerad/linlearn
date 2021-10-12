from linlearn import Classifier
import numpy as np
import logging
import pickle
from datetime import datetime
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from scipy.special import logsumexp
import itertools
from collections import namedtuple
from linlearn.estimator.ch import holland_catoni_estimator
from linlearn.estimator.tmean import fast_trimmed_mean
from linlearn._loss import median_of_means
import joblib
import argparse
from data_loaders import load_aus, load_stroke, load_heart, load_adult, load_htru2, load_bank, load_mnist, load__iris, load_simulated


def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

experiment_logfile = "exp_archives/logitclassif_exp.log"
experiment_name = "logitclassif"

ensure_directory("exp_archives/")

file_handler = logging.FileHandler(filename=experiment_logfile)
stdout_handler = logging.StreamHandler(sys.stdout)
handlers = [file_handler, stdout_handler]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=handlers,
)

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default="Stroke", choices=["Stroke", "Bank", "Heart", "Adult", "weatherAUS", "htru2", "MNIST", "iris", "simulated"])
parser.add_argument("--loss", type=str, default="logistic", choices=["logistic", "squaredhinge"])
parser.add_argument("--penalty", type=str, default="none", choices=["none", "l1", "l2", "elasticnet"])
parser.add_argument("--lamda", type=float, default=1.0)
parser.add_argument("--tol", type=float, default=0.0001)
parser.add_argument("--step_size", type=float, default=1.0)
parser.add_argument("--test_size", type=float, default=0.3)
parser.add_argument("--block_size", type=float, default=0.07)
parser.add_argument("--percentage", type=float, default=0.01)
parser.add_argument("--l1_ratio", type=float, default=0.5)
parser.add_argument("--meantype", type=str, default="mom", choices=["ordinary", "mom", "tmean", "ch"])
parser.add_argument("--random_seed", type=int, default=43)
parser.add_argument("--n_samples", type=int, default=10000)  # for simulated data
parser.add_argument("--n_features", type=int, default=20)  # for simulated data
parser.add_argument("--n_classes", type=int, default=5)  # for simulated data
parser.add_argument("--n_repeats", type=int, default=1)
parser.add_argument("--max_iter", type=int, default=300)

args = parser.parse_args()

logging.info(64 * "=")
logging.info("Running new experiment session")
logging.info(64 * "=")

loss = args.loss

n_repeats = args.n_repeats
random_state = args.random_seed
max_iter = args.max_iter
fit_intercept = True

block_size = args.block_size
percentage = args.percentage
test_size = args.test_size

n_samples = args.n_samples
n_features = args.n_features
n_classes = args.n_classes

meantype = args.meantype

dataset = args.dataset
penalty = args.penalty
lamda = args.lamda  # /np.sqrt(X_train.shape[0])
tol = args.tol
step_size = args.step_size
l1_ratio = args.l1_ratio

if dataset in ["MNIST", "iris"] or (dataset == "simulated" and n_classes > 2):
    binary = False
else:
    binary = True

logging.info("Received parameters : \n %r" % args)


def load_dataset(dataset):

    if dataset == "Bank":
        X_train, X_test, y_train, y_test = load_bank(test_size=test_size, random_state=random_state)
    elif dataset == "Adult":
        X_train, X_test, y_train, y_test = load_adult(test_size=test_size, random_state=random_state)
    elif dataset == "Heart":
        X_train, X_test, y_train, y_test = load_heart(test_size=test_size, random_state=random_state)
    elif dataset == "Stroke":
        X_train, X_test, y_train, y_test = load_stroke(test_size=test_size, random_state=random_state)
    elif dataset == "weatherAUS":
        X_train, X_test, y_train, y_test = load_aus(test_size=test_size, random_state=random_state)
    elif dataset == "htru2":
        X_train, X_test, y_train, y_test = load_htru2(test_size=test_size, random_state=random_state)
    elif dataset == "MNIST":
        X_train, X_test, y_train, y_test = load_mnist(test_size=test_size, random_state=random_state)
    elif dataset == "iris":
        X_train, X_test, y_train, y_test = load__iris(test_size=test_size, random_state=random_state)
    elif dataset == "simulated":
        X_train, X_test, y_train, y_test = load_simulated(n_samples, n_features, n_classes, test_size=test_size, random_state=random_state)
    else:
        ValueError("unknown dataset")

    std_scaler = StandardScaler()

    X_train = std_scaler.fit_transform(X_train)
    X_test = std_scaler.transform(X_test)

    if binary:
        y_train = 2 * y_train - 1
        y_test = 2 * y_test - 1
    for dat in [X_train, X_test, y_train, y_test]:
        dat = np.ascontiguousarray(dat)
    print("n_features : %d"%X_train.shape[1])

    return X_train, X_test, y_train, y_test


logging.info(
    "Parameters are : loss = %s , n_repeats = %d , max_iter = %d , fit_intercept=%r, meantype = %s"
    % (loss, n_repeats, max_iter, fit_intercept, meantype)
)


def l1_penalty(x):
    return np.sum(np.abs(x))

def l2_penalty(x):
    return 0.5 * np.sum(x ** 2)

def elasticnet_penalty(x):
    return l1_ratio * l1_penalty(x) + (1.0 - l1_ratio) * l2_penalty(x)


penalties = {"l1": l1_penalty, "l2": l2_penalty, "elasticnet": elasticnet_penalty}

def logit(x):
    if x > 0:
        return np.log(1 + np.exp(-x))
    else:
        return -x + np.log(1 + np.exp(x))

vec_logit = np.vectorize(logit)


def objective(X, y, clf, meantype=meantype, block_size=block_size, percentage=0.01):
    if binary:
        sample_objectives = vec_logit(clf.decision_function(X) * y)
    else:
        scores = clf.decision_function(X)
        sample_objectives = -scores[np.arange(X.shape[0]), y] + logsumexp(
            scores, axis=1
        )

    if meantype == "ordinary":
        obj = sample_objectives.mean()
    elif meantype == "ch":
        obj = holland_catoni_estimator(sample_objectives)
    elif meantype == "mom":
        obj = median_of_means(sample_objectives, int(block_size * len(sample_objectives)))
    elif meantype == "tmean":
        obj = fast_trimmed_mean(sample_objectives, len(sample_objectives), percentage)
    else:
        raise ValueError("unknown mean")
    if penalty != "none":
        obj += lamda * penalties[penalty](clf.coef_)
    return obj


def accuracy(X, y, clf, meantype=meantype, block_size=block_size, percentage=0.01):
    if binary:
        scores = clf.decision_function(X)#clf.predict(X)
        decisions = ((y * scores) > 0).astype(int).astype(float)
    else:
        predictions = clf.predict(X)
        decisions = (y == predictions).astype(int).astype(float)

    if meantype == "ordinary":
        acc = decisions.mean()
    elif meantype == "ch":
        acc = holland_catoni_estimator(decisions)
    elif meantype == "mom":
        acc = median_of_means(decisions, int(block_size * len(decisions)))
    elif meantype == "tmean":
        acc = fast_trimmed_mean(decisions, len(decisions), percentage=percentage)
    else:
        raise ValueError("unknown mean")
    return acc


Algorithm = namedtuple("Algorithm", ["name", "solver", "estimator", "max_iter"])

algorithms = [
    Algorithm(
        name="batch_gd", solver="batch_gd", estimator="erm", max_iter=3 * max_iter
    ),
    # Algorithm(name="saga", solver="saga", estimator="erm", max_iter=3 * max_iter),
    Algorithm(name="mom_cgd", solver="cgd", estimator="mom", max_iter=2 * max_iter),
    Algorithm(name="mom_cgd_IS", solver="cgd", estimator="mom", max_iter=2 * max_iter),
    Algorithm(name="erm_cgd", solver="cgd", estimator="erm", max_iter=3 * max_iter),
    # Algorithm(
    #    name="catoni_cgd", solver="cgd", estimator="ch", max_iter=max_iter
    # ),
    Algorithm(name="tmean_cgd", solver="cgd", estimator="tmean", max_iter=max_iter),
    # Algorithm(name="gmom_gd", solver="gd", estimator="gmom", max_iter=3 * max_iter),
    Algorithm(
        name="implicit_gd", solver="gd", estimator="llm", max_iter=9 * max_iter
    ),
    Algorithm(name="erm_gd", solver="gd", estimator="erm", max_iter=5 * max_iter),
    # Algorithm(
    #    name="holland_gd", solver="gd", estimator="ch", max_iter=max_iter
    # ),
    # Algorithm(name="svrg", solver="svrg", estimator="erm", max_iter=2 * max_iter),
    Algorithm(name="sgd", solver="sgd", estimator="erm", max_iter=4 * max_iter),
]



def announce(rep, x, status):
    logging.info(str(rep) + " : " + x + " " + status)

def run_algorithm(data, algo, rep, col_try, col_algo, col_metric, col_val, col_time):

    X_train, X_test, y_train, y_test = data
    n_samples = len(y_train)
    announce(rep, algo.name, "running")
    clf = Classifier(
        tol=tol,
        max_iter=max_iter,
        solver=algo.solver,
        loss=loss,
        estimator=algo.estimator,
        fit_intercept=fit_intercept,
        step_size=step_size,
        penalty=penalty,
        cgd_IS=algo.name[-2:] == "IS",
        l1_ratio=l1_ratio,
        C=1/(n_samples * lamda),
    )

    clf.fit(X_train, y_train, dummy_first_step=True)
    announce(rep, algo.name, "fitted")
    clf.compute_objective_history(X_train, y_train)
    clf.compute_objective_history(X_test, y_test)
    announce(rep, algo.name, "computed history")

    records = clf.history_.records[1:]

    for j, metric in enumerate(["train_loss", "test_loss"]):
        # for i in range(len(records[0])):
        for i in range(records[0].cursor):
            col_try.append(rep)
            col_algo.append(algo.name)
            col_metric.append(metric)
            col_val.append(records[1+j].record[i])
            col_time.append(records[0].record[i] - records[0].record[0])#i)#


def run_repetition(rep):
    col_try, col_algo, col_metric, col_val, col_time = [], [], [], [], []
    data = load_dataset(dataset)
    for algo in algorithms:
        run_algorithm(data, algo, rep, col_try, col_algo, col_metric, col_val, col_time)

    logging.info("repetition done")
    return col_try, col_algo, col_metric, col_val, col_time

if os.cpu_count() > 8:
    logging.info("running parallel repetitions")
    results = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(run_repetition)(rep) for rep in range(1, n_repeats+1)
    )
else:
    results = [run_repetition(rep) for rep in range(1, n_repeats+1)]


col_try = list(itertools.chain.from_iterable([x[0] for x in results]))
col_algo = list(itertools.chain.from_iterable([x[1] for x in results]))
col_metric = list(itertools.chain.from_iterable([x[2] for x in results]))
col_val = list(itertools.chain.from_iterable([x[3] for x in results]))
col_time = list(itertools.chain.from_iterable([x[4] for x in results]))


data = pd.DataFrame(
    {
        "repeat": col_try,
        "algorithm": col_algo,
        "metric": col_metric,
        "value": col_val,
        "time": col_time,
    }
)

# save_results:
logging.info("Saving results ...")
now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

filename = experiment_name + "_" + dataset + "_results_" + now + ".pickle"
ensure_directory("exp_archives/"+ experiment_name + "/")
with open("exp_archives/" + experiment_name + "/" + filename, "wb") as f:
    pickle.dump({"datetime": now, "results": data}, f)

logging.info("Saved results in file %s" % filename)


g = sns.FacetGrid(data, col="metric", height=4, legend_out=True, sharey=False)
g.map(
    sns.lineplot,
    "time",
    "value",
    "algorithm",
    # lw=4,
)  # .set(yscale="log")#, xlabel="", ylabel="")

# g.set_titles(col_template="{col_name}")

# g.set(ylim=(0, 1))
axes = g.axes.flatten()

# for ax in axes:
#     ax.set_title("")

# _, y_high = axes[2].get_ylim()
# axes[2].set_ylim([0.75, y_high])

# for i, dataset in enumerate(df["dataset"].unique()):
#     axes[i].set_xticklabels([0, 1, 2, 5, 10, 20, 50], fontsize=14)
#     axes[i].set_title(dataset, fontsize=18)


plt.legend(
    list(data["algorithm"].unique()),
    # bbox_to_anchor=(0.3, 0.7, 1.0, 0.0),
    loc="upper right",
    # ncol=1,
    # borderaxespad=0.0,
    # fontsize=14,
)
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle(
    "data : %s , loss=%s, meantype=%s"
    % (dataset, loss.upper(), meantype)
)

plt.show()


# save figure :
now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
ensure_directory("exp_archives/" + experiment_name + "/")
specs = "%s_nrep=%d_meantype=%s_" % (dataset, n_repeats, meantype)
fig_file_name = "exp_archives/" + experiment_name + "/" + specs + now + ".pdf"
g.fig.savefig(fname=fig_file_name)  # , bbox_inches='tight')
logging.info("Saved figure into file : %s" % fig_file_name)

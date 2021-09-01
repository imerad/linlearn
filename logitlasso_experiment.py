from linlearn import BinaryClassifier
from linlearn.robust_means import Holland_catoni_estimator, gmom, gmom_njit, median_of_means
import numpy as np
import logging
import pickle
from datetime import datetime
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import itertools
from tqdm import tqdm
import joblib
import time
from sklearn.metrics import accuracy_score

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

step_size = 0.1

random_state = 43

max_iter = 20
fit_intercept = True

MOM_block_size = 0.1

test_loss_meantype = "ordinary"

n_samples = None
n_repeats = 1

logging.info("Parameters are : n_repeats = %d , n_samples = %d , max_ter = %d , fit_intercept=%r, MOM_block_size = %.2f, test_loss_meantype = %s" % (n_repeats, n_samples or 0, max_iter, fit_intercept, MOM_block_size, test_loss_meantype))

if not save_results:
    logging.info("WARNING : results will NOT be saved at the end of this session")

logging.info("loading data ...")

dataset = "Bank"#"Stroke"#"Heart"#"weatherAUS"#"Adult"#

def load_heart(test_size=0.3):
    csv_heart = pd.read_csv("heart/heart.csv")
    categoricals = ["sex", "cp", "fbs", "restecg", "exng", "slp", "caa", "thall"]
    label = "output"
    for cat in categoricals:
        one_hot = pd.get_dummies(csv_heart[cat], prefix=cat)
        csv_heart = csv_heart.drop(cat, axis=1)
        csv_heart = csv_heart.join(one_hot)
    df_train, df_test = train_test_split(
        csv_heart,
        test_size=test_size,
        shuffle=True,
        random_state=random_state,
        stratify=csv_heart[label],
    )
    y_train = df_train.pop(label)
    y_test = df_test.pop(label)
    return df_train.to_numpy(), df_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()

def load_stroke(test_size=0.3):
    csv_stroke = pd.read_csv("stroke/healthcare-dataset-stroke-data.csv").drop("id", axis=1).dropna()

    categoricals = ["gender", "hypertension", "heart_disease",
                    "ever_married", "work_type", "Residence_type", "smoking_status"]
    label = "stroke"
    for cat in categoricals:
        one_hot = pd.get_dummies(csv_stroke[cat], prefix=cat)
        csv_stroke = csv_stroke.drop(cat, axis=1)
        csv_stroke = csv_stroke.join(one_hot)
    df_train, df_test = train_test_split(
        csv_stroke,
        test_size=test_size,
        shuffle=True,
        random_state=random_state,
        stratify=csv_stroke[label],
    )
    y_train = df_train.pop(label)
    y_test = df_test.pop(label)
    return df_train.to_numpy(), df_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()

def load_aus(test_size=0.3):
    #drop columns with too many NA values and Location because it has too many categories
    csv_aus = pd.read_csv("weatherAUS/weatherAUS.csv").drop(["Sunshine", "Evaporation", "Cloud9am", "Cloud3pm", "Location"], axis=1)
    #convert date to yearday
    csv_aus = csv_aus.rename(columns={"Date": "yearday"})
    csv_aus["yearday"] = csv_aus.yearday.apply(lambda x: 30 * (int(x[5:7]) - 1) + int(x[8:])).astype(int)
    categoricals = ['WindGustDir','WindDir9am','WindDir3pm']
    label = "RainTomorrow"
    for cat in categoricals:
        one_hot = pd.get_dummies(csv_aus[cat], prefix=cat)
        csv_aus = csv_aus.drop(cat, axis=1)
        csv_aus = csv_aus.join(one_hot)

    csv_aus['RainToday'] = LabelBinarizer().fit_transform(csv_aus['RainToday'].astype('str'))
    csv_aus['RainTomorrow'] = LabelBinarizer().fit_transform(csv_aus['RainTomorrow'].astype('str'))
    csv_aus = csv_aus.dropna()
    df_train, df_test = train_test_split(
        csv_aus,
        test_size=test_size,
        shuffle=True,
        random_state=random_state,
        stratify=csv_aus[label],
    )
    y_train = df_train.pop(label)
    y_test = df_test.pop(label)
    return df_train.to_numpy(), df_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()

if dataset in ["Adult", "Bank"]:
    with open("pickled_%s.pickle"% dataset.lower(), "rb") as f:
        X_train, X_test, y_train, y_test = pickle.load(f)
    f.close()
elif dataset == "Heart":
    X_train, X_test, y_train, y_test = load_heart()
elif dataset == "Stroke":
    X_train, X_test, y_train, y_test = load_stroke()
elif dataset == "weatherAUS":
    X_train, X_test, y_train, y_test = load_aus()
else:
    ValueError("unknown dataset")

y_train = 2 * y_train - 1
y_test = 2 * y_test - 1
for dat in [X_train, X_test, y_train, y_test]:
    dat = np.ascontiguousarray(dat)

#The below estimation is probably too sensitive to outliers
#Lip = np.linalg.eigh((X_train.T @ X_train)/X_train.shape[0])[0][-1] # highest eigenvalue
Lip = np.max([median_of_means(X_train[:,j]**2, int(MOM_block_size*X_train.shape[0])) for j in range(X_train.shape[1])])

Lip_step = 1/(0.25*Lip) # 0.25 is Lipschitz smoothness of logistic loss

penalty = None#"l2"
lamda = 0.01#1/np.sqrt(X_train.shape[0])

def l1_penalty(x):
    return np.sum(np.abs(x))

def l2_penalty(x):
    return np.sum(x ** 2)

def l1_apply_single(x, t):
    if x > t:
        return x - t
    elif x < -t:
        return x + t
    else:
        return 0.0

def l1_apply(x, t):
    for j in range(len(x)):
        x[j] = l1_apply_single(x[j], t)

def l2_apply(x, t):
    x /= (1 + t)

penalties = {"l1" : l1_penalty, "l2" : l2_penalty}
penalties_apply = {"l1" : l1_apply, "l2" : l2_apply}


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


def sample_objectives(X, y, w, fit_intercept=fit_intercept, lnlearn=False):
    if fit_intercept:
        w0 = w[0] if lnlearn else w[0]
        w1 = w[1] if lnlearn else w[1:]
    else:
        w0 = 0
        w1 = w
    scores = X @ w1 + w0
    return vec_logit(y*scores)

def objective(X, y, w, fit_intercept=fit_intercept, lnlearn=False, meantype="ordinary"):
    objectives = sample_objectives(X, y, w, fit_intercept=fit_intercept, lnlearn=lnlearn)
    if meantype == "ordinary":
        obj = objectives.mean()
    elif meantype == "catoni":
        obj = Holland_catoni_estimator(objectives)
    elif meantype == "mom":
        obj = median_of_means(objectives, int(MOM_block_size*len(objectives)))
    else:
        ValueError("unknown mean")
    if penalty:
        if fit_intercept:
            if lnlearn:
                return obj + lamda * penalties[penalty](w[1])
            else:
                return obj + lamda * penalties[penalty](w[1:])
        else:
            return obj + lamda * penalties[penalty](w)
    else:
        return obj

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

def accuracy(X, y, w, fit_intercept=fit_intercept, lnlearn=False):
    if fit_intercept:
        w0 = w[0] if lnlearn else w[0]
        w1 = w[1] if lnlearn else w[1:]
    else:
        w0 = 0
        w1 = w
    scores = X @ w1 + w0
    return ((y*scores) > 0).astype(int).mean()


def train_loss(w, algo_name=""):
    return objective(X_train, y_train, w, fit_intercept=fit_intercept, lnlearn=algo_name in linlearn_algorithms)
def test_loss(w, algo_name=""):
    return objective(X_test, y_test, w, fit_intercept=fit_intercept, lnlearn=algo_name in linlearn_algorithms, meantype=test_loss_meantype)

def test_accuracy(w, algo_name=""):
    return accuracy(X_test, y_test, w, fit_intercept=fit_intercept, lnlearn=algo_name in linlearn_algorithms)


def linlearn_cgd(X_train, y_train, strategy="mom"):

    mom_logreg = BinaryClassifier(tol=1e-17, max_iter=max_iter, strategy=strategy, fit_intercept=fit_intercept, penalty=penalty or "none", C=1/(X_train.shape[0] * lamda) if penalty else 1.0,
                              step_size=step_size, loss="logistic", block_size=MOM_block_size)
    param_record = Record((X_train.shape[1]+int(fit_intercept),), max_iter)
    time_record = Record(1, max_iter)

    # param_record.update(np.zeros(X_train.shape[1]+int(fit_intercept)))
    # time_record.update(time.time())

    mom_logreg.fit(X_train, y_train, trackers=[lambda w: param_record.update(np.vstack(w).flatten()), lambda _:time_record.update(time.time())])

    return param_record, time_record

#@njit
def SAGA(X, y, w0=None, fit_intercept=fit_intercept):
    if w0 is None:
        wt = np.zeros(X.shape[1] + int(fit_intercept))
    else:
        wt = w0

    param_record = Record((X.shape[1]+int(fit_intercept),), max_iter)
    time_record = Record(1, max_iter)

    table = X @ wt[1:] + wt[0] if fit_intercept else X @ wt
    table = -y * vec_sigmoid(-y * table)
    if fit_intercept:
        mean_grad = np.hstack((table[:,np.newaxis], np.einsum("i,ij->ij", table, X))).mean(axis=0)
    else:
        mean_grad = (table * X)

    step = step_size*Lip_step/X.shape[0]

    for i in tqdm(range(max_iter), desc="SAGA"):
        for sam in range(X.shape[0]):
            j = np.random.randint(X.shape[0])
            new_j_score = X[j,:].dot(wt[1:]) + wt[0] if fit_intercept else X[j,:].dot(wt)
            new_j_der = -y[j] * sigmoid(-y[j] * new_j_score)
            #Xj  = (np.concatenate((np.array([1]), X[j,:])) if fit_intercept else X[j,:])
            Xj = (np.insert(X[j, :], 0, 1) if fit_intercept else X[j, :])
            j_grad_diff = (new_j_der - table[j]) * Xj
            wt -= step * (j_grad_diff + mean_grad)
            mean_grad += j_grad_diff*Xj/X.shape[0]
            table[j] = new_j_der

            if penalty:
                penalties_apply[penalty](wt[int(fit_intercept):], lamda * step)
            # for k in range(int(fit_intercept), len(wt)):
            #     wt[k] = l1_apply_single(wt[k], l1_penalty*step)
        param_record.update(wt)
        time_record.update(time.time())

    return param_record, time_record


def SVRG(X, y, w0=None, T=max_iter, fit_intercept=fit_intercept):
    if w0 is None:
        w0 = np.zeros((X.shape[1] + int(fit_intercept)))
    w_tilde = w0
    wt = w0
    step = step_size*Lip_step/(X.shape[0])
    m = X.shape[0]

    param_record = Record((X.shape[1]+int(fit_intercept),), max_iter)
    time_record = Record(1, max_iter)

    for i in tqdm(range(T), desc="SVRG"):
        mu = gradient(X, y, w_tilde, fit_intercept=fit_intercept)

        for j in range(m):
            ind = np.random.randint(X.shape[0])
            X_ind, y_ind = X[ind:ind+1,:], y[ind:ind+1]
            wt -= step*(gradient(X_ind, y_ind, wt, fit_intercept=fit_intercept) - gradient(X_ind, y_ind, w_tilde, fit_intercept=fit_intercept) + mu)
            if penalty:
                penalties_apply[penalty](wt[int(fit_intercept):], lamda * step)
        param_record.update(wt)
        time_record.update(time.time())

        w_tilde = wt

    return param_record, time_record



def Prasad_heavyTails_gd(X, y, w0=None, T=max_iter, fit_intercept=fit_intercept, delta=0.01):
    if w0 is None:
        w0 = np.zeros((X.shape[1] + int(fit_intercept)))
    n_blocks = 1 + int(3.5 * np.log(1 / delta))
    block_size = X.shape[0] // n_blocks
    wt = w0

    param_record = Record((X.shape[1]+int(fit_intercept),), max_iter)
    time_record = Record(1, max_iter)
    step = step_size*Lip_step

    for i in tqdm(range(T), desc = "prasad_heavy_tail"):
        gradients = sample_gradients(X, y, wt, fit_intercept=fit_intercept)
        permutation = np.random.permutation(X.shape[0])
        block_means = []
        for j in range(n_blocks):
            block_means.append(np.mean(gradients[permutation[j * block_size:(j + 1) * block_size], :], axis=0).reshape(-1))
        grad = gmom(np.array(block_means)).reshape(wt.shape)
        wt -= step * grad

        if penalty:
            penalties_apply[penalty](wt[int(fit_intercept):], lamda * step)

        # for k in range(int(fit_intercept), len(wt)):
        #     wt[k] = l1_apply_single(wt[k], lamda * step_size)

        param_record.update(wt)
        time_record.update(time.time())

    return param_record, time_record

def Lecue_gd(X, y, w0=None, T=max_iter, fit_intercept=fit_intercept, n_blocks=21):
    """n_blocks must be uneven"""
    if w0 is None:
        w0 = np.zeros((X.shape[1] + int(fit_intercept)))

    def argmedian(x):
        return np.argpartition(x, len(x) // 2)[len(x) // 2]
    block_size = X.shape[0] // n_blocks
    wt = w0
    param_record = Record((X.shape[1]+int(fit_intercept),), max_iter)
    time_record = Record(1, max_iter)
    step = step_size*Lip_step

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
        wt -= step * grad
        if penalty:
            penalties_apply[penalty](wt[int(fit_intercept):], lamda * step)

        param_record.update(wt)
        time_record.update(time.time())

    return param_record, time_record

def Holland_gd(X, y, w0=None, T=max_iter, fit_intercept=fit_intercept):
    if w0 is None:
        w0 = np.zeros((X.shape[1] + int(fit_intercept)))
    wt = w0
    param_record = Record((X.shape[1]+int(fit_intercept),), max_iter)
    time_record = Record(1, max_iter)
    catoni_avg_grad = np.zeros_like(wt)
    step = step_size*Lip_step
    for i in range(10):
        Holland_catoni_estimator(np.random.randn(100))

    for i in tqdm(range(T), desc="Holland"):
        gradients = sample_gradients(X, y, wt, fit_intercept=fit_intercept)
        for k in range(len(wt)):
            catoni_avg_grad[k] = Holland_catoni_estimator(gradients[:, k])

        wt -= step * catoni_avg_grad
        if penalty:
            penalties_apply[penalty](wt[int(fit_intercept):], lamda * step)

        param_record.update(wt)
        time_record.update(time.time())

    return param_record, time_record



metrics = [train_loss, test_loss, test_accuracy]

def run_repetition(rep):
    col_try, col_algo, col_metric, col_val, col_time = [], [], [], [], []

    outputs = {}
    def announce(x):
        logging.info(str(rep)+" : "+x+" done")

    outputs["mom_cgd"] = linlearn_cgd(X_train, y_train, strategy="mom")
    announce("mom_cgd")

    # outputs["erm_cgd"] = linlearn_cgd(X_train, y_train, strategy="erm")
    # announce("erm_cgd")

    outputs["catoni_cgd"] = linlearn_cgd(X_train, y_train, strategy="catoni")
    announce("catoni_cgd")

    outputs["Prasad_heavyTails_gd"] = Prasad_heavyTails_gd(X_train, y_train)
    announce("Prasad_heavyTails_gd")

    outputs["Lecue_gd"] = Lecue_gd(X_train, y_train)
    announce("Lecue_gd")

    outputs["Holland"] = Holland_gd(X_train, y_train)
    announce("Holland")

    # outputs["SAGA"] = SAGA(X_train, y_train)
    # announce("SAGA")
    #
    # outputs["SVRG"] = SVRG(X_train, y_train)
    # announce("SVRG")

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
    data, col="metric", height=4, legend_out=True, sharey=False
)
g.map(
    sns.lineplot,
    "time",
    "value",
    "algorithm",
    #lw=4,
)#.set(yscale="log")#, xlabel="", ylabel="")

#g.set_titles(col_template="{col_name}")

#g.set(ylim=(0, 1))

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
    g.fig.savefig(fname=fig_file_name)#, bbox_inches='tight')
    logging.info("Saved figure into file : %s" % fig_file_name)

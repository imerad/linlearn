from linlearn import MOMRegressor
from linlearn.catoni import Holland_catoni_estimator, standard_catoni_estimator, estimate_sigma
import numpy as np
import logging
import pickle
from datetime import datetime
from scipy.optimize import minimize
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

file_handler = logging.FileHandler(filename='exp_archives/linreg_exp.log')
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

n_repeats = 5#10

n_samples = 200
n_features = 5

n_outliers = 10
outliers = False

mom_thresholding = True
mom_thresholding = False
mom_K = 20
adamom_K_init = 20

catoni_thresholding = True
catoni_thresholding = False

random_seed = 42

noise_sigma = {"gaussian": 20, "lognormal": 1.75, "pareto": 10}

Sigma_X = np.diag(np.arange(1, n_features+1))
mu_X = np.ones(n_features)

w_star_dist = "normal"
noise_dist = "gaussian"

step_size = 0.01
T = 20

logging.info("Lauching experiment with parameters : \n n_repeats = %d , n_samples = %d , n_features = %d , outliers = %r" % (n_repeats, n_samples, n_features, outliers))
if outliers:
    logging.info("n_outliers = %d" % n_outliers)
logging.info("nb blocks for MOM_CGD is %d" % mom_K)
logging.info("initial nb blocks for adaptive MOM is %d" % adamom_K_init)
logging.info("mom_thresholding = %r , random_seed = %d , mu_X = %r , Sigma_X = %r" % (mom_thresholding, random_seed, mu_X, Sigma_X))
logging.info("w_star_dist = %s , noise_dist = %s , sigma = %f" % (w_star_dist, noise_dist, noise_sigma[noise_dist]))
logging.info("step_size = %f , T = %d" % (step_size, T))

rng = np.random.RandomState(random_seed) ## Global random generator

def gen_w_star(d, dist="normal"):
    if dist =="normal":
        return rng.multivariate_normal(np.zeros(d), np.eye(d)).reshape(d)
    elif dist == "uniform":
        return 10 * rng.uniform(size=d) - 5
    else:
        raise Exception("Unknown w_star distribution")


def generate_gaussian_noise_sample(n_samples, sigma=20):
    noise = sigma * rng.normal(size=n_samples)
    expect_noise = 0
    noise_2nd_moment = sigma**2

    return noise, expect_noise, noise_2nd_moment


def generate_lognormal_noise_sample(n_samples, sigma=1.75):
    noise = rng.lognormal(0, sigma, n_samples)
    expect_noise = np.exp(0.5 * sigma**2)
    noise_2nd_moment = np.exp(2 * sigma**2)

    return noise, expect_noise, noise_2nd_moment

def generate_pareto_noise_sample(n_samples, sigma=10, pareto=3):
    noise = sigma * rng.pareto(pareto, n_samples)
    expect_noise = (sigma)/(pareto - 1)
    noise_2nd_moment = expect_noise**2 + (sigma**2) * pareto/(((pareto-1)**2)*(pareto-2))

    return noise, expect_noise, noise_2nd_moment

def Holland_gradient(w):
    """compute the grandient used by Holland et al."""
    sample_gradients = np.multiply((X @ w - y)[:,np.newaxis], X)
    catoni_avg_grad = np.zeros_like(w)
    for i in range(w.shape[0]):
        catoni_avg_grad[i] = Holland_catoni_estimator(sample_gradients[:,i])
    return catoni_avg_grad

def catoni_cgd_descent(funs_to_track, x0, step_size, T, steps=None):
    """run coordinate gradient descent using catoni estimates for the gradient coordinates instead of MOM"""
    x = x0
    if steps is None:
        steps = np.ones(len(x0))
    tracks = [np.zeros(T) for i in range(len(funs_to_track)+1)]
    for t in range(T):
        grad_error = 0.0
        for i in np.random.permutation(x0.shape[0]):
            sample_gradients = np.multiply((X @ x - y)[:, np.newaxis], X)
            grad_i = Holland_catoni_estimator(sample_gradients[:,i])
            grad_error += (grad_i - true_gradient(x)[i])**2
            x[i] -= step_size * steps[i] * grad_i

        for i, f in enumerate(funs_to_track):
            tracks[i][t] = f(x)
        tracks[len(funs_to_track)][t] = np.sqrt(grad_error)
    return tracks

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


def adaptive_mom_cgd(funs_to_track, x0, step_size, T, steps=None, K_init=adamom_K_init):
    """a mom cgd algorithm that adapts block sizes coordinatewise according to changes in coord. gradient variance estimates"""
    x = x0
    K = K_init * np.ones(n_features)
    sample_gradients = np.multiply((X @ x - y)[:, np.newaxis], X)
    vars_init = np.array([estimate_sigma(sample_gradients[:,co]) for co in range(n_features)])
    vars = vars_init.copy()
    if steps is None:
        steps = np.ones(len(x0))
    tracks = [np.zeros(T) for i in range(len(funs_to_track)+1)]
    for t in range(T):
        gradient_error = 0.0
        for i in np.random.permutation(x0.shape[0]):
            sample_gradients = np.multiply((X @ x - y)[:, np.newaxis], X)
            vars[i] = estimate_sigma(sample_gradients[:,i])
            K[i] = int(min(n_samples//2, max(K_init*((vars_init[i]/vars[i])**2) , K_init)))
            grad_i = mom_func(sample_gradients[:,i], n_samples // K[i])
            #K[i] = int(min(n_samples//2, max(0.1*n_samples/(vars[i]**2) , K_init)))
            #grad_i = mom_func(sample_gradients[:, i], n_samples // (K_init))
            gradient_error += (grad_i - true_gradient(x)[i])**2

            x[i] -= step_size * steps[i] * grad_i

        for i, f in enumerate(funs_to_track):
            tracks[i][t] = f(x)
        tracks[len(funs_to_track)][t] = np.sqrt(gradient_error)
    return tracks


def mom_cgd(funs_to_track, x0, step_size, T, steps=None, K=mom_K):
    """a function performing mom cgd, I wrote it as shortcut to track gradient error without modifying linlearn
    it is a simplified version of adaptive_mom_cgd"""
    x = x0
    if steps is None:
        steps = np.ones(len(x0))
    tracks = [np.zeros(T) for i in range(len(funs_to_track)+1)]
    for t in range(T):
        gradient_error = 0.0
        for i in np.random.permutation(x0.shape[0]):
            sample_gradients = np.multiply((X @ x - y)[:, np.newaxis], X)
            grad_i = mom_func(sample_gradients[:,i], n_samples // K)

            gradient_error += (grad_i - true_gradient(x)[i])**2
            x[i] -= step_size * steps[i] * grad_i

        for i, f in enumerate(funs_to_track):
            tracks[i][t] = f(x)
        tracks[len(funs_to_track)][t] = np.sqrt(gradient_error)
    return tracks


def gradient_descent(funs_to_track, x0, grad, step_size, T):
    """run gradient descent for given gradient grad and step size for T steps"""
    x = x0
    tracks = [np.zeros(T) for i in range(len(funs_to_track)+1)]
    for t in range(T):
        grad_x = grad(x)
        grad_error = np.linalg.norm(grad_x - true_gradient(x))
        x -= step_size * grad_x
        for i, f in enumerate(funs_to_track):
            tracks[i][t] = f(x)
        tracks[len(funs_to_track)][t] = grad_error
    return tracks

col_try, col_t, col_algo, col_metric, col_val = [], [], [], [], []

if noise_dist=="gaussian":
    noise_fct = generate_gaussian_noise_sample
elif noise_dist=="lognormal":
    noise_fct = generate_lognormal_noise_sample
elif noise_dist=="pareto":
    noise_fct = generate_pareto_noise_sample
else:
    raise Exception("unknown noise dist")

metrics = ["excess_empirical_risk", "excess_risk"]#, "gradient_error"]

logging.info("tracked metrics are %r" % metrics)

for rep in range(n_repeats):
    if not save_results:
        logging.info("WARNING : results will NOT be saved at the end of this session")

    logging.info(128*'-')
    logging.info("repeat : %d" % (rep+1))
    logging.info(128*'-')

    logging.info("generating data ...")
    X = rng.multivariate_normal(mu_X, Sigma_X, size=n_samples)
    w_star = gen_w_star(n_features, dist=w_star_dist)
    noise, expect_noise, noise_2nd_moment = noise_fct(n_samples, noise_sigma[noise_dist])
    y = X @ w_star + noise

    #outliers
    if outliers:
        X = np.concatenate((X, np.ones((n_outliers, n_features))), axis=0)
        y = np.concatenate((y, 10*np.ones(n_outliers)*np.max(np.abs(y))))

    logging.info("generating risks and gradients ...")
    def empirical_risk(w):
        return ((X.dot(w) - y)**2).mean()/2

    def true_risk(w): return 0.5*(noise_2nd_moment + np.dot(mu_X, w - w_star)**2 - 2*expect_noise*np.dot(mu_X, w - w_star) + np.dot(w-w_star, Sigma_X @ (w-w_star)))

    def true_gradient(w): return Sigma_X @ (w-w_star) + (-expect_noise + np.dot(mu_X, w-w_star))*mu_X

    XXT = X.T @ X
    Xy = X.T @ y

    def empirical_gradient(w):
        return (XXT @ w - Xy)/n_samples

    #optimal_risk = true_risk(w_star)
    optimal_risk = minimize(true_risk, np.zeros(n_features), jac=true_gradient).fun
    optimal_empirical_risk = minimize(empirical_risk, np.zeros(n_features), jac=empirical_gradient).fun

    def excess_empirical_risk(w): return empirical_risk(w.flatten()) - optimal_empirical_risk
    def excess_risk(w): return true_risk(w.flatten()) - optimal_risk

    outputs = {}

    logging.info("Running empirical, oracle and Holland gradients ...")

    for gradient in [empirical_gradient, true_gradient, Holland_gradient]:
        outputs[gradient.__name__] = gradient_descent([excess_empirical_risk, excess_risk], np.zeros(n_features), gradient, step_size, T)

    MOMreg_block_size = 0.05
    logging.info("generating cgd step sizes with block size = %f ..." % MOMreg_block_size)
    MOM_regressor = MOMRegressor(tol=1e-17, max_iter=T, fit_intercept=False, strategy="mom", thresholding=mom_thresholding, step_size=step_size, block_size=MOMreg_block_size)
    MOM_regressor.fit(X, y, tracked_funs=[excess_empirical_risk, excess_risk])

    outputs["mom_cgd"] = MOM_regressor.optimization_result_.tracked_funs
    #logging.info("running MOM cgd")
    #outputs["mom_cgd"] = mom_cgd([excess_empirical_risk, excess_risk], np.zeros(n_features), step_size, T, steps=MOM_regressor._steps)

    logging.info("running catoni cgd")
    catoni_regressor = MOMRegressor(tol=1e-17, max_iter=T, fit_intercept=False, strategy="catoni",
                                 thresholding=catoni_thresholding, step_size=step_size)
    catoni_regressor.fit(X, y, tracked_funs=[excess_empirical_risk, excess_risk])
    outputs["catoni_cgd"] = catoni_regressor.optimization_result_.tracked_funs
    #outputs["catoni_cgd"] = catoni_cgd_descent([excess_empirical_risk, excess_risk], np.zeros(n_features), step_size, T, steps=MOM_regressor._steps)
    logging.info("running adaptive MOM cgd")
    outputs["adaptive_mom_cgd"] = adaptive_mom_cgd([excess_empirical_risk, excess_risk], np.zeros(n_features), step_size, T, steps=MOM_regressor._steps)

    logging.info("saving repetition data ...")
    for tt in range(T):
        for alg in outputs.keys():
            for ind_metric, metric in enumerate(metrics):
                col_t.append(tt)
                col_try.append(rep)
                col_algo.append(alg)
                col_metric.append(metric)
                col_val.append(outputs[alg][ind_metric][tt])

data = pd.DataFrame({"t":col_t, "repeat":col_try, "algo":col_algo, "metric":col_metric, "value":col_val})

if save_results:
    now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

    filename = "linreg_results_" + now + ".pickle"
    with open("exp_archives/linreg/" + filename, "wb") as f:
        pickle.dump({"datetime": now, "results": data}, f)

    logging.info("Saved results in file %s" % filename)


estimator = "mean"

sns.lineplot(data=data.query("metric == 'excess_empirical_risk'"), x="t", y="value", hue="algo", estimator=estimator).set_title("Excess Empirical Risk")

plt.show()

sns.lineplot(data=data.query("metric == 'excess_risk'"), x="t", y="value", hue="algo", estimator=estimator).set_title("Excess Risk")

plt.show()

# sns.lineplot(data=data.query("metric == 'gradient_error'"), x="t", y="value", hue="algo", estimator=estimator).set_title("Gradient error")
#
# plt.show()

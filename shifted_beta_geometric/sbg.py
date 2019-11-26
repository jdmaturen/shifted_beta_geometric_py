"""
Implementation of the shifted beta geometric (sBG) model from "How to Project Customer Retention" (Fader and Hardie 2006)

http://www.brucehardie.com/papers/021/sbg_2006-05-30.pdf

Apache 2 License
"""

from math import log

import numpy as np

from scipy.optimize import minimize
from scipy.special import hyp2f1

__author__ = 'JD Maturen'


def generate_probabilities(alpha, beta, x):
    """Generate probabilities in one pass for all t in x"""
    p = [alpha / (alpha + beta)]
    for t in range(1, x):
        pt = (beta + t - 1) / (alpha + beta + t) * p[t-1]
        p.append(pt)
    return p


def probability(alpha, beta, t):
    """Probability function P"""
    if t == 0:
        return alpha / (alpha + beta)
    return (beta + t - 1) / (alpha + beta + t) * probability(alpha, beta, t-1)


def survivor(probabilities, t):
    """Survivor function S"""
    s = 1 - probabilities[0]
    for x in range(1, t + 1):
        s = s - probabilities[x]
    return s


def log_likelihood(alpha, beta, data, survivors=None):
    """Function to maximize to obtain ideal alpha and beta parameters"""
    if alpha <= 0 or beta <= 0:
        return -1000
    if survivors is None:
        survivors = survivor_rates(data)
    probabilities = generate_probabilities(alpha, beta, len(data))
    final_survivor_likelihood = survivor(probabilities, len(data) - 1)

    return sum([s * log(probabilities[t]) for t, s in enumerate(survivors)]) + data[-1] * log(final_survivor_likelihood)


def log_likelihood_multi_cohort(alpha, beta, data):
    """Function to maximize to obtain ideal alpha and beta parameters using data across multiple (contiguous) cohorts.
    `data` must be a list of cohorts each with an absolute number per observed time unit."""
    if alpha <= 0 or beta <= 0:
        return -1000
    probabilities = generate_probabilities(alpha, beta, len(data[0]))

    cohorts = len(data)
    total = 0
    for i, cohort in enumerate(data):
        total += sum([(cohort[j]-cohort[j+1])*log(probabilities[j]) for j in range(len(cohort)-1)])
        total += cohort[-1] * log(survivor(probabilities, cohorts - i - 1))
    return total


def survivor_rates(data):
    s = []
    for i, x in enumerate(data):
        if i == 0:
            s.append(1 - data[0])
        else:
            s.append(data[i-1] - data[i])
    return s


def maximize(data):
    survivors = survivor_rates(data)
    func = lambda x: -log_likelihood(x[0], x[1], data, survivors)
    x0 = np.array([100., 100.])
    res = minimize(func, x0, method='nelder-mead', options={'xtol': 1e-8})
    return res


def maximize_multi_cohort(data):
    func = lambda x: -log_likelihood_multi_cohort(x[0], x[1], data)
    x0 = np.array([1., 1.])
    res = minimize(func, x0, method='nelder-mead', options={'xtol': 1e-8})
    return res


def predicted_retention(alpha, beta, t):
    """Predicted retention probability at t. Function 8 in the paper"""
    return (beta + t) / (alpha + beta + t)


def predicted_survival(alpha, beta, x):
    """Predicted survival probability, i.e. percentage of customers retained, for all t in x.
    Function 1 in the paper"""
    s = [predicted_retention(alpha, beta, 0)]
    for t in range(1, x):
        s.append(predicted_retention(alpha, beta, t) * s[t-1])
    return s


def fit(data):
    res = maximize(data)
    if res.status != 0:
        raise Exception(res.message)
    return res.x


def fit_multi_cohort(data):
    res = maximize_multi_cohort(data)
    if res.status != 0:
        raise Exception(res.message)
    return res.x


def derl(alpha, beta, d, n):
    """discounted expected residual lifetime from "Customer-Base Valuation in a Contractual Setting: The Perils of
    Ignoring Heterogeneity" (Fader and Hardie 2009)"""
    return predicted_retention(alpha, beta, n) * hyp2f1(1, beta + n + 1, alpha + beta + n + 1, 1 / (1 + d))


def test():
    """Test against the High End subscription retention data from the paper"""
    example_data = [.869, .743, .653, .593, .551, .517, .491]
    ll11 = log_likelihood(1., 1., example_data)
    print(np.allclose(ll11, -2.115, 1e-3))

    res = maximize(example_data)
    alpha, beta = res.x
    print(res.status == 0 and np.allclose(alpha, 0.668, 1e-3) and np.allclose(beta, 3.806, 1e-3))
    print()

    print("real\t", ["{0:.1f}%".format(x*100) for x in example_data])
    print("pred\t", ["{0:.1f}%".format(x*100) for x in predicted_survival(alpha, beta, 12)])
    print()

    print(list(map("{0:f}".format, [derl(alpha, beta, 0.1, x) for x in range(12)])))
    print()

    multi_cohort_data = [[10000, 8000, 6480, 5307, 4391], [10000, 8000, 6480, 5307], [10000, 8000, 6480], [10000, 8000]]
    alpha, beta = fit_multi_cohort(multi_cohort_data)
    print(np.allclose(alpha, 3.80, 1e-2) and np.allclose(beta, 15.19, 1e-2))


if __name__ == '__main__':
    test()

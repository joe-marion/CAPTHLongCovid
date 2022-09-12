import numpy as np
import pandas as pd
import numpy.linalg as la
import scipy.stats as stats
from source.utils import flatten


def make_partial_x(domain, data):
    """
    Creates a domain-level design matrix.  Maps the numbers 0 to K to an indicator matrix
    with K rows (0 is mapped to the 0 vector).

    Parameters
    ----------
    domain: domain.Domain
        describes how to name and transform the data. Domain has K arms
    data: pd.DataFrame
        data frame of N subject data, from a patient object.

    Returns
    -------
        Matrix of shape N by K, with 1's in the colums corresponding to randomization

    """
    n = data.shape[0]
    x = np.zeros((n, domain.K+1))
    x[range(n), data[domain.name].values.astype('int')] = 1.0
    return x[:, 1:]


def sum_matrix(k):
    """
    Creates a restriction matrix of dim K by K.  Used to ensure that for
    a vector y sum(Ry)=0.

    Parameters
    ----------
    k: int
        dimension of the matrix

    Returns
    -------
    R: np.array
        restriction matrix with shape (k, k)

    """
    r = np.ones((k, k))
    if k > 1:
        r = r*(-(k ** 2 - k) ** -0.5)
        r[range(k), range(k)] = (1 - 1 / k) ** 0.5
    return r


# TODO: update this when I finish the domain structure
def make_stan_data(data, domains):
    # Prior specification data
    Hs = flatten([d.H for d in domains])
    final = np.cumsum(Hs).astype('int')
    K = sum(Hs)  # Total number of warms
    H = len(Hs)

    X_ = np.concatenate([make_partial_x(d, data) for d in domains], axis=1)
    A = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ])
    A = np.eye(K)
    return {
        # Meta data for the domain HMs
        # 'K':K,                     # Total number of warms
        # 'H':H,
        # 'first':1 + final - np.array(Hs).astype('int'),
        # 'final': final,
        # 'n0s': np.ones(H),
        # 'B': la.block_diag(*[sum_matrix(h) for h in Hs]), # Ensures the sum-to-1 constraint
        # 'A': np.eye(K), # Fix this later
        # Meta data for the strata
        'S': domains[0].S,
        # Data data
        'N': data.shape[0],
        'strata': data.strata.values.astype('int') + 1,
        'X': np.matmul(X_, A),
        'y': data.endpoint.values
    }


def fit_linear_model(stan_data):
    """
    Fits a Bayesian conjugate linear regression to the stan data.

    Parameters
    ----------
    stan_data: dict-like
        the output of prepare stan data

    Returns
    -------
    dictionary with estimates from the stan model.

    """
    # Deep magic to create the design matrices
    strata_X = ((stan_data['strata'] - 1)[:, None] == np.arange(stan_data['S'])[None, :]) * 1.0
    lm_X = (stan_data['X'][:, :, None] * strata_X[:, None, :]).reshape(stan_data['N'], -1)
    lm_X = np.concatenate([strata_X, lm_X], 1)
    lm_Y = stan_data['y']

    # Compute the covariance matrix
    N, K = lm_X.shape
    Phi0 = np.eye(K) * 0.01
    Phi = np.matmul(lm_X.T, lm_X) + Phi0
    Sigma = la.inv(Phi)

    # Compute the mean
    Mu = np.matmul(Sigma, np.matmul(lm_X.T, lm_Y))

    # Standard deviations
    a = 1.0 + N / 2.0
    b = 1 + 0.5 * ((lm_Y ** 2).sum() - np.matmul(np.matmul(Mu.T, Phi), Mu))

    lm_mean = Mu[stan_data['S']:]
    lm_sd = (Sigma.diagonal()[stan_data['S']:] * (b / a)) ** 0.5

    # This thing
    return {
        'lm_mean': lm_mean,
        'lm_sd': lm_sd,
        'lm_super': 1 - stats.norm().cdf(lm_mean / lm_sd).round(2),
        'lm_lower': lm_mean - 1.96 * lm_sd,
        'lm_upper': lm_mean + 1.96 * lm_sd,
    }


def fit_stan_model(stan_data, sm):
    fit = sm.quiet_sampling(stan_data, chains=1, pars=['delta'],
                            warmup=int(1e3), iter=int(1e3 + 1e3), control={'adapt_delta': 0.9})
    deltas = fit.extract()['delta']

    return fit, {
        'bayes_mean': deltas.mean(0).flatten(),
        'bayes_sd': deltas.std(0).flatten(),
        'bayes_upper': np.percentile(deltas, [97.5], 0).flatten(),
        'bayes_lower': np.percentile(deltas, [2.5], 0).flatten(),
        'bayes_super': np.mean(deltas < 0, 0).flatten()
    }

import numpy as np
import scipy.stats as stats


def flatten(list_of_list):
    return [item for sub_list in list_of_list for item in sub_list]


def stratified_randomization(N, ps):
    inds = (np.linspace(0, 1, N + 1)[1:] + stats.uniform().rvs(1)) % 1
    np.random.shuffle(inds)
    return (inds[:, None] <= ps.cumsum()).argmax(1)

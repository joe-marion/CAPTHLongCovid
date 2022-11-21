import os
import sys
sys.path.append(os.getcwd())

from source.patient import MultivariatePatients, Patients
from source.arm import Arm
from source.domain import Domain
from source.stan_utils import StanWrapper
from source.analysis import make_partial_x

import numpy as np
import pandas as pd
import time

import multiprocessing as mp
from numexpr import set_num_threads
set_num_threads(56)

# Load the stan models
stan_models = {
    'uni': {
        'ind': StanWrapper('models/linear_model.stan', load=True),
        'bhm': StanWrapper('models/hierarchical_linear_model.stan', load=True),
    },
    'mv': {
        'ind': StanWrapper('models/mv_linear_model.stan', load=True),
        'bhm': StanWrapper('models/mv_hierarchical_linear_model.stan', load=True),
    }
}


# Initial patients
def simulate_univariate_data(n, seed, scenario):
    np.random.seed(seed)

    # Create the patient object
    patients = Patients(
        names=['Brain Fog', 'PEM', 'Sleep Dist.', 'Fatigue', 'Headache'],
        prevalence=np.repeat(0.2, 5),
        accrual_rate=20,
        dropout_rate=0.0001
    )

    # Scenarios
    scenarios = {
        'null': np.repeat(-0.5, 5),
        'all': np.repeat(0.0, 5),
        'one': np.array([0.0, 0.0, 0.0, 0.0, -0.5]),
        'two': np.array([0.0, -0.5, -0.5, 0.0, 0.0]),
    }

    # Initial domains
    domains = [Domain(
        arms=[Arm('Loratidine', scenarios[scenario])],
        name='Antihistamine',
        allocation=np.repeat(0.5, 2)
    )]

    patients.enroll_n(n, 0, domains)

    return patients.data, domains[0]


# Simulates multiple endpoints per patient
def simulate_multivariate_data(n, seed, scenario):
    np.random.seed(seed)

    Sigma = np.ones([5, 5]) * 0.2
    np.fill_diagonal(Sigma, 1.0)

    # PEM / SD / Fatigue Group
    Sigma[2, 3] = 0.5
    Sigma[3, 2] = 0.5

    Sigma[1, 2] = 0.5
    Sigma[2, 1] = 0.5

    Sigma[1, 3] = 0.5
    Sigma[3, 1] = 0.5

    # BF / Headache Group
    Sigma[0, 4] = 0.8
    Sigma[4, 0] = 0.8

    # Create the patient object
    patients = MultivariatePatients(
        names=['Brain Fog', 'PEM', 'Sleep Dist.', 'Fatigue', 'Headache'],
        Sigma=Sigma,
        prevalence_mu=np.repeat(-0.45, 5),
        prevalence_sigma=np.eye(5),
        accrual_rate=20,
        dropout_rate=0.0001
    )

    # Scenarios
    scenarios = {
        'null': np.repeat(-0.0, 5),
        'all': np.repeat(-0.5, 5),
        'one': np.array([0.0, 0.0, 0.0, 0.0, -0.5]),
        'two': np.array([0.0, -0.5, -0.5, 0.0, 0.0]),
    }

    # Initial domains
    domains = [Domain(
        arms=[Arm('Loratidine', scenarios[scenario])],
        name='Antihistamine',
        allocation=np.repeat(0.5, 2)
    )]

    patients.enroll_n(n, 0, domains)

    return patients.data, domains[0]


# Returns the stan data
def get_stan_data(data, domain):
    # Some work get the
    pids = data.pid.unique()
    pids.sort()
    inds = [np.where(data.pid == p)[0] for p in pids]

    return {
        # Meta data
        'N': data.shape[0],
        'K': domain.K,
        'S': data.strata.max() + 1,
        'P': data.pid.max() + 1,
        # Data data
        'y': data.endpoint.values,
        'strata': data.strata + 1,
        'narm': data[domain.name].values.astype('int'),
        'X': make_partial_x(domain, data),
        # Patient level indexing
        'first': [ind[0] + 1 for ind in inds],
        'final': [ind[-1] + 1 for ind in inds],
        'arm': data.groupby('pid').aggregate({domain.name: np.min})[domain.name].values.astype('int')
    }


# Fits the stan model
def fit_stan_model(stan_data, sm):
    fit = sm.quiet_sampling(stan_data, chains=1, pars=['delta'],
                            warmup=int(1e3), iter=int(1e3 + 2e3), control={'adapt_delta': 0.9})
    deltas = fit.extract()['delta']

    return fit, {
        'bayes_mean': deltas.mean(0).flatten(),
        'bayes_sd': deltas.std(0).flatten(),
        'bayes_upper': np.percentile(deltas, [97.5], 0).flatten(),
        'bayes_lower': np.percentile(deltas, [2.5], 0).flatten(),
        'bayes_super': np.mean(deltas < 0, 0).flatten()
    }


# Perfoms one simulation
def sim_one(s, seed, N, scenario, typ, model):

    # Simulate the models
    data, domain = simulate_multivariate_data(N, seed, scenario)
    stan_data = get_stan_data(data, domain)
    stan_fit, est = fit_stan_model(stan_data, sm=stan_models[typ][model])

    # Package the results
    treatment = data.query("Antihistamine >0"). \
        groupby(['strata'], as_index=False). \
        aggregate({'endpoint': 'mean', 'arrival': 'count'}). \
        rename(columns={'endpoint': 'treatment_mean', 'arrival': 'treatment', 'Antihistamine': 'drug'})

    # Summarize the raw data
    control = data.query("Antihistamine == 0"). \
        groupby(['strata'], as_index=False). \
        aggregate({'endpoint': 'mean', 'arrival': 'count'}). \
        rename(columns={'endpoint': 'control_mean', 'arrival': 'control'})

    cols = ['bayes_mean', 'bayes_super', 'bayes_sd']

    return pd.DataFrame({
        'N': N,
        'sim': s,
        'seed': seed,
        'scenario': scenario,
        'typ': typ,
        'model': model,
        'strata': range(5),
        'effect': domain.arms[1].effects,
    }). \
        merge(treatment, how='left', on=['strata']). \
        merge(control, how='left', on=['strata']). \
        assign(delta=lambda x: x.treatment_mean - x.control_mean). \
        join(pd.DataFrame({c: est.get(c) for c in cols}))


# Performs one simulation
def sim_some(s, N, scenario):
    seed = (os.getpid() * int(time.time())) % 123456789

    return pd.concat([
        sim_one(s, seed, N, scenario, 'uni', 'ind'),
        sim_one(s, seed, N, scenario, 'uni', 'bhm'),
        sim_one(s, seed, N, scenario, 'mv', 'ind'),
        sim_one(s, seed, N, scenario, 'mv', 'bhm')
    ])


# Run all the simulations
def main(sims, N, scenario, threads):

    # setup the multiprocessor
    set_num_threads(threads)
    pool = mp.Pool(processes=threads)  # mp.cpu_count())
    start = time.time()

    # The vanilla crm
    processes = [pool.apply_async(sim_some, args=(seed, N, scenario)) for seed in range(sims)]
    output = [p.get() for p in processes]

    # Write to disk
    if not os.path.isdir('results/multivariate'):
        os.makedirs('results/multivariate')
    fname = 'results/multivariate/multivariate_N_{}_scenario_{}_sims_{}.csv'
    pd.concat(output).to_csv(fname.format(N, scenario, sims), index=False)
    print('N={} complete in {} minutes'.format(N, np.round((time.time() - start) / 60, 1)))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    import optparse
    parser = optparse.OptionParser()

    parser.add_option(
        '-s',
        '--sims',
        type='int'
    )

    parser.add_option(
        '-n',
        '--N',
        type='int'
    )

    parser.add_option(
        '-r',
        '--scenario',
        type='string'
    )
    parser.add_option(
        '-t',
        '--threads',
        default=55,
        type='int'
    )

    opts, args = parser.parse_args()
    main(opts.sims, opts.N, opts.scenario, opts.threads)

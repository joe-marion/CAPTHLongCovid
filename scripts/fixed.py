import os
import sys
sys.path.append(os.getcwd())
from source.stan_utils import StanWrapper
from source.patient import StratifiedPatients
from source.arm import Arm
from source.domain import Domain
from source.analysis import *

import multiprocessing as mp
#from numexpr import set_num_threads
#set_num_threads(56)
import time

SM = StanWrapper('models/simple_model.stan', load=True)

def wrapper(seed, N):
    # np.random.seed(seed)

    # Setup the domains
    domains = [Domain(
        arms=[
            Arm('Cetirizine', np.zeros(5)),
            Arm('Loratidine', np.array([-0.5, -0.5, -0.5, -0.5, -0.5])),
            Arm('+Famotidine (C)', np.array([0.0, 0.0, 0.0, -0.5, -0.5])),
            Arm('+Famotidine (L)', np.array([-0.5, -0.5, -0.5, 0.0, 0.0])),
        ],
        name='Antihistamine',
        allocation=np.array([2, 1, 1, 1, 1]) / 6.0
    )]

    # Simulated patient population
    patients = StratifiedPatients(
        names=['Brain Fog', 'PEM', 'Dyspnea', 'Fatigue', 'Headache'],
        prevalence=np.array([0.20, 0.25, 0.15, 0.30, 0.10]),
        accrual_rate=20,
        dropout_rate=0.00000001
    )
    patients.enroll_n(N, 0, domains)

    # Summarize the data
    data = patients.get_complete().copy()
    stan_data = make_stan_data(data, domains)
    stan_data['n0'] = 1

    # Fit the models
    stan_fit, est = fit_stan_model(stan_data, sm=SM)
    lm_est = fit_linear_model(stan_data)
    est.update(lm_est)
    cols = ['bayes_mean', 'bayes_super', 'lm_mean', 'lm_super']

    # Package the results
    strata_names = ['Brain Fog', 'PEM', 'Dyspnea', 'Fatigue', 'Headache']
    strata_id = range(len(strata_names))

    drug_names = [a.name for a in domains[0].arms if a.name != 'Placebo']
    drug_id = range(len(drug_names))

    return pd.DataFrame({
            'N': N,
            'sim': seed,
            # 'strata_name': flatten([strata_names for d in drug_names]),
            'strata_id': flatten([strata_id + 1 for d in drug_names]),
            # 'drug_name': flatten([[d for s in strata_names] for d in drug_names]),
            'drug_id': flatten([[d + 1 for s in strata_names] for d in drug_id]),
            'effect': flatten([a.effects for a in domains[0].arms if a.name != 'Placebo']),
        }). \
        join(pd.DataFrame({c: est.get(c) for c in cols}))


def main(sims, N):

    if not os.path.isdir('results/fixed'):
        os.makedirs('results/fixed')

    # Cross the main scenarios with offsets
    pool = mp.Pool(processes=48)  # mp.cpu_count())
    start = time.time()

    # The vanilla crm
    processes = [pool.apply_async(wrapper, args=(seed, N)) for seed in range(sims)]
    output = [p.get() for p in processes]
    fname = 'results/fixed/fixed_N_{}_sims_{}.csv'
    pd.concat(output).to_csv(fname.format(N, sims), index=False)
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

    opts, args = parser.parse_args()
    main(opts.sims, opts.N)

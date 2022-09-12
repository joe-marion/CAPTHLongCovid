import os
import sys
sys.path.append(os.getcwd())
from source.stan_utils import StanWrapper
from source.patient import StratifiedPatients
from source.arm import Arm
from source.domain import Domain
from source.analysis import *

import multiprocessing as mp
from numexpr import set_num_threads
set_num_threads(56)
import time

SM = StanWrapper('models/simple_model.stan', load=True)
NS = range(100, 520, 20)


def wrapper(seed, N):
    np.random.seed(seed)

    # Setup the domains
    domains = [Domain(
        arms=[
            Arm('Cetirizine', np.zeros(5)),
            Arm('Loratidine', np.array([-0.66, -0.66, -0.60, -0.66, -0.66])),
            Arm('+Famotidine (C)', np.array([0.0, 0.0, 0.0, -1.0, 0.0])),
            Arm('+Famotidine (L)', np.repeat(0, 5)),
        ],
        name='Antihistamine',
        allocation=np.array([2, 1, 1, 1, 1]) / 6.0
    )]

    # Simulated patient population
    patients = StratifiedPatients(
        names=['Brain Fog', 'PEM', 'Dyspnea', 'Fatigue', 'Headache'],
        prevalence=np.array([0.20, 0.25, 0.15, 0.30, 0.10]),
        accrual_rate=20,
        dropout_rate=0.0001
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

    # Treatment data summary
    treatment = data.query("Antihistamine >0"). \
        groupby([d.name for d in domains] + ['strata'], as_index=False). \
        aggregate({'endpoint': 'mean', 'arrival': 'count'}). \
        rename(columns={'endpoint': 'treatment_mean', 'arrival': 'treatment'})

    # Control data summary
    control = data.query("Antihistamine == 0"). \
        groupby([d.name for d in domains] + ['strata'], as_index=False). \
        aggregate({'endpoint': 'mean', 'arrival': 'count'}). \
        rename(columns={'endpoint': 'control_mean', 'arrival': 'control'}). \
        drop([d.name for d in domains], axis=1)

    # Combine and save the results
    cols = ['bayes_mean', 'bayes_super', 'lm_mean', 'lm_super']
    return treatment. \
        merge(control, on='strata', how='left'). \
        join(pd.DataFrame({c: est.get(c) for c in cols})). \
        assign(sim=seed)


def main(sims):

    if not os.path.isdir('results/fixed'):
        os.makedirs('results/fixed')

    # Cross the main scenarios with offsets
    pool = mp.Pool(processes=mp.cpu_count())
    for N in NS:
        start = time.time()

        # The vanilla crm
        processes = [pool.apply_async(wrapper, args=(seed, )) for seed in range(sims)]
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

    opts, args = parser.parse_args()
    main(opts.sims)

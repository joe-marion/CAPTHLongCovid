import numpy as np
import pandas as pd
from scipy import stats
from source.utils import stratified_randomization
import warnings


class Patients:

    def __init__(self, names, prevalence, accrual_rate, dropout_rate):
        """
        Simulates the patients from different strata.  Includes enrollment and tracks new recruits.
        Enrollment is and dropout follow an exponential distribution.

        Parameters
        ----------
        names: list-like
            names of each group
        accrual_rate: positive float
            overall accrual rate in patients per week
        dropout_rate: positive float
            rate of dropout (per year)
        """

        # Meta data
        self.S = len(names)
        self.ind = range(self.S)
        self.followup = 12.0 * np.ones(self.S)

        # Endpoint information that might be changed later
        self.sigma = np.ones(self.S)
        self.untreated = np.zeros(self.S)

        # Strata simulation assumptions
        self.prevalence = prevalence

        # Random number generators
        self.accrual_rng = stats.expon(scale=1/accrual_rate)
        self.dropout_rng = stats.expon(scale=-52/np.log(dropout_rate))
        self.noise_rnd = stats.norm()

        # Information about the strata
        self.strata_names = names
        self.prevalence = prevalence / prevalence.sum()  # Normalize

        # Accrual and dropout
        self.accrual_rate = accrual_rate
        self.dropout_rate = dropout_rate

        # Storage for patient data
        self.enrolled = 0
        self.data = pd.DataFrame({
            'pid': np.array([], dtype='int64'),
            'strata': np.array([], dtype='int64'),
            'arrival': np.array([], dtype='float64'),
            'complete': np.array([], dtype='float64'),
            'dropout': np.array([], dtype='float64')
        })

    @staticmethod
    def _simulate_0():
        """
        Simulates 0 patients.

        Returns
        -------
        list of empty np.array summarizing the next n patients recruited
            strata indicator
            arrival time (in weeks)
            complete time (in weeks)
            dropout time (in weeks)
        """

        # Init containers of the correct types.
        pid = np.array([], dtype='int64')
        strata = np.array([], dtype='int64')
        arrival = np.array([], dtype='float64')
        complete = np.array([], dtype='float64')
        dropout = np.array([], dtype='float64')
        noise = np.array([], dtype='float64')

        return pid, strata, arrival, complete, dropout, noise

    def _simulate_strata(self, n):
        """ Simple strata """
        strata = stratified_randomization(n, ps=self.prevalence)
        noise = self.noise_rnd.rvs(n) * self.sigma[strata]
        return np.arange(n), strata, noise

    def _simulate(self, n, arrival):
        """ Simulates n patients arriving at specific times """

        assert len(arrival) == n

        # Handle the 0 case
        if n == 0:
            return self._simulate_0()

        pid, strata, noise = self._simulate_strata(n)

        # Simulate the timing
        complete = arrival[pid] + self.followup[strata]
        dropout = arrival[pid] + self.dropout_rng.rvs(n).cumsum()[pid]

        return pid, strata, arrival[pid], complete, dropout, noise

    def _simulate_n(self, n, time):
        """ Simulates strata, arrival times and dropout times for n patients beginning at a specified time. """
        # If 0 patients are simulated...
        arrival = time + self.accrual_rng.rvs(n).cumsum()
        return self._simulate(n, arrival)

    def _simulate_t(self, t, time):
        """  strata, arrival times and dropout times for patients enrolled over t weeks beginning at time t. """

        # Figure out how many patients arrive
        # Sum of n exp(lamda) is a Gamma(n, lamda)
        # So we want to sample enough that there is a high probability of enrolling enough
        # i.e n/accrual - 3*sqrt(n)/accrual > t
        # This next lines solves that using the quadratic equation
        n_max = np.ceil((4 + np.sqrt(16 + 4*t*self.accrual_rate))**2/4).astype('int')
        arrival_ = self.accrual_rng.rvs(n_max).cumsum()
        n = np.sum(arrival_ < t)

        # Raise a warning if my upper bound doesn't hold
        if t > arrival_.max():
            warnings.warn('simulate_t did not simulate enough patients (t={} remaining)'.format(t - arrival_.max()))

        return self._simulate(n, arrival_[0:n])

    def _enroll(self, pid, strata, arrival, complete, dropout, noise, domains):
        """
        Enrolls a set of simulated patients in the study. Each patient is randomized amongst the different domains.
        Updates the data set held within this object.

        Parameters
        ----------
        pid: np.array of int
        strata: np.array of int
        arrival: np.array of float
        complete: np.array of float
        dropout: np.array of float
        domains: list of source.domain.Domain

        """

        # Determine the treatment effect
        # todo: need to update randomized patients to add a strata argument
        assignments = [d.randomize_patients(pid, strata) for d in domains]
        effects = np.array([d.treatment_effect(strata, a) for d, a in zip(domains, assignments)]).sum(0)

        # Keep track of the meta data
        patient_dict = {
            'pid': pid + self.enrolled,
            'strata': strata,
            'arrival': arrival,
            'complete': complete,
            'dropout': dropout,
            'endpoint': effects + noise,
        }

        # Add the randomization decisions
        patient_dict.update({d.name: a for d, a in zip(domains, assignments)})

        self.data = self.data.append(pd.DataFrame(patient_dict))
        if len(pid) > 0:
            self.enrolled = self.enrolled + pid.max()

    def enroll_n(self, n, time, domains):
        """
        Enrolls n patients and randomizes them among the arms in each domain

        Parameters
        ----------
        n: int
            number of patients to simulate
        time: float
            time at which to begin enrolling patients
        domains: list of source.domain.Domain
            describes the drugs currently enrolling in the study
        """

        pid, strata, arrival, complete, dropout, noise = self._simulate_n(n, time)
        self._enroll(pid, strata, arrival, complete, dropout, noise, domains)

    def enroll_t(self, t, time, domains):
        """
        Enrolls patients for t weeks and randomizes them among the arms in each domain

        Parameters
        ----------
        t: float
            number of patients to simulate
        time: float
            time at which to begin enrolling patients
        domains: list of source.domain.Domain
            describes the drugs currently enrolling in the study
        """

        pid, strata, arrival, complete, dropout, noise = self._simulate_t(t, time)
        self._enroll(pid, strata, arrival, complete, dropout, noise, domains)

    # A bunch of methods for getting specific cuts of the data
    def get_enrolled(self, arrival_time=np.inf):
        return self.data.query('arrival < {}'.format(arrival_time))

    def get_n_enrolled(self, arrival_time=np.inf):
        return self.get_enrolled(arrival_time).pid.nunique()

    def get_otc(self, analysis_time=np.inf, arrival_time=np.inf):
        return self.get_enrolled(arrival_time).\
            query('complete < {}'.format(analysis_time))

    def get_n_otc(self, analysis_time=np.inf, arrival_time=np.inf):
        return self.get_otc(analysis_time, arrival_time).pid.nunique()

    def get_complete(self, analysis_time=np.inf, arrival_time=np.inf):
        return self.get_otc(analysis_time, arrival_time).\
            query('dropout > complete')

    def get_n_complete(self, analysis_time=np.inf, arrival_time=np.inf):
        return self.get_complete(analysis_time, arrival_time).pid.nunique()


class MultivariatePatients(Patients):

    def __init__(self, names, Sigma, prevalence_mu, prevalence_sigma, accrual_rate, dropout_rate):
        """ Class that allows for patients to occur in multiple strata """

        # Items specific to the multivariate framework
        assert prevalence_mu.shape[0] == prevalence_sigma.shape[0]
        self.prevalence_rng = stats.multivariate_normal(mean=prevalence_mu, cov=prevalence_sigma)
        prevalence = stats.norm.cdf(prevalence_mu)

        Patients.__init__(self, names=names, prevalence=prevalence, accrual_rate=accrual_rate, dropout_rate=dropout_rate)

        # Add multivariate noise
        self.noise_rnd = stats.multivariate_normal(mean=np.zeros(self.S), cov=Sigma)

    def _simulate_strata(self, n):
        """ Simulates the strata for n patients, allowing multiple measurements per patient"""

        # Sample the latent variables
        latent = self.prevalence_rng.rvs(n*2)
        n_temp = (latent > 0).any(1).sum()

        # Use additional samples if you need to
        while n_temp < n:
            latent = np.vstack([latent, self.prevalence_rng.rvs(n)])
            n_temp = (latent > 0).any(1).sum()

        # Determine which measurements to include
        has_symptom = (latent > 0).any(1)
        include = has_symptom & (has_symptom.cumsum() <= n) # limits to first n patients

        # Determine which observations / strata to use
        # todo: verify that I am using 0 indexing for strata..?
        pid, strata = np.where(latent[include] > 0)
        noise = self.noise_rnd.rvs(n)[pid, strata]

        return pid, strata, noise

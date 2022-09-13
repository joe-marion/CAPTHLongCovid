import numpy as np
import pandas as pd
from scipy import stats
from source.utils import stratified_randomization
import warnings


class StratifiedPatients:

    def __init__(self, names, prevalence, accrual_rate, dropout_rate):
        """
        Simulates the patients from different strata.  Includes enrollment and tracks new recruits.
        Enrollment is and dropout follow an exponential distribution.

        Parameters
        ----------
        names: list-like
            names of each strata
        prevalence: np.array
            relative prevalence of each strata
        accrual_rate: positive float
            overall accrual rate in patients per week
        dropout_rate: positive float
            rate of dropout (per year)
        """

        # Limited error handling
        assert len(names) == len(prevalence)

        # Meta data
        self.S = len(names)
        self.ind = range(self.S)
        self.followup = 24.0 * np.ones(self.S)
        self.sigma = 1.0 * np.ones(self.S)

        # Random number generators
        self.accrual_rng = stats.expon(scale=1/accrual_rate)
        self.dropout_rng = stats.expon(scale=-52/np.log(dropout_rate))
        self.noise_rnd = stats.norm()

        # Information about the strata
        self.strata_names = names
        self.prevalence = prevalence / prevalence.sum()  # Normalize
        self.accrual_rate = accrual_rate
        self.dropout_rate = dropout_rate

        # Storage for patient data
        self.data = pd.DataFrame({
            'strata': [],
            'arrival': [],
            'complete': [],
            'dropout': []
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
        strata = np.array([], dtype='int32')
        arrival = np.array([], dtype='float64')
        complete = np.array([], dtype='float64')
        dropout = np.array([], dtype='float64')

        return strata, arrival, complete, dropout

    def _simulate_n(self, n, time):
        """
        Simulates strata, arrival times and dropout times for n patients beginning at a specified time.

        Parameters
        ----------
        n: int
            number of patients to simulate
        time: float
            time at which to begin enrolling patients

        Returns
        -------
        list of np.array summarizing the next n patients recruited
            strata indicator
            arrival time (in weeks)
            complete time (in weeks)
            dropout time (in weeks)
        """

        # If 0 patients are simulated...
        if n == 0:
            return self._simulate_0()

        strata = stratified_randomization(n, ps=self.prevalence)

        # Simulate the riming
        arrival = time + self.accrual_rng.rvs(n).cumsum()
        complete = arrival + self.followup[strata]
        dropout = arrival + self.dropout_rng.rvs(n).cumsum()

        return strata, arrival, complete, dropout

    def _simulate_t(self, t, time):
        """
        Simulates strata, arrival times and dropout times for patients enrolled over t weeks beginning at time.

        Parameters
        ----------
        t: float
            number of patients to simulate
        time: float
            time at which to begin enrolling patients

        Returns
        -------
        list of np.array summarizing t weeks of recruitment
            strata indicator
            arrival time (in weeks)
            complete time (in weeks)
            dropout time (in weeks)
        """

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

        # If 0 patients are simulated...
        if n == 0:
            return self._simulate_0()

        # Otherwise do the simulations
        else:
            strata = stratified_randomization(n, ps=self.prevalence)
            arrival = time + arrival_[0:n]
            complete = arrival + self.followup[strata]
            dropout = arrival + self.dropout_rng.rvs(n).cumsum()

            return strata, arrival, complete, dropout

    def _enroll(self, strata, arrival, complete, dropout, domains):
        """
        Enrolls a set of simulated patients in the study. Each patient is randomized amongst the different domains.
        Updates the data set held within this object.

        Parameters
        ----------

        strata: np.array of int
        arrival: np.array of float
        complete: np.array of float
        dropout: np.array of float
        domains: list of source.domain.Domain

        """

        # Determine the treatment effect
        assignments = [d.randomize_patients(strata) for d in domains]
        effects = np.array([d.treatment_effect(strata, a) for d, a in zip(domains, assignments)]).sum(0)
        noise = self.noise_rnd.rvs(len(strata)) * self.sigma[strata]

        # Keep track of the meta data
        patient_dict = {
            'strata': strata,
            'arrival': arrival,
            'complete': complete,
            'dropout': dropout,
            'endpoint': effects + noise,
        }

        # Add the randomization decisions
        patient_dict.update({d.name: a for d, a in zip(domains, assignments)})

        self.data = self.data.append(pd.DataFrame(patient_dict))

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

        strata, arrival, complete, dropout = self._simulate_n(n, time)
        self._enroll(strata, arrival, complete, dropout, domains)

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

        strata, arrival, complete, dropout = self._simulate_t(t, time)
        self._enroll(strata, arrival, complete, dropout, domains)

    # A bunch of methods for getting specific cuts of the data
    def get_enrolled(self, arrival_time=np.inf):
        return self.data.query('arrival < {}'.format(arrival_time))

    def get_n_enrolled(self, arrival_time=np.inf):
        return self.get_enrolled(arrival_time).shape[0]

    def get_otc(self, analysis_time=np.inf, arrival_time=np.inf):
        return self.get_enrolled(arrival_time).\
            query('complete < {}'.format(analysis_time))

    def get_n_otc(self, analysis_time=np.inf, arrival_time=np.inf):
        return self.get_otc(analysis_time, arrival_time).shape[0]

    def get_complete(self, analysis_time=np.inf, arrival_time=np.inf):
        return self.get_otc(analysis_time, arrival_time).\
            query('dropout > complete')

    def get_n_complete(self, analysis_time=np.inf, arrival_time=np.inf):
        return self.get_complete(analysis_time, arrival_time).shape[0]

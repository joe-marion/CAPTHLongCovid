import numpy as np


class Arm:

    def __init__(self, name, effects):
        """
        A single drug arm.  The basic building block for a platform trail

        Parameters
        ----------
        name:str
            the name of the drug
        effects: np.array of floats
            the treatment effect in each strata. Assumes that negative is a benefit.
        """

        self.name = name
        self.effects = effects
        self.S = len(effects)


class Placebo:

    def __init__(self, strata):
        """
        A generic placebo arm

        """

        Arm.__init__(self, name='Placebo', effects=np.zeros(strata))

import pickle
import os
from pystan import StanModel


class StanWrapper:

    def __init__(self, fname, load=True, **kwargs):
        """ Wrapper for a stan model that adds some convenience functions.
         Theres probably a better way to do this that inherits from pystan.StanModel"""

        self.SM = self._load_stan_model(fname, load)

    @staticmethod
    def _load_stan_model(fname, load=True):
        """
        Tries to load a stan model from disk

        Parameters
        ----------
        fname: str
            path to (compiled) stan model. If the model
            isn't compiled this will compile it.
        load: bool
            if true, re-compiles the stan model

        Returns
        -------
        compiled pystan.StanModel
        """

        # Load the model if you can
        compiled_fname = fname.split('.stan')[0] + '.pkl'

        if os.path.exists(compiled_fname) and load:
            with open(compiled_fname, 'rb') as f:
                SM = pickle.load(f)

        # Otherwise compile the model and write to disk
        else:
            SM = StanModel(file=fname)
            with open(compiled_fname, 'wb') as f:
                pickle.dump(SM, f)

        return SM

    def quiet_sampling(self, *args, **kwargs):
        with suppress_stdout_stderr():
            return self.SM.sampling(*args, **kwargs)

    def sampling(self, *args, **kwargs):
        return self.SM.sampling(*args, **kwargs)


class suppress_stdout_stderr(object):
    """https://github.com/facebook/prophet/issues/223

    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function. This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through)."""

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        for fd in self.null_fds + self.save_fds:
            os.close(fd)

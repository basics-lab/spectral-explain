"""
A shell Class for common interface to an input signal. This class should be extended when implemented
"""
import numpy as np
from .utils import gwht_tensored, igwht_tensored, save_data, load_data
from pathlib import Path


class Signal:
    """
    Class to encapsulate a time domain signal and its q-ary Fourier transform.

    Attributes
    ---------
    n : int
    number of bits: number of function inputs.
    
    q : int
    Locations of true peaks in the W-H spectrum. Elements must be integers in [0, q ** n - 1].
    
    noise_sd : scalar
    The standard deviation of the added noise.

    signal_t
    Time domain representation of the signal.

    signal_w
    Frequency domain representation of the signal,

    calc_w
    If True and signal_w is not included, it is computed based on signal_t.

    foldername
    If signal_t is not provided, the signal will be read from {foldername}/signal_t.pickle. If signal_t is provided, a
    copy of the signal is written to {foldername}/signal_t.pickle
    """

    def __init__(self, **kwargs):
        self._set_params(**kwargs)
        self._init_signal()

    def _set_params(self, **kwargs):
        self.n = kwargs.get("n")
        self.q = kwargs.get("q")
        self.noise_sd = kwargs.get("noise_sd")
        self.N = self.q ** self.n
        self.signal_t = kwargs.get("signal_t")
        self.signal_w = kwargs.get("signal_w")
        self.calc_w = kwargs.get("calc_w", False)
        self.foldername = kwargs.get("folder")
        self.is_synt = False

    def _init_signal(self):

        if self.signal_t is None:
            signal_path = Path(f"{self.foldername}/signal_t.pickle")
            if signal_path.is_file():
                self.signal_t = load_data(Path(f"{self.foldername}/signal_t.pickle"))
            else:
                self.sample()
                Path(f"{self.foldername}").mkdir(exist_ok=True)
                save_data(self.signal_t, Path(f"{self.foldername}/signal_t.pickle"))

        if self.calc_w and self.signal_w is None:
            self.signal_w = gwht_tensored(self.signal_t, self.q, self.n)
            if np.linalg.norm(self.signal_t - igwht_tensored(self.signal_w, self.q, self.n)) / self.N < 1e-5:
                print("verified transform")

    def sample(self):
        raise NotImplementedError

    def shape(self):
        '''
        shape: returns the shape of the time domain signal.

        Returns
        -------
        shape of time domain signal
        '''
        return tuple([self.q for i in range(self.n)])

    def get_time_domain(self, inds):
        '''
        Arguments
        ---------
        inds: tuple of 1d n-element arrays that represent the indicies to be queried

        Returns
        -------
        indices : linear output of the queried indicies
        '''
        inds = np.array(inds)
        if len(inds.shape) == 3:
            return [self.signal_t[tuple(inds)] for inds in inds]
        elif len(inds.shape) == 2:
            return self.signal_t[tuple(inds)]

'''
Methods for the query generator: specifically, to

1. generate sparsity coefficients b and subsampling matrices M
2. get the indices of a signal subsample
3. compute a subsampled and delayed Walsh-Hadamard transform.
'''
import time
import numpy as np
from .utils import fwht, gwht, bin_to_dec, binary_ints, qary_ints
from .BCH import BCH
from functools import partial


def get_Ms_simple(n, b, **kwargs):
    '''
    Sets Ms[0] = [I 0 ...], Ms[1] = [0 I ...], Ms[2] = [0 0 I 0 ...] and so forth. See get_Ms for full signature.
    '''
    Ms = []
    for i in range(kwargs["num_to_get"] - 1, -1, -1):
        M = np.zeros((n, b), dtype=int32)
        M[(b * i) : (b * (i + 1)), :] = np.eye(b)
        Ms.append(M)

    return Ms


def get_Ms_complex(n, b, q, **kwargs):
    """
    Generate M uniformly at random.
    """
    Ms = []
    # TODO Prevent duplicate M (Not a problem for large n, b)
    for i in range(kwargs["num_to_get"]):
        M = np.random.randint(q, size=(n, b))
        Ms.append(M)
    return Ms


def get_Ms_parity(n, b, q, **kwargs):
    """
    Generate M uniformly at random.
    """
    Ms = []
    # TODO Prevent duplicate M (Not a problem for large n, b)
    for i in range(kwargs["num_to_get"]):
        M = np.hstack((np.ones((n, 1), dtype=int), np.random.randint(q, size=(n, b-1))))
        Ms.append(M)
    return Ms

def get_Ms_antithetic(n, b, q, **kwargs):
    """
    Returns
    -------
    Antithetic sampling matrix M.
    """
    Ms = []
    # TODO Prevent duplicate M (Not a problem for large n, b)
    for i in range(kwargs["num_to_get"]):
        M1 = np.random.randint(q, size=(n, b-1))
        M2 = (1 - M1[:, 0]).astype(int)
        M = np.hstack((M2[:, np.newaxis], M1))
        Ms.append(M)
    return Ms
def get_Ms_complex_low_degree(n, b, q, **kwargs):
    C = kwargs["num_to_get"]
    M = np.zeros((n, b * C), dtype=int)
    for i in range(b * C):
        for j in range(2**i):
            M[j::2**(i+1), i] = 1
    Ms = []
    for i in range(C):
        Ms.append(M[:, i*b:(i+1)*b])
    return Ms


def get_Ms(n, b, q, method="simple", **kwargs):
    '''
    Gets subsampling matrices for different sparsity levels.

    Arguments
    ---------
    n : int
    log_q of the signal length (number of inputs to function).

    b : int
    subsampling dimension.

    num_to_get : int
    The number of M matrices to return.

    method : str
    The method to use. All methods referenced must use this signature (minus "method".)

    Returns
    -------
    Ms : list of numpy.ndarrays, shape (n, b)
    The list of subsampling matrices.
    '''
    if "num_to_get" not in kwargs:
        kwargs["num_to_get"] = max(n // b, 3)

    if method == "simple" and kwargs["num_to_get"] > n // b:
        raise ValueError("When query_method is 'simple', the number of M matrices to return cannot be larger than n // b")

    return {
        "simple": get_Ms_simple,
        "complex": get_Ms_complex,
        "complex_ld": get_Ms_complex_low_degree,
        "parity": get_Ms_parity,
        "antithetic": get_Ms_antithetic,
    }.get(method)(n=n, b=b, q=q, **kwargs)


def get_D_identity(n, **kwargs):
    int_delays = np.zeros(n, )
    int_delays = np.vstack((int_delays, np.eye(n)))
    return int_delays.astype(int)


def get_D_random(n, **kwargs):
    '''
    Gets a random delays matrix of dimension (num_delays, n). See get_D for full signature.
    '''
    q=kwargs.get("q")
    num_delays = kwargs.get("num_delays")
    return np.random.choice(q, (num_delays, n))


def get_D_joint_coded(n, **kwargs):
    t = kwargs.get("t")
    D = BCH(n, t).get_delay_matrix()
    return np.array(D, dtype=int)


def get_D_source_coded(n, **kwargs):
    q = kwargs.get("q")
    t_max = kwargs.get("t")
    D = ReedSolomon(n, t_max, q).get_delay_matrix()
    return np.array(D, dtype=int)


def get_D_nso(n, D_source, **kwargs):
    '''
    Get a repetition code based (NSO-SPRIGHT) delays matrix. See get_D for full signature.
    '''
    num_repeat = kwargs.get("num_repeat")
    q = kwargs.get("q")
    random_offsets = get_D_random(n, q=q, num_delays=num_repeat)
    D = []
    for row in random_offsets:
        modulated_offsets = (row - D_source) % q
        D.append(modulated_offsets)
    return D


def get_D_channel_coded(n, D, **kwargs):
    raise NotImplementedError("One day this might be implemented")

def get_D_channel_identity(n, D, **kwargs):
    q = kwargs.get("q")
    return [D % q]

def get_D(n, **kwargs):
    '''
    Delay generator: gets a delays matrix.

    Arguments
    ---------
    n : int
    number of bits: log2 of the signal length.

    Returns
    -------
    D : numpy.ndarray of binary ints, dimension (num_delays, n).
    The delays matrix; if num_delays is not specified in kwargs, see the relevant sub-function for a default.
    '''
    delays_method_source = kwargs.get("delays_method_source", "random")
    D = {
        "random": get_D_random,
        "identity": get_D_identity,
        "coded": get_D_source_coded,
        "joint-coded": get_D_joint_coded,
    }.get(delays_method_source)(n, **kwargs)

    delays_method_channel = kwargs.get("delays_method_channel", "identity")
    D = {
            "nso": get_D_nso,
            "coded": get_D_channel_coded,
            "identity": get_D_channel_identity,
            "identity-siso": get_D_channel_identity,
    }.get(delays_method_channel)(n, D, **kwargs)
    return D


def subsample_indices(M, d):
    '''
    Query generator: creates indices for signal subsamples.

    Arguments
    ---------
    M : numpy.ndarray, shape (n, b)
    The subsampling matrix; takes on binary values.

    d : numpy.ndarray, shape (n,)
    The subsampling offset; takes on binary values.

    Returns
    -------
    indices : numpy.ndarray, shape (B,)
    The (decimal) subsample indices. Mostly for debugging purposes.
    '''
    L = binary_ints(M.shape[1])
    inds_binary = np.mod(np.dot(M, L).T + d, 2).T
    return bin_to_dec(inds_binary)


def compute_delayed_gwht(signal, M, D, q):
    """
    Computes the Fourier transform of the delayed signal for some M and for each row in the delay matrix D
    """
    b = M.shape[1]
    L = np.array(qary_ints(b, q))  # List of all length b qary vectors
    base_inds = [(M @ L + np.outer(d, np.ones(q ** b, dtype=int))) % q for d in D]
    used_inds = np.swapaxes(np.array(base_inds), 0, 1)
    used_inds = np.reshape(used_inds, (used_inds.shape[0], -1))
    samples_to_transform = signal.get_time_domain(base_inds)
    transform = np.array([gwht(row, q, b) for row in samples_to_transform])
    return transform, used_inds


def get_Ms_and_Ds(n, q, **kwargs):
    """
    Based on the parameters provided in kwargs, generates Ms and Ds.
    """
    timing_verbose = kwargs.get("timing_verbose", False)
    if timing_verbose:
        start_time = time.time()
    query_method = kwargs.get("query_method")
    b = kwargs.get("b")
    num_subsample = kwargs.get("num_subsample")
    Ms = get_Ms(n, b, q, method=query_method, num_to_get=num_subsample)
    if timing_verbose:
        print(f"M Generation:{time.time() - start_time}")
    Ds = []
    if timing_verbose:
        start_time = time.time()
    D = get_D(n, q=q, **kwargs)
    if timing_verbose:
        print(f"D Generation:{time.time() - start_time}")
    for M in Ms:
        Ds.append(D)
    return Ms, Ds


def compute_delayed_wht(signal, M, D):
    '''
    Creates random delays, subsamples according to M and the random delays,
    and returns the subsample WHT along with the delays.

    Arguments
    ---------
    signal : Signal object
    The signal to subsample, delay, and compute the WHT of.

    M : numpy.ndarray, shape (n, b)
    The subsampling matrix; takes on binary values.

    num_delays : int
    The number of delays to apply; or, the number of rows in the delays matrix.

    force_identity_like : boolean
    Whether to make D = [0; I] like in the noiseless case; for debugging.
    '''
    inds = np.array([subsample_indices(M, d) for d in D])
    used_inds = set(np.unique(inds))
    samples_to_transform = signal.signal_t[np.array([subsample_indices(M, d) for d in D])] # subsample to allow small WHTs
    return np.array([fwht(row) for row in samples_to_transform]), used_inds # compute the small WHTs


def get_reed_solomon_dec(n, t_max, q):
    """
    Gets a suitable reed solomon decoder

    Returns
    -------
    A Reed Solomon syndrome decoder for a t_max error correcting code.
    """
    primeset = [2, 3, 5, 7, 11, 13, 15, 17, 19, 23, 29]
    if q in primeset:
        return ReedSolomon(n, t_max, q).syndrome_decode
    else:
        raise NotImplementedError("q is not a prime number under 30!")


def get_bch_decoder(n, t, dectype="hard", chase_depth=None):
    if dectype == "hard":
        dec = BCH(n, t).parity_decode
    elif dectype == "soft" or dectype == "soft-list":
        dec = partial(BCH(n, t).parity_decode_2chase,
                      list_dec=(dectype == "soft-list"),
                      chase_depth=chase_depth)
    elif dectype == "subseq-soft-t2" or dectype == "subseq-soft-t2-list":
        dec = partial(BCH(n, t).parity_decode_2chase_t2,
                      list_dec=(dectype == "subseq-soft-t2-list"),
                      chase_depth=chase_depth)
    elif dectype == "ml-soft-t2" or dectype == "ml-soft-t2-list":
        dec = partial(BCH(n, t).parity_decode_2chase_t2_max_likelihood,
                      list_dec=(dectype == "ml-soft-t2-list"),
                      chase_depth=chase_depth)
    elif dectype == "subseq-soft-t2-opt" or dectype == "subseq-soft-t2-opt-list":
        dec = partial(BCH(n, t).parity_decode_2chase_t2,
                      list_dec=(dectype == "subseq-soft-t2-opt-list"),
                      chase_depth=chase_depth)
    return dec

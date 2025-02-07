import math
import galois
import numpy as np
import heapq
from .utils import comb
from functools import cache
import itertools
from .utils import k_smallest_partial_sums_with_indices

class BCH(galois.BCH):
    """
    A class that extends the galois.BCH class to provide additional functionality for BCH codes.

    Attributes:
    n (int): The length of the code.
    t (int): The error-correcting capability of the code.
    shortend (int): The shortened length of the code.
    chase_depth (int): The depth of the chase algorithm used in decoding.

    Methods:
    __init__(n, t): Initializes the BCH object with the given parameters.
    parity_decode_2chase(parity, list_dec=False, chase_depth=None, valid_k=None, **kwargs): Decodes the given parity bits using a 2-chase algorithm.
    parity_decode_2chase_t2(parity, list_dec=False, chase_depth=None, **kwargs): Decodes the given parity bits using a 2-chase algorithm with a specified depth.
    parity_decode_2chase_t2_max_likelihood(parity, list_dec=False, chase_depth=None, **kwargs): Decodes the given parity bits using a 2-chase algorithm with maximum likelihood.
    parity_decode_2chase_t2_opt(parity, list_dec=False, chase_depth=None, **kwargs): Optimized version of the 2-chase algorithm for decoding parity bits.
    parity_decode(parity, **kwargs): Decodes the given parity bits.
    get_delay_matrix(): Returns the delay matrix for the BCH code.
    get_parity_length(): Returns the length of the parity bits.
    parameter_search(n, t): Searches for the optimal parameters for the BCH code.
    """
    def __init__(self, n, t):
        nc, kc = BCH.parameter_search(n, t)
        self.shortend = kc - n
        self.chase_depth = t
        super().__init__(n=nc, k=kc)

    def parity_decode_2chase(self, parity, list_dec=False, chase_depth=None, valid_k=None, **kwargs):
        """
        Decodes the given parity bits using a 2-chase algorithm.

        Parameters:
        parity (list or np.ndarray): The parity bits to be decoded.
        list_dec (bool, optional): If True, enables list decoding. Defaults to False.
        chase_depth (int, optional): The depth of the chase algorithm. Defaults to None.
        valid_k (callable, optional): A function to validate the decoded codeword. Defaults to None.
        **kwargs: Additional keyword arguments.

        Returns:
        tuple: A tuple containing the decoded codeword and the number of errors, or a list of such tuples if list_dec is True.
        """
        if chase_depth is None:
            chase_depth = self.chase_depth
        early_exit = not (valid_k is None)
        if not early_exit:
            def valid_k(k):
                return True
        # HDD output of parity bits
        hard_parity = (np.array(parity) < 0).astype(int)
        dec_cw, n_errors = self.parity_decode(list(hard_parity))
        if n_errors != -1 and valid_k(np.array(dec_cw[0, :], dtype=int))[0]:
            return dec_cw, n_errors
        chase_indicies = heapq.nsmallest(chase_depth,
                                         range(self.n - self.k),
                                         key=lambda i: abs(parity[i]))
        chase_indicies = np.array(chase_indicies)
        S = [(dec_cw, n_errors)]
        err = np.zeros(shape=(self.n - self.k,), dtype=int)
        for d in range(1, chase_depth + 1):
            positions = itertools.combinations(np.arange(chase_depth), d)
            for pos in positions:
                err[chase_indicies[pos,],] = 1
                dec_cw, n_errors = self.parity_decode(list((hard_parity + err) % 2))
                err[chase_indicies[pos,],] = 0
                if n_errors != -1 and valid_k(np.array(dec_cw[0, :], dtype=int))[0]:
                    if early_exit:
                        return dec_cw, n_errors
                    # Add to list if result is unique
                    is_new = True
                    i = 0
                    while is_new and i < len(S) - 1:
                        is_new = is_new and (not (S[i + 1][0] == dec_cw).all())
                        i += 1
                    if is_new:
                        S.append((dec_cw, n_errors))
        S.sort(key=lambda x: x[1])
        if (not list_dec) or len(S) <= 2:  # List decoding is off, and there was only 1 or 0 unique decoded cw
            return S[0] if len(S) == 1 else S[1]
        else:
            return S[1:]

    def parity_decode_2chase_t2(self, parity, list_dec=False, chase_depth=None, **kwargs):
        """
        Decodes the given parity bits using a 2-chase algorithm with a specified depth.

        Parameters:
        parity (list or np.ndarray): The parity bits to be decoded.
        list_dec (bool, optional): If True, enables list decoding. Defaults to False.
        chase_depth (int, optional): The depth of the chase algorithm. Defaults to None.
        **kwargs: Additional keyword arguments.

        Returns:
        tuple: A tuple containing the decoded codeword and the number of errors, or a list of such tuples if list_dec is True.
        """
        if chase_depth is None:
            chase_depth = self.chase_depth
        # HDD output of parity bits
        hard_parity = (np.array(parity) < 0).astype(int)
        dec_cw, n_errors = self.parity_decode(list(hard_parity))
        if n_errors != -1:
            return dec_cw, n_errors

        # If hard decoding failed, chase through all the potential error patterns
        chase_indicies = heapq.nsmallest(chase_depth,
                                         range(self.n - self.k),
                                         key=lambda i: abs(parity[i]))
        S = [(dec_cw, n_errors)]
        a = 0
        l = 0
        while l < chase_depth:
            r = l
            err = np.zeros(shape=(self.n-self.k,), dtype=int)
            while r < chase_depth:
                a += 1
                err[chase_indicies[r]] = 1
                dec_cw, n_errors = self.parity_decode(list((hard_parity + err) % 2))
                if n_errors != -1:
                    # Add to list if result is unique
                    is_new = True
                    i = 0
                    while is_new and i < len(S)-1:
                        is_new = is_new and (not (S[i+1][0] == dec_cw).all())
                        i += 1
                    if is_new:
                        S.append((dec_cw, n_errors, sum(err)))
                r += 1
            l += 1
        S.sort(key=lambda x: x[1])
        #print(f"Done |S|={len(S)}, iter={a}, deg={S[0][-1] if len(S) == 1 else S[1][-1]}")
        if (not list_dec) or len(S) <= 2:  # List decoding is off, and there was only 1 or 0 unique decoded cw
            return S[0] if len(S) == 1 else S[1]
        else:
            return S[1:]


    def parity_decode_2chase_t2_max_likelihood(self, parity, list_dec=False, chase_depth=None, **kwargs):
        """
        Decodes the given parity bits using a 2-chase algorithm with maximum likelihood.

        Parameters:
        parity (list or np.ndarray): The parity bits to be decoded.
        list_dec (bool, optional): If True, enables list decoding. Defaults to False.
        chase_depth (int, optional): The depth of the chase algorithm. Defaults to None.
        **kwargs: Additional keyword arguments.

        Returns:
        tuple: A tuple containing the decoded codeword and the number of errors, or a list of such tuples if list_dec is True.
        """
        if chase_depth is None:
            chase_depth = self.chase_depth
        # HDD output of parity bits
        hard_parity = (np.array(parity) < 0).astype(int)
        dec_cw, n_errors = self.parity_decode(list(hard_parity))
        if n_errors != -1:
            return dec_cw, n_errors
        # If hard decoding failed, chase through all the potential error patterns
        chase_indicies = heapq.nsmallest(chase_depth,
                                         range(self.n - self.k),
                                         key=lambda i: abs(parity[i]))
        chase_indicies = np.array(chase_indicies)
        input_arr = [abs(parity[chase_indicies[p]]) for p in range(len(chase_indicies))]
        positions_raw = k_smallest_partial_sums_with_indices(input_arr, chase_depth * (chase_depth + 1) // 2)
        positions = [p[1] for p in positions_raw.__reversed__()]
        S = [(dec_cw, n_errors)]
        err = np.zeros(shape=(self.n - self.k,), dtype=int)
        a = 0
        for pos in positions:
            a += 1
            err[chase_indicies,] = pos
            dec_cw, n_errors = self.parity_decode(list((hard_parity + err) % 2))
            err[chase_indicies,] = 0
            if n_errors != -1:
                # Add to list if result is unique
                is_new = True
                i = 0
                while is_new and i < len(S) - 1:
                    is_new = is_new and (not (S[i + 1][0] == dec_cw).all())
                    i += 1
                if is_new:
                    S.append((dec_cw, n_errors, sum(pos)))
        S.sort(key=lambda x: x[1])
        #print(f"Done |S|={len(S)}, iter={a}, deg={S[0][-1] if len(S) == 1 else S[1][-1]}")
        if (not list_dec) or len(S) <= 2:  # List decoding is off, and there was only 1 or 0 unique decoded cw
            return S[0] if len(S) == 1 else S[1]
        else:
            return S[1:]


    def parity_decode_2chase_t2_opt(self, parity, list_dec=False, chase_depth=None, **kwargs):
        """
        Decodes the given parity bits using an optimized 2-chase algorithm.

        Parameters:
        parity (list or np.ndarray): The parity bits to be decoded.
        list_dec (bool, optional): If True, enables list decoding. Defaults to False.
        chase_depth (int, optional): The depth of the chase algorithm. Defaults to None.
        **kwargs: Additional keyword arguments.

        Returns:
        tuple: A tuple containing the decoded codeword and the number of errors, or a list of such tuples if list_dec is True.
        """
        if chase_depth is None:
            chase_depth = self.chase_depth
        # HDD output of parity bits
        hard_parity = (np.array(parity) < 0).astype(int)
        dec_cw, n_errors = self.parity_decode(list(hard_parity))
        if n_errors != -1:
            return dec_cw, n_errors

        # If hard decoding failed, chase through all the potential error patterns
        chase_indicies = heapq.nsmallest(chase_depth,range(self.n - self.k),key=lambda i: abs(parity[i]))
        S = [(dec_cw, n_errors)]
        a = 0
        l = 0
        while l < chase_depth:
            r = l
            err = np.zeros(shape=(self.n-self.k,), dtype=int)
            while r < chase_depth:
                a += 1
                err[chase_indicies[r]] = 1
                dec_cw, n_errors = self.parity_decode(list((hard_parity + err) % 2))
                if n_errors != -1:
                    return dec_cw, n_errors
            return S[0]


    def parity_decode(self, parity, **kwargs):
        """
        Decodes the given parity bits.

        Parameters:
        parity (list or np.ndarray): The parity bits to be decoded.
        **kwargs: Additional keyword arguments.

        Returns:
        tuple: A tuple containing the decoded codeword and the number of errors.
        """
        parity = galois.GF2(parity)
        cw = np.concatenate((galois.Array.Zeros(shape=(self.k - self.shortend,)), parity))
        dec_cw, n_errors = self.decode(cw, errors=True)
        return dec_cw[:, np.newaxis].T, n_errors

    def get_delay_matrix(self):
        """
        Returns the delay matrix for the BCH code.

        The delay matrix is constructed by vertically stacking a zero matrix of shape
        (self.k - self.shortend, self.k - self.shortend) and the transpose of the last
        (self.k - self.shortend) rows of the generator matrix G.

        Returns:
        np.ndarray: The delay matrix.
        """
        return np.vstack((galois.Array.Zeros(shape=(self.k - self.shortend,)),
                          self.G[-(self.k-self.shortend):, self.k:].T))

    def get_parity_length(self):
        """
        Returns the length of the parity bits.

        Returns:
        int: The length of the parity bits.
        """
        return self.n - self.k

    def parameter_search(n, t):
        """
        Searches for the optimal parameters for the BCH code.

        Parameters:
        n (int): The length of the code.
        t (int): The error-correcting capability of the code.

        Returns:
        tuple: A tuple containing the optimal code length (nc) and the dimension of the code (k).
        """
        d = 2 * t + 1
        m = math.ceil(np.log2(d + 1))
        while m < 20:
            nc = (2 ** m) - 1
            test_bch = galois.BCH(n=nc, d=(2 * t + 1))
            if test_bch.k >= n:
                break
            m += 1
        if m >= 20:
            print("Did you really mean to use n>10^6?")
        return nc, test_bch.k


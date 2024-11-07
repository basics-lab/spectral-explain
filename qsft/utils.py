'''
Utility functions.
'''
import functools

import numpy as np
import scipy.fft as fft
import itertools
import math
import time

from scipy.spatial import ConvexHull
import zlib
import pickle
import json
import heapq


def fwht(x):
    """Recursive implementation of the 1D Cooley-Tukey FFT"""
    # x = np.asarray(x, dtype=float)
    N = x.shape[0]
    if N == 1:
        return x
    else:
        X_even = fwht(x[0:(N//2)])
        X_odd = fwht(x[(N//2):])
        return np.concatenate([(X_even + X_odd),
                               (X_even - X_odd)])


def gwht(x,q,n):
    """Computes the GWHT of an input signal with forward scaling"""
    x_tensor = np.reshape(x, [q] * n)
    x_tf = fft.fftn(x_tensor) / (q ** n)
    x_tf = np.reshape(x_tf, [q ** n])
    return x_tf

def gwht_tensored(x,q,n):
    """Computes the GWHT of an input signal with forward scaling"""
    x_tf = fft.fftn(x) / (q ** n)
    return x_tf

def igwht(x,q,n):
    """Computes the IGWHT of an input signal with forward scaling"""
    x_tensor = np.reshape(x, [q] * n)
    x_tf = fft.ifftn(x_tensor) * (q ** n)
    x_tf = np.reshape(x_tf, [q ** n])
    return x_tf

def igwht_tensored(x,q,n):
    """Computes the IGWHT of an input signal with forward scaling"""
    x_tf = fft.ifftn(x) * (q ** n)
    return x_tf

def bin_to_dec(x):
    n = len(x)
    c = 2**(np.arange(n)[::-1])
    return c.dot(x).astype(int)


def nth_roots_unity(n):
    return np.exp(-2j * np.pi / n * np.arange(n))


def near_nth_roots(ratios, q, eps):
    in_set = np.zeros(ratios.shape, dtype=bool)
    omega = nth_roots_unity(q)
    for i in range(q):
        in_set = in_set | (np.square(np.abs(ratios - omega[i])) < eps)
    is_singleton = in_set.all()
    return is_singleton


def qary_vec_to_dec(x, q):
    n = x.shape[0]
    return np.array([q ** (n - (i + 1)) for i in range(n)], dtype=object) @ np.array(x,  dtype=object)


def dec_to_qary_vec(x, q, n):
    qary_vec = []
    for i in range(n):
        qary_vec.append(np.array([a // (q ** (n - (i + 1))) for a in x], dtype=object))
        x = x - (q ** (n-(i + 1))) * qary_vec[i]
    return np.array(qary_vec, dtype=int)


def dec_to_bin(x, num_bits):
    assert x < 2**num_bits, "number of bits are not enough"
    u = bin(x)[2:].zfill(num_bits)
    u = list(u)
    u = [int(i) for i in u]
    return np.array(u)


def binary_ints(m):
    '''
    Returns a matrix where row 'i' is dec_to_bin(i, m), for i from 0 to 2 ** m - 1.
    From https://stackoverflow.com/questions/28111051/create-a-matrix-of-binary-representation-of-numbers-in-python.
    '''
    a = np.arange(2 ** m, dtype=int)[np.newaxis,:]
    b = np.arange(m, dtype=int)[::-1,np.newaxis]
    return np.array(a & 2**b > 0, dtype=int)

def angle_q(x,q):
    return (((np.angle(x) % (2*np.pi) // (np.pi/q)) + 1) // 2) % q # Can be made much faster

def qary_ints(m, q, dtype=int):
    return np.array(list(itertools.product(np.arange(q), repeat=m)), dtype=dtype).T

def comb(n, k):
    return math.factorial(n) // math.factorial(k) // math.factorial(n - k)


def qary_ints_low_order(m, q, order):
    num_of_ks = np.sum([comb(m, o) * ((q-1) ** o) for o in range(order + 1)])
    K = np.zeros((num_of_ks, m))
    counter = 0
    for o in range(order + 1):
        positions = itertools.combinations(np.arange(m), o)
        for pos in positions:
            K[counter:counter+((q-1) ** o), pos] = np.array(list(itertools.product(1 + np.arange(q-1), repeat=o)))
            counter += ((q-1) ** o)
    return K.T

def base_ints(q, m):
    '''
    Returns a matrix where row 'i' is the base-q representation of i, for i from 0 to q ** m - 1.
    Covers the functionality of binary_ints when n = 2, but binary_ints is faster for that case.
    '''
    get_row = lambda i: np.array([int(j) for j in np.base_repr(i, base=q).zfill(m)])
    return np.vstack((get_row(i) for i in range(q ** m)))

def polymod(p1, p2, q, m):
    '''
    Computes p1 modulo p2, and takes the coefficients modulo q.
    '''
    p1 = np.trim_zeros(p1, trim='f')
    p2 = np.trim_zeros(p2, trim='f')
    while len(p1) >= len(p2) and len(p1) > 0:
        p1 -= p1[0] // p2[0] * np.pad(p2, (0, len(p1) - len(p2)))
        p1 = np.trim_zeros(p1, trim='f')
    return np.pad(np.mod(p1, q), (m + 1 - len(p1), 0))

def rref(A, b, q):
    '''
    Row reduction, to easily solve finite field systems.
    '''
    raise NotImplementedError()

def sign(x):
    '''
    Replacement for np.sign that matches the convention (footnote 2 on page 11).
    '''
    return (1 - np.sign(x)) // 2


def flip(x):
    '''
    Flip all bits in the binary array x.
    '''
    return np.bitwise_xor(x, 1)


def random_signal_strength_model(sparsity, a, b):
    magnitude = np.random.uniform(a, b, sparsity)
    phase = np.random.uniform(0, 2*np.pi, sparsity)
    return magnitude * np.exp(1j*phase)


def random_skewed_signal_strength_model(sparsity, a, b, skewed):
    x_0 = a
    x_1 = b
    magnitude = np.power(np.linspace(x_0, x_1, sparsity),  (-1*skewed))
    signs = (2*(np.random.uniform(0, 1, sparsity) > 0.5).astype(int)) - 1
    return np.random.permutation(magnitude * signs)


def best_convex_underestimator(points):
    hull = ConvexHull(points)
    vertices = points[hull.vertices]
    first_point_idx = np.argmin(vertices[:, 0])
    last_point_idx = np.argmax(vertices[:, 0])

    if last_point_idx == vertices.shape[0]:
        return vertices[first_point_idx:]
    if first_point_idx < last_point_idx:
        return vertices[first_point_idx:last_point_idx+1]
    else:
        return np.concatenate((vertices[first_point_idx:], vertices[:last_point_idx+1]))

def sort_qary_vecs(qary_vecs):
    qary_vecs = np.array(qary_vecs)
    idx = np.lexsort(qary_vecs.T[::-1, :])
    return qary_vecs[idx]

def calc_hamming_weight(qary_vecs):
    qary_vecs = np.array(qary_vecs)
    return np.sum(qary_vecs != 0, axis = 1)

def save_data(data, filename):
    with open(filename, 'wb') as f:
        f.write(zlib.compress(pickle.dumps(data, pickle.HIGHEST_PROTOCOL), 9))

def load_data(filename):
    start = time.time()
    with open(filename, 'rb') as f:
        data = pickle.loads(zlib.decompress(f.read()))
    return data

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)



def k_smallest_partial_sums_with_indices(arr, k):
    """
    Finds the k smallest partial sums and their corresponding indices.

    Parameters:
    arr (list of float): The input array of numbers.
    k (int): The number of smallest partial sums to find.

    Returns:
    list of tuple: A list of tuples, each containing a partial sum and the corresponding indices.
    """
    h = []
    heapq.heappush(h, (0, [0]*len(arr)))
    for i in range(len(arr)):
        current_min = h[0][0]
        # add to the heap
        to_add = []
        for x in h:
            if len(h) < k or x[0] - arr[i] >= current_min:
                to_add.append((x[0] - arr[i], [x[1][j] if j != i else 1 for j in range(len(x[1]))]))
        if len(to_add) > 0:
            for el in to_add:
                heapq.heappush(h, el)
        # cull the heap
        if len(h) > k:
            h = heapq.nlargest(k, h)[::-1]
    return list(h)



if __name__ == '__main__':
    print(k_smallest_partial_sums_with_indices([0, 0.1, 0.5, 1.0, 2.0, 4.0], 3))
    print(k_smallest_partial_sums_with_indices([0, 0.1, 0.5, 1.0, 2.0, 4.0], 6))


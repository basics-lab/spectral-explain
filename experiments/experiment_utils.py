from spectral_explain.support_recovery import support_recovery
from spectral_explain.qsft.qsft import fit_regression
from spectral_explain.qsft.utils import qary_ints_low_order
from spectral_explain.utils import mobius_to_fourier, fourier_to_mobius
import numpy as np
from math import comb

def lime(signal, b, **kwargs):
    return fit_regression('lasso', {'locations': qary_ints_low_order(signal.n, 2, 1).T}, signal, signal.n, b,
                          fourier_basis=False)[0]
def linear(signal, b, order=1, **kwargs):
    return fit_regression('linear', {'locations': qary_ints_low_order(signal.n, 2, order).T}, signal, signal.n, b)[0]

def lasso(signal, b, order=1, **kwargs):
    return fit_regression('lasso', {'locations': qary_ints_low_order(signal.n, 2, order).T}, signal, signal.n, b)[0]

def faith_banzhaf(signal, b, order=1, **kwargs):
    return fit_regression('lasso', {'locations': qary_ints_low_order(signal.n, 2, order).T}, signal, signal.n, b,
                          fourier_basis=False)[0]

def faith_shapley(signal, b, order=1, **kwargs):
    return fit_regression('shapley', {'locations': qary_ints_low_order(signal.n, 2, order).T}, signal, signal.n, b,
                          fourier_basis=False)[0]

def qsft_hard(signal, b, t=5, **kwargs):
    return support_recovery("hard", signal, b, t=t)["transform"]


def qsft_soft(signal, b, t=5, **kwargs):
    return support_recovery("soft", signal, b, t=t)["transform"]


class Alternative_Sampler:
    def __init__(self, type, sampling_function, qsft_signal, n):
        assert type in ["uniform", "shapley", "lime"]
        self.queries_finder = {
            "uniform": self.uniform_queries,
            "shapley": self.shapley_queries,
            "lime": self.lime_queries
        }.get(type, NotImplementedError())

        self.n = n
        self.all_queries = []
        self.all_samples = []
        for m in range(len(qsft_signal.all_samples)):
            queries_subsample = []
            samples_subsample = []
            for d in range(len(qsft_signal.all_samples[0])):
                queries = self.queries_finder(len(qsft_signal.all_queries[m][d]))
                if type == "shapley" and m == 0 and d == 0:
                    queries[0, :] = np.zeros(n)
                    queries[1, :] = np.ones(n)
                queries_subsample.append(queries)
                samples_subsample.append(sampling_function(queries))
            self.all_queries.append(queries_subsample)
            self.all_samples.append(samples_subsample)

    def uniform_queries(self, num_samples):
        return np.random.choice(2, size=(num_samples, self.n))

    def shapley_queries(self, num_samples):
        shapley_kernel = np.array([1 / (comb(self.n, d) * d * (self.n - d)) for d in range(1, self.n)])
        degrees = np.random.choice(range(1, self.n), size=num_samples, replace=True,
                                   p=shapley_kernel / np.sum(shapley_kernel))
        queries = []
        for sample in range(num_samples):
            q = np.zeros(self.n)
            q[np.random.choice(range(self.n), size=degrees[sample], replace=False)] = 1
            queries.append(q)
        return np.array(queries)

    def lime_queries(self, num_samples):
        exponential_kernel = np.array([np.sqrt(np.exp(-(self.n - d) ** 2 / 25 ** 2)) for d in range(1, self.n)])
        degrees = np.random.choice(range(1, self.n), size=num_samples, replace=True,
                                   p=exponential_kernel / np.sum(exponential_kernel))
        queries = []
        for sample in range(num_samples):
            q = np.zeros(self.n)
            q[np.random.choice(range(self.n), size=degrees[sample], replace=False)] = 1
            queries.append(q)
        return np.array(queries)

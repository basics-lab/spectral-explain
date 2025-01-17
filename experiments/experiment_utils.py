from spectral_explain.support_recovery import support_recovery
from spectral_explain.qsft.qsft import fit_regression
from spectral_explain.qsft.utils import qary_ints_low_order
from spectral_explain.utils import mobius_to_fourier, fourier_to_mobius
import numpy as np
import time
import shapiq
import warnings
warnings.filterwarnings("ignore")

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

def faith_shapley(sampling_function, num_samples, n, order=1, **kwargs):
    explainer = shapiq.Explainer(
        model=sampling_function,
        data=np.zeros((1,n)),
        index="FSII",
        max_order=order,
    )
    fsii = explainer.explain(np.ones((1, n)), budget=num_samples)
    mobius_dict = {}
    for interaction, ref in fsii.interaction_lookup.items():
        loc = [0] * n
        for ele in interaction:
            loc[ele] = 1
        mobius_dict[tuple(loc)] = fsii.values[ref]
    return mobius_to_fourier(mobius_dict)

def qsft_hard(signal, b, t=5, **kwargs):
    return support_recovery("hard", signal, b, t=t)["transform"]


def qsft_soft(signal, b, t=5, **kwargs):
    return support_recovery("soft", signal, b, t=t)["transform"]

def shapley(sampling_function, num_samples, n):
    explainer = shapiq.Explainer(
        model=sampling_function,
        data=np.zeros((1,n)),
        index="SV",
        max_order=1
    )
    shapley_values = explainer.explain(np.ones((1, n)), budget=num_samples)
    return shapley_values.values[1:]

def banzhaf(sampling_function, num_samples, n):
    # maximum sample reuse strategy from
    # Eq (5) of Data Banzhaf: https://arxiv.org/pdf/2205.15466
    coordinates = np.random.choice(2, size=(num_samples, n))
    values = sampling_function(coordinates)

    banzhaf_values = np.zeros(n)
    for idx in range(n):
        mask = coordinates[:, idx] > 0.5
        banzhaf_values[idx] = (1 / np.sum(mask)) * np.sum(values[mask]) - (1 / np.sum(1 - mask)) * np.sum(
            values[1 - mask])

    return banzhaf_values



class Alternative_Sampler:
    def __init__(self, type, sampling_function, qsft_signal, n):
        assert type in ["uniform", "lime"]
        self.queries_finder = {
            "uniform": self.uniform_queries,
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
        self.num_samples = sum([len(b) for batch in self.all_samples for b in batch])
        self.sampling_function = sampling_function

    def uniform_queries(self, num_samples):
        return np.random.choice(2, size=(num_samples, self.n))

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

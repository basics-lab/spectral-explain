from spectral_explain.support_recovery import support_recovery
from spectral_explain.qsft.qsft import fit_regression
from spectral_explain.qsft.utils import qary_ints_low_order
from spectral_explain.utils import mobius_to_fourier, fourier_to_mobius
import numpy as np
import time
import shapiq
import lime.lime_tabular
import warnings
warnings.filterwarnings("ignore")


SAMPLER_DICT = {
    "qsft_hard": "qsft",
    "qsft_soft": "qsft",
    "linear": "uniform",
    "lasso": "uniform",
    "lime": "dummy", # will use sampling from lime later on
    "faith_shapley": "dummy",  # will use sampling from Shap-IQ later on
    "shapley": "dummy",  # will use sampling from Shap-IQ later on
    "banzhaf": "uniform",
    "shapley_taylor": "dummy"
}

def LIME(signal, b, **kwargs):
    training_data = np.zeros((2, signal.n))
    training_data[1,:] = np.ones(signal.n)
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(training_data, mode='regression',
                                                            categorical_features=range(signal.n),
                                                            kernel_width=25)  # used in LimeTextExplainer
    lime_values = lime_explainer.explain_instance(np.ones(signal.n), signal.sampling_function,
                                                  num_samples=signal.num_samples(b),
                                                  num_features=signal.n,
                                                  distance_metric='cosine')  # used in LimeTextExplainer
    output = {}
    output[tuple([0] * signal.n)] = lime_values.intercept[1]
    for loc, val in lime_values.local_exp[1]:
        ohe_loc = [0] * signal.n
        ohe_loc[loc] = 1
        output[tuple(ohe_loc)] = val
    return mobius_to_fourier(output)

def linear(signal, b, order=1, **kwargs):
    return fit_regression('linear', {'locations': qary_ints_low_order(signal.n, 2, order).T}, signal, signal.n, b)[0]

def lasso(signal, b, order=1, **kwargs):
    return fit_regression('lasso', {'locations': qary_ints_low_order(signal.n, 2, order).T}, signal, signal.n, b)[0]

def faith_shapley(signal, b, order=1, **kwargs):
    explainer = shapiq.Explainer(
        model=signal.sampling_function,
        data=np.zeros((1,signal.n)),
        index="FSII",
        max_order=order,
    )
    fsii = explainer.explain(np.ones((1, signal.n)), budget=signal.num_samples(b))
    fsii_dict = {}
    for interaction, ref in fsii.interaction_lookup.items():
        loc = [0] * signal.n
        for ele in interaction:
            loc[ele] = 1
        fsii_dict[tuple(loc)] = fsii.values[ref]
    return mobius_to_fourier(fsii_dict)

def shapley_taylor(signal, b, order=1, **kwargs):
    explainer = shapiq.Explainer(
        model=signal.sampling_function,
        data=np.zeros((1,signal.n)),
        index="STII",
        max_order=order,
    )
    stii = explainer.explain(np.ones((1, signal.n)), budget=signal.num_samples(b))
    stii_dict = {}
    for interaction, ref in stii.interaction_lookup.items():
        loc = [0] * signal.n
        for ele in interaction:
            loc[ele] = 1
        stii_dict[tuple(loc)] = stii.values[ref]
    return mobius_to_fourier(stii_dict)

def qsft_hard(signal, b, t=5, **kwargs):
    return support_recovery("hard", signal, b, t=t)["transform"]


def qsft_soft(signal, b, t=5, **kwargs):
    return support_recovery("soft", signal, b, t=t)["transform"]

def shapley(signal, b, **kwargs):
    explainer = shapiq.Explainer(
        model=signal.sampling_function,
        data=np.zeros((1,signal.n)),
        index="SV",
        max_order=1
    )
    shapley = explainer.explain(np.ones((1, signal.n)), budget=signal.num_samples(b))
    shapley_dict = {}
    for interaction, ref in shapley.interaction_lookup.items():
        loc = [0] * signal.n
        for ele in interaction:
            loc[ele] = 1
        shapley_dict[tuple(loc)] = shapley.values[ref]
    return mobius_to_fourier(shapley_dict)



def banzhaf(signal, b, **kwargs):
    # maximum sample reuse strategy from
    # Eq (5) of Data Banzhaf: https://arxiv.org/pdf/2205.15466
    coordinates = []
    values = []
    for m in range(len(signal.all_samples)):
        for d in range(len(signal.all_samples[0])):
            for z in range(2 ** b):
                coordinates.append(signal.all_queries[m][d][z])
                values.append(np.real(signal.all_samples[m][d][z]))
    null_value = signal.sampling_function([[0]*signal.n]).item()
    coordinates = np.array(coordinates)
    values = np.array(values) - null_value

    banzhaf_dict = {}
    for idx in range(signal.n):
        mask = coordinates[:, idx] > 0.5
        not_mask = np.logical_not(mask)
        if sum(mask) == 0 or sum(mask) == len(coordinates):
            banzhaf_value_idx = 0
        else:
            banzhaf_value_idx = ((1 / np.sum(mask)) * np.sum(values[mask])) - ((1 / np.sum(not_mask)) * np.sum(values[not_mask]))
        loc = [0] * signal.n
        loc[idx] = 1
        banzhaf_dict[tuple(loc)] = banzhaf_value_idx
    banzhaf_dict[tuple([0] * signal.n)] = null_value
    return mobius_to_fourier(banzhaf_dict)


class AlternativeSampler:
    def __init__(self, type, sampling_function, qsft_signal, n):
        assert type in ["uniform", "dummy"]
        # dummy just needs access to n, sampling_function and num samples
        self.queries_finder = {
            "uniform": self.uniform_queries,
            "dummy": self.dummy_queries
        }.get(type, NotImplementedError())

        self.n = n
        self.all_queries = []
        self.all_samples = []
        for m in range(len(qsft_signal.all_samples)):
            queries_subsample = []
            samples_subsample = []
            for d in range(len(qsft_signal.all_samples[0])):
                queries = self.queries_finder(len(qsft_signal.all_queries[m][d]))
                queries_subsample.append(queries)
                samples_subsample.append(sampling_function(queries))
            self.all_queries.append(queries_subsample)
            self.all_samples.append(samples_subsample)
        self.num_samples = lambda b: len(qsft_signal.all_samples) * len(qsft_signal.all_samples[0]) * (2 ** b)
        self.sampling_function = sampling_function

    def uniform_queries(self, num_samples):
        return np.random.choice(2, size=(num_samples, self.n))

    def dummy_queries(self, num_samples):
        # Will sample later with method-specific sampling pattern
        return np.zeros((num_samples, self.n))

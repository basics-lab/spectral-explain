from spex.support_recovery import support_recovery
from spex.utils import mobius_to_fourier, fit_regression, bin_vecs_low_order, lgboost_to_fourier
import numpy as np
import shapiq
import lime.lime_tabular
import warnings
import pyrootutils
import lightgbm as lgb

warnings.filterwarnings("ignore")

SAMPLER_DICT = {
    "spex_hard": "spex",
    "spex_soft": "spex",
    "shapley": "dummy",  # will use sampling from Shap-IQ later on
    "banzhaf": "uniform",
    "lime": "dummy",  # will use sampling from lime later on
    "faith_shapley": "dummy",  # will use sampling from Shap-IQ later on
    "faith_banzhaf": "uniform",
    "shapley_taylor": "dummy",  # will use sampling from Shap-IQ later on
    "proxy_spex": "uniform"
}


def spex_hard(signal, b, order=5, **kwargs):
    return support_recovery("hard", signal, b, t=order)


def spex_soft(signal, b, order=5, **kwargs):
    return support_recovery("soft", signal, b, t=order)


def LIME(signal, b, **kwargs):
    training_data = np.zeros((2, signal.n))
    training_data[1, :] = np.ones(signal.n)
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(training_data, mode='regression',
                                                            categorical_features=range(signal.n),
                                                            kernel_width=25)  # used in LimeTextExplainer
    lime_values = lime_explainer.explain_instance(np.ones(signal.n), signal.sampling_function,
                                                  num_samples=signal.num_samples(b),
                                                  num_features=signal.n,
                                                  distance_metric='cosine')  # used in LimeTextExplainer
    output = {tuple([0] * signal.n): lime_values.intercept[1]}
    for loc, val in lime_values.local_exp[1]:
        ohe_loc = [0] * signal.n
        ohe_loc[loc] = 1
        output[tuple(ohe_loc)] = val
    return mobius_to_fourier(output)


def shapley(signal, b, **kwargs):
    explainer = shapiq.Explainer(
        model=signal.sampling_function,
        data=np.zeros((1, signal.n)),
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
    return fit_regression('ridge', {'locations': bin_vecs_low_order(signal.n, 1).T}, signal, signal.n, b,
                          fourier_basis=False)[0]


def faith_shapley(signal, b, order=1, **kwargs):
    explainer = shapiq.Explainer(
        model=signal.sampling_function,
        data=np.zeros((1, signal.n)),
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


def faith_banzhaf(signal, b, order=1, **kwargs):
    return fit_regression('ridge', {'locations': bin_vecs_low_order(signal.n, order).T}, signal, signal.n, b,
                          fourier_basis=False)[0]


def shapley_taylor(signal, b, order=1, **kwargs):
    explainer = shapiq.Explainer(
        model=signal.sampling_function,
        data=np.zeros((1, signal.n)),
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

def proxy_spex(signal, b, **kwargs):
    coordinates = []
    values = []
    for m in range(len(signal.all_samples)):
        for d in range(len(signal.all_samples[0])):
            for z in range(2 ** b):
                coordinates.append(signal.all_queries[m][d][z])
                values.append(np.real(signal.all_samples[m][d][z]))

    coordinates = np.array(coordinates)
    values = np.array(values)
    model = lgb.LGBMRegressor(verbose=-1)
    model.fit(coordinates, values)
    return lgboost_to_fourier(model)

def get_ordered_methods(methods, max_order):
    ordered_methods = []
    for method in methods:
        if method in ['shapley', 'banzhaf', 'lime']:
            ordered_methods.append((method, 1))
        elif method in ['faith_banzhaf', 'faith_shapley', 'shapley_taylor']:
            ordered_methods += [(method, order) for order in range(2, max_order + 1)]
        else:
            # spex methods use maximum order 5
            ordered_methods.append((method, 5))
    return ordered_methods


class AlternativeSampler:
    def __init__(self, type, sampling_function, qsft_signal, n):
        self.n = n
        self.num_samples = lambda b: len(qsft_signal.all_samples) * len(qsft_signal.all_samples[0]) * (2 ** b)
        self.sampling_function = sampling_function
        assert type in ["uniform", "dummy"]

        if type == "uniform":
            self.all_queries = []
            self.all_samples = []
            for m in range(len(qsft_signal.all_samples)):
                queries_subsample = []
                samples_subsample = []
                for d in range(len(qsft_signal.all_samples[0])):
                    queries = self.uniform_queries(len(qsft_signal.all_queries[m][d]))
                    queries_subsample.append(queries)
                    samples_subsample.append(sampling_function(queries))
                self.all_queries.append(queries_subsample)
                self.all_samples.append(samples_subsample)
        else:
            self.all_queries = None
            self.all_samples = None

    def uniform_queries(self, num_samples):
        return np.random.choice(2, size=(num_samples, self.n))


def setup_root():
    root = pyrootutils.setup_root(
        search_from=__file__,
        indicator=[".git", "pyproject.toml"],
        pythonpath=True,
        cwd=True,
        dotenv=True,
    )
    print(f"Current working directory is set to project root: {root}.")

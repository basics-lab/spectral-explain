from spectral_explain.support_recovery import support_recovery
from spectral_explain.qsft.qsft import fit_regression
from spectral_explain.qsft.utils import qary_ints_low_order
from spectral_explain.utils import mobius_to_fourier, fourier_to_mobius
import numpy as np
import dill as pickle
from math import comb

def lime(signal, b, **kwargs):
    return fit_regression('lasso', {'locations': qary_ints_low_order(signal.n, 2, 1).T}, signal, signal.n, b)[0]
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


def get_methods(method_list,max_order):
    ordered_methods = []
    for regression in ['linear', 'lasso', 'faith_banzhaf', 'faith_shapley']:
        if regression in method_list:
            ordered_methods += [(regression, order) for order in range(1, max_order + 1)]
            method_list.remove(regression)
    ordered_methods += [(method, 0) for method in method_list]
    return ordered_methods



def get_and_save_samples(sampling_function, type = 'test', num_test_samples = 10000, n = 128, save_dir = None, use_cache = True, qsft_signal = None):
    assert type in ['test','uniform', 'shapley', 'lime']
    
    if type == 'test':
        if use_cache and os.path.exists(f'{save_dir}/test_samples.parquet'):
            test_df = pd.read_parquet(f'{save_dir}/test_samples.parquet')
            test_labels = test_df['target'].values
            test_queries = test_df.drop(columns=['target']).values
            saved_samples_test = test_queries, test_labels
            return saved_samples_test
        else:
            query_indices_test = np.random.choice(2, size=(num_test_samples, n))
            saved_samples_test = query_indices_test, sampling_function(query_indices_test)
            test_queries_pd = pd.DataFrame(query_indices_test, columns = [f'loc_{i}' for i in range(n)])
            test_queries_pd['target'] = saved_samples_test[1]
            test_queries_pd.to_parquet(f'{save_dir}/test_samples.parquet')
            return saved_samples_test
    else:
        if use_cache and os.path.exists(f'{save_dir}/{type}_signal.pickle'):
            with open(f'{save_dir}/{type}_signal.pickle', 'rb') as handle:
                signal = pickle.load(handle)
            return signal
        else:
            signal = BatchedAlternative_Sampler(type, sampling_function, qsft_signal, n)
            with open(f'{save_dir}/{type}_signal.pickle', 'wb') as handle:
                pickle.dump(signal, handle, protocol=pickle.HIGHEST_PROTOCOL)
            return signal

class BatchedAlternative_Sampler:
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
        self.query_type = type
        for m in range(len(qsft_signal.all_samples)):
            queries_subsample = []
            for d in range(len(qsft_signal.all_samples[0])):
                queries = self.queries_finder(len(qsft_signal.all_queries[m][d]))
                if type == "shapley" and m == 0 and d == 0:
                    queries[0, :] = np.zeros(n)
                    queries[1, :] = np.ones(n)
                queries_subsample.append(queries)
            self.all_queries.append(queries_subsample)
        query_matrix = np.concatenate([np.concatenate(queries_subsample, axis=0) for queries_subsample in self.all_queries], axis=0)
        samples = sampling_function(query_matrix)
        #self.flattened_queries = query_matrix
        #self.flattened_samples = samples
        count = 0
        for queries_subsample in self.all_queries: # list of numpy arrays
            samples_subsample = []
            for query in queries_subsample:
                samples_subsample.append(samples[count: count + len(query)])
                count += len(query)
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
from spectral_explain.support_recovery import support_recovery
from spectral_explain.qsft.qsft import fit_regression
from spectral_explain.qsft.utils import qary_ints_low_order
from spectral_explain.utils import mobius_to_fourier, fourier_to_mobius
from spectral_explain.support_recovery import sampling_strategy, get_num_samples
import numpy as np
import dill as pickle
from math import comb
import torch
import gc
import shapiq
from copy import copy
from tqdm import tqdm
import lime.lime_tabular
import os, pandas as pd
import warnings
warnings.filterwarnings("ignore")

# def lime(signal, b, **kwargs):
#     return fit_regression('lasso', {'locations': qary_ints_low_order(signal.n, 2, 1).T}, signal, signal.n, b)[0]

SAMPLER_DICT = {
    "qsft_hard": "qsft",
    "qsft_soft": "qsft",
    "linear": "uniform",
    "lasso": "uniform",
    "lime": "lime",
    "faith_banzhaf": "uniform",
    #"faith_shapley": "shapley",
    #"shapley": "shapley",
    "banzhaf": "uniform"
}
METHODS = ['linear', 'lasso', 'lime', 'qsft_hard', 'qsft_soft', 'faith_shapley'] #'faith_banzhaf'
SAMPLING_SET = set(SAMPLER_DICT.values())

# Sampling methods 


# returns a shapley explainer object
def shapley_sampling(sampling_function, qsft_signal, order, b, **kwargs):
    explainer = shapiq.Explainer(
        model=sampling_function,
        data=np.zeros((1,qsft_signal.n)),
        index="FSII",
        max_order=order,
    )
    num_samples = get_num_samples(qsft_signal, b)
    print(num_samples)
    fsii = explainer.explain(np.ones((1, qsft_signal.n)), budget=num_samples)
    flush()
    return fsii

# returns a lime explainer object
def lime_sampling(sampling_function, qsft_signal, b, **kwargs):
    print(qsft_signal.n)
    training_data = np.zeros((2, qsft_signal.n))
    training_data[1,:] = np.ones(qsft_signal.n)
    num_samples = get_num_samples(qsft_signal, b)
    print(num_samples)
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(training_data, mode='regression',
                                                            categorical_features=range(qsft_signal.n),
                                                            kernel_width=25)  # used in LimeTextExplainer
    lime_values = lime_explainer.explain_instance(np.ones(qsft_signal.n), sampling_function,
                                                  num_samples=num_samples,
                                                  num_features=qsft_signal.n,
                                                  distance_metric='cosine')  # used in LimeTextExplainer
    flush()
    return lime_values

# Returns uniform queries  
def uniform_sampling(sampling_function, qsft_signal,b, **kwargs):
    num_samples = get_num_samples(qsft_signal, b)
    queries = np.random.choice(2, size=(num_samples, qsft_signal.n))
    samples = sampling_function(queries)
    flush()
    return queries, samples

def assign_sampler(sampling_type,sampling_function, qsft_signal = None, order = None, b = None):
    if sampling_type == 'shapley':
        return shapley_sampling(sampling_function, qsft_signal, order, b)
    elif sampling_type == 'lime':
        return lime_sampling(sampling_function, qsft_signal, b)
    elif sampling_type == 'uniform':
        return uniform_sampling(sampling_function, qsft_signal, b)
    else:
        raise NotImplementedError(f"Sampling mechanism {sampling_type} not implemented")

# runs sampling and saves signals 
# If a signal is already saved, it loads it 
def run_sampling(model, explicand, sampling_function, b = 3, sampling_set = SAMPLING_SET, 
                save_dir = None, order = 1,num_test_samples = 10000, verbose = True, **kwargs):

    if verbose:
        print(f"Running sampling for explicand {explicand}")
    # Save explicand information 
    explicand_info = copy(explicand)
    n = len(explicand['input'])
    os.makedirs(save_dir, exist_ok=True)
    with open(f'{save_dir}/explicand_information.pickle', 'wb') as handle:
        pickle.dump(explicand_info, handle, protocol=pickle.HIGHEST_PROTOCOL)

    signals = {}
    # Get test samples 
    if not os.path.exists(f'{save_dir}/test_samples.parquet'):
        test_queries = np.random.choice(2, size=(num_test_samples, n))
        test_samples = sampling_function(test_queries)
        test_df = pd.DataFrame(test_queries, columns = [f'loc_{i}' for i in range(n)])
        test_df['target'] = test_samples
        test_df.to_parquet(f'{save_dir}/test_samples.parquet')
        flush()
    else:
        test_df = pd.read_parquet(f'{save_dir}/test_samples.parquet')
        test_queries = test_df.drop(columns=['target']).values
        test_samples = test_df['target'].values
        signals['test_queries'] = test_queries
        signals['test_samples'] = test_samples

    # get qsft signal 
    qsft_signal, num_samples = sampling_strategy(sampling_function, b, n, save_dir) 
    signals['qsft_signal'] = qsft_signal
    flush()

    # compute other sampling methods 
    for sampler in tqdm(sampling_set):
        if sampler == 'qsft':
            continue
        else:
            if os.path.exists(f'{save_dir}/{sampler}_{b}_signal.pickle'):
                with open(f'{save_dir}/{sampler}_{b}_signal.pickle', 'rb') as handle:
                    signal = pickle.load(handle)
            else:
                signal = assign_sampler(sampler, sampling_function, qsft_signal, order, b)
                with open(f'{save_dir}/{sampler}_{b}_signal.pickle', 'wb') as handle:
                    pickle.dump(signal, handle, protocol=pickle.HIGHEST_PROTOCOL)
            signals[sampler] = signal


    return signals






# Reconstruction methods 

def linear(signal, b, order=1, **kwargs):
    return fit_regression('linear', {'locations': qary_ints_low_order(signal.n, 2, order).T}, signal, signal.n, b)[0]

def lasso(signal, b, order=1, **kwargs):
    return fit_regression('lasso', {'locations': qary_ints_low_order(signal.n, 2, order).T}, signal, signal.n, b)[0]

def faith_banzhaf(signal, b, order=1, **kwargs):
    return fit_regression('lasso', {'locations': qary_ints_low_order(signal.n, 2, order).T}, signal, signal.n, b,
                          fourier_basis=False)[0]

def LIME(lime_values, qsft_signal, **kwargs):
    output = {}
    output[tuple([0] * qsft_signal.n)] = lime_values.intercept[1]
    for loc, val in lime_values.local_exp[1]:
        ohe_loc = [0] * qsft_signal.n
        ohe_loc[loc] = 1
        output[tuple(ohe_loc)] = val
    return mobius_to_fourier(output)


def faith_shapley(shapiq_explainer,n,  **kwargs):
    
    mobius_dict = {}
    for interaction, ref in shapiq_explainer.interaction_lookup.items():
        loc = [0] * n
        for ele in interaction:
            loc[ele] = 1
        mobius_dict[tuple(loc)] = shapiq_explainer.values[ref]
    return mobius_to_fourier(mobius_dict)


def qsft_hard(signal, b, t=5, **kwargs):
    return support_recovery("hard", signal, b, t=t)["transform"]


def qsft_soft(signal, b, t=5, **kwargs):
    return support_recovery("soft", signal, b, t=t)["transform"]

def banzhaf(uniform_signal, b, **kwargs):
    uniform_queries, uniform_samples = uniform_signal
    coordinates = uniform_queries
    values = np.array(uniform_samples)
    n = coordinates.shape[1]

    banzhaf_dict = {}
    for idx in range(n):
        mask = coordinates[:, idx] > 0.5
        if sum(mask) == 0 or sum(mask) == len(coordinates):
            banzhaf_value_idx = 0
        else:
            banzhaf_value_idx = ((1 / np.sum(mask)) * np.sum(values[mask])) - ((1 / np.sum(1 - mask)) * np.sum(values[1 - mask]))
        loc = [0] * n
        loc[idx] = 1
        banzhaf_dict[tuple(loc)] = banzhaf_value_idx
    return mobius_to_fourier(banzhaf_dict)





# Other utilities
def get_methods(method_list,max_order):
    ordered_methods = []
    for regression in ['linear', 'lasso', 'faith_banzhaf', 'faith_shapley']:
        if regression in method_list:
            ordered_methods += [(regression, order) for order in range(1, max_order + 1)]
            method_list.remove(regression)
    ordered_methods += [(method, 0) for method in method_list]
    return ordered_methods

    #return signal.sampling_function(np.random.choice(2, size=(b, signal.n)))

def flush(): 
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    




# def get_and_save_samples(sampling_function, type = 'test', num_test_samples = 10000, n = 128, save_dir = None, use_cache = True, qsft_signal = None):
#     assert type in ['test','uniform', 'shapley', 'lime']
    
#     if type == 'test':
#         if use_cache and os.path.exists(f'{save_dir}/test_samples.parquet'):
#             try: 
#                 test_df = pd.read_parquet(f'{save_dir}/test_samples.parquet')
#                 test_labels = test_df['target'].values
#                 test_queries = test_df.drop(columns=['target']).values
#                 saved_samples_test = test_queries, test_labels
#                 return saved_samples_test
#             except Exception as e:
#                 print(f"Error reading test samples: {e}")
#                 os.remove(f'{save_dir}/test_samples.parquet')
#                 return get_and_save_samples(sampling_function, type = 'test', num_test_samples = num_test_samples, n = n, save_dir = save_dir, use_cache = False, qsft_signal = None)
#         else:
#             os.makedirs(save_dir, exist_ok=True)
#             query_indices_test = np.random.choice(2, size=(num_test_samples, n))
#             saved_samples_test = query_indices_test, sampling_function(query_indices_test)
#             test_queries_pd = pd.DataFrame(query_indices_test, columns = [f'loc_{i}' for i in range(n)])
#             test_queries_pd['target'] = saved_samples_test[1]
#             test_queries_pd.to_parquet(f'{save_dir}/test_samples.parquet')
#             return saved_samples_test
#     else:
#         if use_cache and os.path.exists(f'{save_dir}/{type}_signal.pickle'):
#             try:
#                 with open(f'{save_dir}/{type}_signal.pickle', 'rb') as handle:
#                     signal = pickle.load(handle)
#                 return signal
#             except Exception as e:
#                 print(f"Error reading {type} signal: {e}")
#                 os.remove(f'{save_dir}/{type}_signal.pickle')
#                 return get_and_save_samples(sampling_function, type = type, num_test_samples = num_test_samples, n = n, save_dir = save_dir, use_cache = False, qsft_signal = qsft_signal)
#         else:
#             os.makedirs(save_dir, exist_ok=True)
#             signal = BatchedAlternative_Sampler(type, sampling_function, qsft_signal, n)
#             with open(f'{save_dir}/{type}_signal.pickle', 'wb') as handle:
#                 pickle.dump(signal, handle, protocol=pickle.HIGHEST_PROTOCOL)
#             return signal





# class BatchedAlternative_Sampler:
#     def __init__(self, type, sampling_function, qsft_signal, n):
#         assert type in ["uniform", "shapley", "lime"]
#         self.queries_finder = {
#             "uniform": self.uniform_queries,
#             "shapley": self.shapley_queries,
#             "lime": self.lime_queries
#         }.get(type, NotImplementedError())
#         self.n = n
#         self.all_queries = []
#         self.all_samples = []
#         self.query_type = type
#         for m in range(len(qsft_signal.all_samples)):
#             queries_subsample = []
#             for d in range(len(qsft_signal.all_samples[0])):
#                 queries = self.queries_finder(len(qsft_signal.all_queries[m][d]))
#                 if type == "shapley" and m == 0 and d == 0:
#                     queries[0, :] = np.zeros(n)
#                     queries[1, :] = np.ones(n)
#                 queries_subsample.append(queries)
#             self.all_queries.append(queries_subsample)
#         query_matrix = np.concatenate([np.concatenate(queries_subsample, axis=0) for queries_subsample in self.all_queries], axis=0)
#         samples = sampling_function(query_matrix)
#         #self.flattened_queries = query_matrix
#         #self.flattened_samples = samples
#         count = 0
#         for queries_subsample in self.all_queries: # list of numpy arrays
#             samples_subsample = []
#             for query in queries_subsample:
#                 samples_subsample.append(samples[count: count + len(query)])
#                 count += len(query)
#             self.all_samples.append(samples_subsample)

#     def uniform_queries(self, num_samples):
#         return np.random.choice(2, size=(num_samples, self.n))

#     def shapley_queries(self, num_samples):
#         shapley_kernel = np.array([1 / (comb(self.n, d) * d * (self.n - d)) for d in range(1, self.n)])
#         degrees = np.random.choice(range(1, self.n), size=num_samples, replace=True,
#                                    p=shapley_kernel / np.sum(shapley_kernel))
#         queries = []
#         for sample in range(num_samples):
#             q = np.zeros(self.n)
#             q[np.random.choice(range(self.n), size=degrees[sample], replace=False)] = 1
#             queries.append(q)
#         return np.array(queries)

#     def lime_queries(self, num_samples):
#         exponential_kernel = np.array([np.sqrt(np.exp(-(self.n - d) ** 2 / 25 ** 2)) for d in range(1, self.n)])
#         degrees = np.random.choice(range(1, self.n), size=num_samples, replace=True,
#                                    p=exponential_kernel / np.sum(exponential_kernel))
#         queries = []
#         for sample in range(num_samples):
#             q = np.zeros(self.n)
#             q[np.random.choice(range(self.n), size=degrees[sample], replace=False)] = 1
#             queries.append(q)
#         return np.array(queries)
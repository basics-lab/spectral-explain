from spectral_explain.support_recovery import support_recovery
from spectral_explain.qsft.qsft import fit_regression
from spectral_explain.qsft.utils import qary_ints_low_order
from spectral_explain.utils import mobius_to_fourier, estimate_r2
from spectral_explain.support_recovery import sampling_strategy, get_num_samples
import numpy as np
import copy
from copy import deepcopy
import dill as pickle
from math import comb
import torch
import gc
import time
import shapiq
from copy import copy
from tqdm import tqdm
import lime.lime_tabular
import os, pandas as pd
import warnings
warnings.filterwarnings("ignore")

# def lime(signal, b, **kwargs):
#     return fit_regression('lasso', {'locations': qary_ints_low_order(signal.n, 2, 1).T}, signal, signal.n, b)[0]

#SAMPLING_SET = ['qsft', 'uniform', 'lime', 'shapley','faith_shapley','shapley_taylor'] #shapley, faith_shapley
#METHODS = ['linear', 'lasso', 'lime', 'qsft_hard', 'qsft_soft', 'faith_shapley']#, 'faith_banzhaf']
METHODS = ['linear','lasso', 'lime', 'qsft_hard', 'qsft_soft', 'banzhaf', 'faith_banzhaf', 'shapley', 'shapley_taylor','faith_shapley'] #'faith_shapley']

SAMPLING_TO_METHOD = {'uniform': 'linear', 'uniform': 'lasso', 'shapley': 'shapley','shapley': 'faith_shapley', 
                      'lime': 'LIME', 'qsft': 'qsft_hard', 'qsft': 'qsft_soft', 'faith_shapley': 'faith_shapley', 'uniform': 'faith_banzhaf'}

METHOD_TO_SAMPLING = {'linear': 'uniform', 'lasso': 'uniform', 'shapley': 'shapley','faith_shapley': 'shapley', 
                      'lime': 'lime', 'qsft_hard': 'qsft', 'qsft_soft': 'qsft', 'faith_banzhaf': 'uniform'}
# Sampling methods 

class UniformSampler:
    def __init__(self, qsft_signal, uniform_signal):
        self.n = qsft_signal.n
        uniform_queries, uniform_samples = uniform_signal
        self.all_queries = []
        self.all_samples = []
        self.flattened_queries = uniform_queries
        self.flattened_samples = uniform_samples
        count = 0
        for m in range(len(qsft_signal.all_samples)):
            queries_subsample = []
            series_subsample = []
            for d in range(len(qsft_signal.all_samples[0])):
                queries = uniform_queries[count: count + len(qsft_signal.all_queries[m][d]),:]
                samples = uniform_samples[count: count + len(qsft_signal.all_samples[m][d])]
                queries_subsample.append(queries)
                series_subsample.append(samples)
                count += len(qsft_signal.all_queries[m][d])
            self.all_queries.append(queries_subsample)
            self.all_samples.append(series_subsample)


def shapley_sampling(sampling_function, qsft_signal, num_samples, index = "SV", order = 1, **kwargs):
    explainer = shapiq.Explainer(
        model= sampling_function,
        data=np.zeros((1,qsft_signal.n)),
        index=index,
        max_order=order
    )
    shapley = explainer.explain(np.ones((1, qsft_signal.n)), budget=num_samples)
    flush()
    return shapley


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

# Assigns a sampler to a sampling type 
def assign_sampler(sampling_type,sampling_function, qsft_signal = None, order = None, b = None, num_samples = None):
    if sampling_type in ['SV', 'FSII', 'STII']:
        return shapley_sampling(sampling_function, qsft_signal = qsft_signal, num_samples = num_samples, index = sampling_type, order = order)
    elif sampling_type == 'lime':
        return lime_sampling(sampling_function, qsft_signal, b)
    elif sampling_type == 'uniform':
        return uniform_sampling(sampling_function, qsft_signal, b)
    else:
        raise NotImplementedError(f"Sampling mechanism {sampling_type} not implemented")

# Checks if the signal is already saved 
def check_signal_exists(save_dir, method, b, order):
    if method == 'test':
        return os.path.exists(f'{save_dir}/test_samples.parquet')
    return os.path.exists(f'{save_dir}/{method}_b{b}_order{order}.pickle')



# runs sampling and saves signals 
# If a signal is already saved, it loads it 
def run_sampling(explicand, sampling_function, Bs = [4,6,8], save_dir = None, 
                 max_order = 4,num_test_samples = 10000, verbose = True, **kwargs):

    if verbose:
        print(f"Running sampling for explicand {explicand}")
    # Save explicand information 
    explicand_info = copy(explicand)
    n = len(explicand['input'])
    os.makedirs(save_dir, exist_ok=True)
    with open(f'{save_dir}/explicand_information.pickle', 'wb') as handle:
        pickle.dump(explicand_info, handle, protocol=pickle.HIGHEST_PROTOCOL)

    signals = {}
    if check_signal_exists(save_dir, 'test', 0, 0):
        pass
    else:
        test_queries  = np.random.choice(2, size=(num_test_samples, n))
        test_samples = sampling_function(test_queries)
        test_df = pd.DataFrame(test_queries, columns = [f'loc_{i}' for i in range(n)])
        test_df['target'] = test_samples
        test_df.to_parquet(f'{save_dir}/test_samples.parquet')

   

    # get qsft signal 
    qsft_signal, num_samples = sampling_strategy(sampling_function, max(Bs), n, save_dir) 
    signals['qsft_signal'] = qsft_signal
    num_samples_per_b = lambda b,signal: len(signal.all_samples) * len(signal.all_samples[0]) * (2**b)
    flush()

    # get other signals
    uniform_sampling_methods = [('uniform', Bs[-1],0)]
    shapley_sampling_methods = [('SV', b, 1) for b in Bs]
    faith_shapley_sampling_methods = [('FSII', b, order) for order in range(1,max_order+1) for b in Bs]
    shapley_taylor_sampling_methods = [('STII', b, order) for order in range(1,max_order+1) for b in Bs]
    lime_sampling_methods = [('lime', b, 1) for b in Bs]
    sampling_methods =  uniform_sampling_methods + shapley_sampling_methods + faith_shapley_sampling_methods + shapley_taylor_sampling_methods + lime_sampling_methods
    for method, b, order in sampling_methods:
        print(f'Running {method} sampling for b = {b} and order = {order}')
        num_samples = num_samples_per_b(b, qsft_signal)
        if os.path.exists(f'{save_dir}/{method}_b{b}_order{order}.pickle'):
            with open(f'{save_dir}/{method}_b{b}_order{order}.pickle', 'rb') as handle:
                signal = pickle.load(handle)
        else:
            if (n >= 64 and order >= 2) or (n >= 32 and order >= 3) or (n >= 16 and order >= 4):
                continue
            else:
                signal = assign_sampler(sampling_type = method, sampling_function = sampling_function, qsft_signal = qsft_signal, 
                                        order = order, b = b, num_samples = num_samples)
                with open(f'{save_dir}/{method}_b{b}_order{order}.pickle', 'wb') as handle:
                    pickle.dump(signal, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'Finished {method} sampling for b = {b} and order = {order}')
        signals[method] = signal


def linear(signal, b, order=1, **kwargs):
    return fit_regression('linear', {'locations': qary_ints_low_order(signal.n, 2, order).T}, signal, signal.n, b)[0]

def lasso(signal, b, order=1, **kwargs):
    return fit_regression('lasso', {'locations': qary_ints_low_order(signal.n, 2, order).T}, signal, signal.n, b)[0]

def faith_banzhaf(signal, b, order=1, **kwargs):
    return fit_regression('lasso', {'locations': qary_ints_low_order(signal.n, 2, order).T}, signal, signal.n, b,
                          fourier_basis=False)[0]

def LIME(signal, n, **kwargs):
    output = {}
    output[tuple([0] * n)] = signal.intercept[1]
    for loc, val in signal.local_exp[1]:
        ohe_loc = [0] * n
        ohe_loc[loc] = 1
        output[tuple(ohe_loc)] = val
    return mobius_to_fourier(output)



def shapley(signal,n, **kwargs):
    
    mobius_dict = {}
    for interaction, ref in signal.interaction_lookup.items():
        loc = [0] * n
        for ele in interaction:
            loc[ele] = 1
        mobius_dict[tuple(loc)] = signal.values[ref]
    return mobius_to_fourier(mobius_dict)


def qsft_hard(signal, b, t=5, **kwargs):
    return support_recovery("hard", signal, b, t=t)["transform"]


def qsft_soft(signal, b, t=5, **kwargs):
    return support_recovery("soft", signal, b, t=t)["transform"]

def banzhaf(signal, b, order = 1, t = 5, **kwargs):
    uniform_queries = signal.flattened_queries
    uniform_samples = signal.flattened_samples
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

    #return signal.sampling_function(np.random.choice(2, size=(b, signal.n)))

def _get_methods(methods,max_order):
        method_list = deepcopy(methods)
        ordered_methods = []
        for regression in ['linear', 'lasso', 'faith_banzhaf']:
            if regression in method_list:
                ordered_methods += [(regression, order) for order in range(1, max_order + 1)]
                method_list.remove(regression)
        ordered_methods += [(method, 0) for method in method_list]
        return ordered_methods

def _get_signal(sampling_type, b, n, order,save_dir):
        if sampling_type == 'qsft':
            return sampling_strategy(lambda X: 1.0, b, n, save_dir) [0]
        else:
            with open(f'{save_dir}/{sampling_type}_b{b}_order{order}.pickle', 'rb') as handle:
                print(f'Loading {sampling_type} signal')
                signal = pickle.load(handle)            
                return signal

def _get_reconstruction(method, signal, order, b, n, t=5):
        start_time = time.time()
        reconstruction = {
            "linear": linear,
            "lasso": lasso,
            "lime": LIME,
            "qsft_hard": qsft_hard,
            "qsft_soft": qsft_soft,
            "banzhaf": banzhaf,
            "faith_banzhaf": faith_banzhaf,
            "FSII": shapley,
            "STII": shapley,
            "SV": shapley, 
            "lime": LIME
        }.get(method, NotImplementedError())(signal = signal, b = b, order=order, t=t, n = n)
        end_time = time.time()
        return reconstruction
 

def flush(): 
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    
def get_and_evaluate_reconstruction(explicand, Bs = [4,6,8], save_dir = 'experiments/results', 
                                    max_order = 4, t = 5, **kwargs):
   

    n = len(explicand['input'])
    test_samples = pd.read_parquet(f'{save_dir}/test_samples.parquet')
    test_samples = test_samples.drop(columns=['target']).values, test_samples['target'].values

    # load qsft signal 
    qsft_signal = _get_signal(sampling_type = 'qsft', b = Bs[-1], n = n, order = 0, save_dir = save_dir)
    
    reconstruction_dict = {}
    r2_results = {}
    reg_methods = [('linear', i) for i in range(max_order+1)] + [('lasso', i) for i in range(max_order+1)] + [('faith_banzhaf', i) for i in range(max_order+1)]
    qsft_methods = [('qsft_hard', 0), ('qsft_soft', 0)]
    shapley_methods = [('SV', 1)] + [('FSII', i) for i in range(1,max_order+1)] + [('STII', i) for i in range(1,max_order+1)]
    lime_methods = [('lime', 1)]
    all_methods = reg_methods + qsft_methods + shapley_methods + lime_methods
    for b in Bs:
        for ordered_method in all_methods:
            method, order = ordered_method
            print(f'Running {method} reconstruction for b = {b} and order = {order}')
            if method == 'qsft_hard' or method == 'qsft_soft':
                qsft_reconstruction = _get_reconstruction(method, qsft_signal, order, b, n, t)
                reconstruction_dict[ordered_method] = qsft_reconstruction
            elif (n >= 64 and order >= 2) or (n >= 32 and order >= 3) or (n >= 16 and order >= 4):
                reconstruction_dict[ordered_method] = None
                continue
            elif ordered_method in reg_methods:
                uniform_signal = _get_signal(sampling_type = 'uniform', b = Bs[-1], n = n, order = 0, save_dir = save_dir)
                signal = UniformSampler(qsft_signal, uniform_signal)
                reconstruction_dict[ordered_method] = _get_reconstruction(method, signal, order, b, n, t)
            elif ordered_method in shapley_methods:
                shap_signal = _get_signal(sampling_type =  method, b = b, n = n, order = order, save_dir = save_dir)
                reconstruction_dict[ordered_method] = _get_reconstruction(method, shap_signal, order, b, n, t)
            elif method == 'lime':
                lime_signal = _get_signal(sampling_type =  method, b = b, n = n, order = order, save_dir = save_dir)
                reconstruction_dict[ordered_method] = _get_reconstruction(method, lime_signal, order, b, n, t)
            r2 = estimate_r2(reconstruction_dict[ordered_method], test_samples)
            r2_results[(method,order,b)] = {'r2': r2, 'reconstruction': reconstruction_dict[ordered_method], 'samples': get_num_samples(qsft_signal,b)}
            print(f'Finished {method} reconstruction for b = {b} and order = {order} with r2 score {r2_results[(method,order,b)]["r2"]} and {r2_results[(method,order,b)]["samples"]} samples')
    return reconstruction_dict
    
    # Run reconstruction 
    #print(get_methods(METHODS, max_order))
    # method_list = _get_methods(METHODS, max_order)
    # r2_results = {}

    # for ordered_method in method_list:
    #     method, order = ordered_method
    #     method_name = f'{ordered_method[0]}{ordered_method[1]}'
    #     time_start = time.time()
    #     if (n >= 64 and ordered_method[1] >= 2) or (n >= 32 and ordered_method[1] >= 3) or (n >= 16 and ordered_method[1] >= 4):
    #             continue 
    #     if os.path.exists(f'{save_dir}/reconstruction/{method_name}_{b}.pickle'):
    #         reconstruction = pickle.load(open(f'{save_dir}/reconstruction/{method_name}_{b}.pickle', 'rb'))
    #         r2 = estimate_r2(reconstruction, test_samples)
    #     else:
    #         print(f'Running {ordered_method} reconstruction')
    #         if method in ['shapley', 'faith_shapley', 'lime']:
    #             if METHOD_TO_SAMPLING[method] not in signal_dict:
    #                 signal_dict[METHOD_TO_SAMPLING[method]] = _get_signal(sampling_type = METHOD_TO_SAMPLING[method], b = b, n = n, save_dir = save_dir)
    #             signal = signal_dict[METHOD_TO_SAMPLING[method]]
    #             #all_reconstructions[method] = _get_reconstruction(method = method, signal = signal, order = order, b = b, t = t)
    #         elif method in ['qsft_hard', 'qsft_soft']:
    #             signal = _get_signal(sampling_type = 'qsft', b = b, n = n, save_dir = save_dir)
    #             signal_dict['qsft'] = signal
    #             #all_reconstructions[method] = _get_reconstruction(method = method, signal = signal, b = b, order = order, t = t)
    #         elif method in ['linear', 'lasso', 'faith_banzhaf','banzhaf']:
    #             if 'qsft' not in signal_dict:
    #                 signal_dict['qsft'] = _get_signal(sampling_type = 'qsft', b = b, n = n, save_dir = save_dir)
    #             if 'uniform' not in signal_dict:
    #                 signal_dict['uniform'] = _get_signal(sampling_type = 'uniform', b = b, n = n, save_dir = save_dir)
            
    #             signal = UniformSampler(signal_dict['qsft'], signal_dict['uniform'])
            
    #         reconstruction = _get_reconstruction(method = method, signal = signal, b = b, order = order, n = n, t = t)
    #         os.makedirs(f'{save_dir}/reconstruction', exist_ok=True)
    #         with open(f'{save_dir}/reconstruction/{method_name}_{b}.pickle', 'wb') as handle:
    #             pickle.dump(reconstruction, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #     r2 = estimate_r2(reconstruction, test_samples)
    #     print(f'{method} reconstruction finished in {time.time() - time_start} seconds with r2 score {r2}')
    #     r2_results[method] = r2
    # with open(f'{save_dir}/r2_results.pickle', 'wb') as handle:
    #     pickle.dump(r2_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

        
    #return all_reconstructions






      #if check_order_limit(order, n):
            #    all_reconstructions[method] = _get_reconstruction(method = method, signal = unif_signal, b = b, order = order, t = t)
            #else:
            #    all_reconstructions[method] = None
        # Save the reconstruction to a file
        # end_time = time.time()
        # print(f'Finished {method} reconstruction in {end_time - time_start} seconds')

        # with open(f'{save_dir}/reconstruction_{b}.pickle', 'wb') as handle:
        #     pickle.dump(all_reconstructions, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #break




            # compute other sampling methods 
    # for b in Bs:
    #     num_samples = num_samples_per_b(b)
    #     for sampler in tqdm(sampling_set):
    #         if sampler == 'qsft':
    #             continue
    #         else:

    #             if os.path.exists(f'{save_dir}/{sampler}_b{b}_signal.pickle'):
    #                 with open(f'{save_dir}/{sampler}_b{b}_signal.pickle', 'rb') as handle:
    #                     signal = pickle.load(handle)
    #             else:
    #                 for order in 
    #                 signal = assign_sampler(sampler, sampling_function, qsft_signal, order, b, num_samples)
    #                 with open(f'{save_dir}/{sampler}_{b}_signal.pickle', 'wb') as handle:
    #                     pickle.dump(signal, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #             signals[sampler] = signal
            
    # for sampler in tqdm(sampling_set):

    #     if sampler == 'qsft':
    #         continue
    #     else:
    #         if os.path.exists(f'{save_dir}/{sampler}_{b}_signal.pickle'):
    #             with open(f'{save_dir}/{sampler}_{b}_signal.pickle', 'rb') as handle:
    #                 signal = pickle.load(handle)
    #         else:
    #             if sampler == 'faith_shapley' and n >= faith_shapley_limit:
    #                 print(f"Skipping faith_shapley for explicand {explicand} because n >= {faith_shapley_limit}")
    #                 signal = None
    #             else:
    #                 signal = assign_sampler(sampler, sampling_function, qsft_signal, order, b)
    #             with open(f'{save_dir}/{sampler}_{b}_signal.pickle', 'wb') as handle:
    #                 pickle.dump(signal, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #         signals[sampler] = signal


    # return signals




# Reconstruction methods 

import numba
import numpy as np
import dill as pickle
import time
from tqdm import tqdm
import os
import shutil
import cProfile
import pstats
from spectral_explain.models.modelloader import get_model
from spectral_explain.support_recovery import sampling_strategy
from spectral_explain.utils import estimate_r2
from experiment_utils import linear, lasso, lime, qsft_hard, qsft_soft, faith_banzhaf, faith_shapley
from math import comb
import torch
# TODO:
# 1. 
# 5. Add a new method for the spectral explaine

def run_and_evaluate_method(method, samples, order, b, saved_samples_test, t=5):
    start_time = time.time()
    reconstruction = {
        "linear": linear,
        "lasso": lasso,
        "lime": lime,
        "qsft_hard": qsft_hard,
        "qsft_soft": qsft_soft,
        "faith_banzhaf": faith_banzhaf,
        "faith_shapley": faith_shapley
    }.get(method, NotImplementedError())(samples, b, order=order, t=t)
    end_time = time.time()
    return end_time - start_time, estimate_r2(reconstruction, saved_samples_test), reconstruction




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

SAMPLER_DICT = {
    "qsft_hard": "qsft",
    "qsft_soft": "qsft",
    "linear": "uniform",
    "lasso": "uniform",
    "faith_banzhaf": "uniform",
    "lime": "lime",
   "faith_shapley": "shapley"
}


def main():
    # choose TASK from parkinsons, cancer, sentiment,
    # sentiment_mini, similarity, similarity_mini,
    # comprehension, comprehension_mini, clinical
    # context_cite (HotpotQA, DROP)
    TASK = 'drop'
    DEVICE = 'auto'
    NUM_EXPLAIN = 1
    MAX_ORDER = 4
    MAX_B = 8
    SEED = 80
    num_test_samples = 10000
    METHODS = ['linear', 'lasso', 'qsft_hard', 'qsft_soft'] #'faith_banzhaf','lime' 'faith_shapley']

    sampler_set = set([SAMPLER_DICT[method] for method in METHODS])

    ordered_methods = []
    for regression in ['linear', 'lasso', 'faith_banzhaf', 'faith_shapley']:
        if regression in METHODS:
            ordered_methods += [(regression, order) for order in range(1, MAX_ORDER + 1)]
            METHODS.remove(regression)
    ordered_methods += [(method, 0) for method in METHODS]

    count_b = MAX_B - 2

    explicands, model = get_model(TASK, NUM_EXPLAIN, DEVICE, SEED)

    # results = {
    #     "samples": np.zeros((len(explicands), count_b)),
    #     "methods": {f'{method}_{order}': {'time': np.zeros((len(explicands), count_b)),
    #                                       'test_r2': np.zeros((len(explicands), count_b))}
    #                 for method, order in ordered_methods},
    #}

    results = {}

    np.random.seed(SEED)
    torch.cuda.empty_cache()
    for i, explicand in enumerate(explicands):
        explicand_results = {}
        print(explicand)
        print(f'sequence length: {len(explicand["input"])}')
        sample_id = explicand['id']
        n = model.set_explicand(explicand)
        sampling_function = lambda X: model.inference(X)

        query_indices_test = np.random.choice(2, size=(num_test_samples, n))
        saved_samples_test = query_indices_test, sampling_function(query_indices_test)

        explicand_results['test_queries'] = saved_samples_test[0]
        explicand_results['test_samples'] = saved_samples_test[1]
        explicand_results['sequence_length'] = n
        explicand_results['explicand'] = explicand
        explicand_results['original_answer'] = (model.original_decoded_output, model.original_output_token_ids[1:])


        unix_time_seconds = str(int(time.time()))
        if not os.path.exists(f'experiments/results/{TASK}/{sample_id}'):
            os.makedirs(f'experiments/results/{TASK}/{sample_id}')
        save_dir = f'experiments/results/{TASK}/{sample_id}' #+ unix_time_seconds

        torch.cuda.empty_cache()


        # Sample explanation function for choice of max b
        qsft_signal, num_samples = sampling_strategy(sampling_function, MAX_B, n, save_dir)
        explicand_results["samples"] = num_samples

        # Draws an equal number of uniform samples
        active_sampler_dict = {"qsft": qsft_signal}
        explicand_results["qsft"] = qsft_signal
        for sampler in tqdm(sampler_set):
            if sampler != "qsft":
                active_sampler_dict[sampler] = BatchedAlternative_Sampler(sampler, sampling_function, qsft_signal, n)
                explicand_results[sampler] = active_sampler_dict[sampler]

                print(f'finished {sampler} sampling')
        
        explicand_results["methods"] = {f'{method}_{order}': {'time': np.zeros((count_b)),
                                          'test_r2': np.zeros((count_b)),
                                          }
                    for method, order in ordered_methods}
        #explicand_results['reconstruction'] = {}
        
        for b in range(3, MAX_B + 1):
            print(f"b = {b}")
            for method, order in ordered_methods:
                method_str = f'{method}_{order}'
                samples = active_sampler_dict[SAMPLER_DICT[method]]
                if (order >= 2 and n >= 128) or (order >= 3 and n >= 32) or (order >= 4 and n >= 16):
                    explicand_results["methods"][method_str]["time"][b - 3] = np.nan
                    explicand_results["methods"][method_str]["test_r2"][b - 3] = np.nan
                else:
                    print(f"running {method_str}")
                    time_taken, test_r2, reconstruction = run_and_evaluate_method(method, samples, order, b, saved_samples_test)
                    explicand_results["methods"][method_str]["time"][b - 3] = time_taken
                    explicand_results["methods"][method_str]["test_r2"][b - 3] = test_r2
                    print(f"{method_str}: {np.round(test_r2, 3)} test r2 in {np.round(time_taken, 3)} seconds")
            print()
        for s in active_sampler_dict.values():
            del s
        #print(explicand_results)
        with open(f'experiments/results/{TASK}/{sample_id}/r2_results.pkl', 'wb') as handle:
            pickle.dump(explicand_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
      

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    numba.set_num_threads(8)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.print_stats(50)





  #results[sample_id] = explicand_results
    
    # with open(f'samples/results/{TASK}/agg_results.pkl', 'wb') as handle:
    #     pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # if not os.path.exists(f'samples/results/{TASK}'):
    #     os.makedirs(f'samples/results/{TASK}')
    # with open(f'samples/results/{TASK}/r2_results.pkl', 'wb') as handle:
    #     pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # with open(f'{TASK}_{unix_time_seconds}.pkl', 'wb') as handle:
    #     pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

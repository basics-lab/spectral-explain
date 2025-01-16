import numba
import numpy as np
import time
import pandas as pd
from tqdm import tqdm
import dill as pickle
import os, shutil, cProfile, pstats, gc, argparse
from spectral_explain.models.modelloader import get_model
from spectral_explain.support_recovery import sampling_strategy
from spectral_explain.utils import estimate_r2
from experiment_utils import linear, lasso, lime, qsft_hard, qsft_soft, faith_banzhaf, faith_shapley, BatchedAlternative_Sampler, get_and_save_samples, get_methods
import torch
from math import comb

SAMPLER_DICT = {
    "qsft_hard": "qsft",
    "qsft_soft": "qsft",
    "linear": "uniform",
    "lasso": "uniform",
    "faith_banzhaf": "uniform",
    "lime": "lime",
   "faith_shapley": "shapley"
}

METHODS = ['linear', 'lasso', 'lime', 'qsft_hard', 'qsft_soft', 'faith_shapley'] #'faith_banzhaf','lime' 'faith_shapley']

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

        
        

def flush():
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()




def main(task = 'drop', seed = 0, device = 'cuda:0', 
        MAX_B = 3, MAX_ORDER = 4, NUM_EXPLAIN = 1,
        num_test_samples = 100, use_cache = True):
    # choose TASK from parkinsons, cancer, sentiment,
    # sentiment_mini, similarity, similarity_mini,
    # comprehension, comprehension_mini, clinical
    # context_cite (HotpotQA, DROP)
    TASK, DEVICE, SEED, USE_CACHE = task, device, seed, use_cache
    MAX_ORDER, NUM_EXPLAIN, MAX_B, num_test_samples = MAX_ORDER, NUM_EXPLAIN, MAX_B, num_test_samples
    sampler_set = set([SAMPLER_DICT[method] for method in METHODS])
    ordered_methods = get_methods(METHODS, MAX_ORDER)
    count_b = MAX_B - 2
    explicands, model = get_model(TASK, NUM_EXPLAIN, DEVICE, SEED)
    np.random.seed(SEED)

  

        
    for i, explicand in enumerate(explicands):
        explicand_results = {}
        print(explicand)
        print(f'sequence length: {len(explicand["input"])}')
        sample_id = explicand['id']
        explicand_results['explicand'] = explicand
        n = model.set_explicand(explicand)
        explicand_results['sequence_length'] = n
        sampling_function = lambda X: model.inference(X)
        save_dir = f'experiments/results/{TASK}/{sample_id}' #+ unix_time_seconds

        saved_samples_test = get_and_save_samples(sampling_function, type = 'test', num_test_samples = num_test_samples
                                                  ,n = n, save_dir = save_dir, use_cache = USE_CACHE)
        explicand_results['original_answer'] = (model.original_decoded_output, model.original_output_token_ids[1:])

        flush()

        # Sample explanation function for choice of max b, caching automatically done
        qsft_signal, num_samples = sampling_strategy(sampling_function, MAX_B, n, save_dir)
        explicand_results["samples"] = num_samples

        # Draws an equal number of uniform samples
        active_sampler_dict = {"qsft": qsft_signal}
        explicand_results["qsft"] = qsft_signal
        for sampler in tqdm(sampler_set):
            if sampler != "qsft":
                active_sampler_dict[sampler] = get_and_save_samples(sampling_function, type = sampler, n = n, save_dir = save_dir,
                                                                    use_cache = USE_CACHE, qsft_signal = qsft_signal)
                print(f'finished {sampler} sampling')
        
        explicand_results["methods"] = {f'{method}_{order}': {'time': np.zeros((count_b)),
                                          'test_r2': np.zeros((count_b))} for method, order in ordered_methods}
        
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
    numba.set_num_threads(5)
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--task", type=str, default='drop')
    parser.add_argument("--max_b", type=int, default=3)
    parser.add_argument("--max_order", type=int, default=4)
    parser.add_argument("--num_explain", type=int, default=1)
    parser.add_argument("--num_test_samples", type=int, default=10000)
    parser.add_argument("--use_cache", type=bool, default=True)
    args = parser.parse_args()
    main(seed = args.seed, device = args.device, task = args.task,
         max_b = args.max_b, max_order = args.max_order, num_explain = args.num_explain,
         num_test_samples = args.num_test_samples, use_cache = args.use_cache)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('tottime')
    #stats.print_stats(50)





  #results[sample_id] = explicand_results
    
    # with open(f'samples/results/{TASK}/agg_results.pkl', 'wb') as handle:
    #     pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # if not os.path.exists(f'samples/results/{TASK}'):
    #     os.makedirs(f'samples/results/{TASK}')
    # with open(f'samples/results/{TASK}/r2_results.pkl', 'wb') as handle:
    #     pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # with open(f'{TASK}_{unix_time_seconds}.pkl', 'wb') as handle:
    #     pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


    # ordered_methods = []
    # for regression in ['linear', 'lasso', 'faith_banzhaf', 'faith_shapley']:
    #     if regression in METHODS:
    #         ordered_methods += [(regression, order) for order in range(1, MAX_ORDER + 1)]
    #         METHODS.remove(regression)
    # ordered_methods += [(method, 0) for method in METHODS]


      # results = {
    #     "samples": np.zeros((len(explicands), count_b)),
    #     "methods": {f'{method}_{order}': {'time': np.zeros((len(explicands), count_b)),
    #                                       'test_r2': np.zeros((len(explicands), count_b))}
    #                 for method, order in ordered_methods},
    #}                #active_sampler_dict[sampler] = BatchedAlternative_Sampler(sampler, sampling_function, qsft_signal, n)
                #explicand_results[sampler] = active_sampler_dict[sampler]


            # if not os.path.exists(f'experiments/results/{TASK}/{sample_id}'):
        #     os.makedirs(f'experiments/results/{TASK}/{sample_id}')
        # else:
        #     if USE_CACHE:
        #         print("Set use_cache = True and directory already exists")
        #         continue
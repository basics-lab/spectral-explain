import numba
import numpy as np
import time
import copy
import pandas as pd
from tqdm import tqdm
import dill as pickle
import shapiq
import os, shutil, cProfile, pstats, gc, argparse
from spectral_explain.models.modelloader import get_model
from spectral_explain.support_recovery import sampling_strategy, get_num_samples
from spectral_explain.utils import estimate_r2
from experiment_utils import linear, lasso, lime, qsft_hard, qsft_soft, faith_banzhaf, faith_shapley, BatchedAlternative_Sampler, get_and_save_samples, get_methods, flush
import torch
from math import comb


SAVE_DIR = f'experiments/results/'
METHODS = ['linear', 'lasso', 'lime', 'qsft_hard', 'qsft_soft', 'faith_shapley', 'faith_banzhaf']
SAMPLER_DICT = {
    "qsft_hard": "qsft",
    "qsft_soft": "qsft",
    "linear": "uniform",
    "lasso": "uniform",
    "lime": "lime",
    "faith_banzhaf": "uniform",
    "faith_shapley": "faith_shapley",
    "shapley": "shapley",
    "banzhaf": "uniform"
}
def main(task, MAX_ORDER):


    pass

if __name__ == "__main__":
    print("Starting main function")
    profiler = cProfile.Profile()
    profiler.enable()
    numba.set_num_threads(8)
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--task", type=str, default='drop')
    parser.add_argument("--MAX_B", type=int, default=8)
    parser.add_argument("--MIN_B", type=int, default=8)
    parser.add_argument("--MAX_ORDER", type=int, default=4)
    parser.add_argument("--NUM_EXPLAIN", type=int, default=25)
    parser.add_argument("--num_test_samples", type=int, default=10000)
    parser.add_argument("--use_cache", type=bool, default=True)
    parser.add_argument("--run_sampling", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=512)
    
    
    # parser.add_argument("--ALL_Bs", type=bool, default=False)
    # args = parser.parse_args()
    # main(args.task, args.seed, args.device, args.MAX_B, args.MAX_ORDER, args.NUM_EXPLAIN, 
    #      args.num_test_samples, args.batch_size, args.use_cache, args.run_sampling, args.ALL_Bs)
    
 





# def run_and_evaluate_method(method, samples, order, b, saved_samples_test, t=5):
#     start_time = time.time()
#     reconstruction = {
#         "linear": linear,
#         "lasso": lasso,
#         "lime": lime,
#         "qsft_hard": qsft_hard,
#         "qsft_soft": qsft_soft,
#         "faith_banzhaf": faith_banzhaf,
#         "faith_shapley": faith_shapley
#     }.get(method, NotImplementedError())(samples, b, order=order, t=t)
#     end_time = time.time()
#     return end_time - start_time, estimate_r2(reconstruction, saved_samples_test), reconstruction

    

# def evaluate_r2(explicand, TASK, MAX_B, MAX_ORDER, ALL_Bs):
#     explicand_results = {}
#     active_sampler_dict = {}
#     n = len(explicand['input'])
#     count_b = MAX_B - 2
#     ordered_methods = get_methods(METHODS, MAX_ORDER)
#     save_dir = f'/scratch/users/{os.getenv("USER")}/results/{TASK}/{explicand["id"]}'
#     explicand_results['explicand'] = explicand
#     explicand_results['sequence_length'] = n
#     try: 
#         explicand_results['original_answer'] = explicand['original_answer']
#     except:
#         explicand_results['original_answer'] = None
#     sampling_function = lambda X: None # Dummy sampling function since resuls are saved
#     #Load QSFT samples
#     qsft_signal, num_samples = sampling_strategy(sampling_function, MAX_B, n, save_dir)
#     active_sampler_dict["qsft"] = qsft_signal
#     explicand_results["samples"] = num_samples
#     print(f"Loaded QSFT samples for explicand {explicand['id']}")

#     # Load other samples for other sampling schemes
#     sampler_set = set([SAMPLER_DICT[method] for method in METHODS])
#     sampling_filenames = {f'{save_dir}/{sampler}.pkl': sampler for sampler in sampler_set}
#     for filename in os.listdir(save_dir):
#         if filename == 'test_samples.parquet':
#             test_df = pd.read_parquet(os.path.join(save_dir, filename))
#             test_queries = test_df.drop(columns = ['sample_id']).to_numpy()
#             test_samples  = test_df['target'].values
#             saved_samples_test = test_queries, test_samples
#             print(f"Loaded test samples for explicand {explicand['id']}")
#         else:
#             if filename in sampling_filenames:
#                 with open(os.path.join(save_dir, filename), 'rb') as file:
#                     active_sampler_dict[sampling_filenames[filename]] = pickle.load(file)
#                 print(f"Loaded {sampling_filenames[filename]} samples for explicand {explicand['id']}")
    
#     for b in range(3 if ALL_Bs else MAX_B, MAX_B + 1):
#         if os.path.exists(f'{save_dir}/r2_results_b{b}.pkl'):
#             explicand_results_b = pickle.load(open(f'{save_dir}/r2_results_b{b}.pkl', 'rb'))
#         else:
#             explicand_results_b = copy(explicand_results)
#             for method, order in get_methods(METHODS, MAX_ORDER):
#                 method_str = f'{method}_{order}'
#             samples = active_sampler_dict[SAMPLER_DICT[method]]
#             if (order >= 2 and n >= 128) or (order >= 3 and n >= 32) or (order >= 4 and n >= 16):
#                 explicand_results_b["methods"][method_str]["time"][b - 3] = np.nan
#                 explicand_results_b["methods"][method_str]["test_r2"][b - 3] = np.nan
#             else:
#                 print(f"running {method_str}")
#                 time_taken, test_r2, reconstruction = run_and_evaluate_method(method, samples, order, b, saved_samples_test)
#                 explicand_results_b["methods"][method_str]["time"][b - 3] = time_taken
#                 explicand_results_b["methods"][method_str]["test_r2"][b - 3] = test_r2
#                 explicand_results_b["methods"][method_str]["reconstruction"] = reconstruction
#                 print(f"{method_str}: {np.round(test_r2, 3)} test r2 in {np.round(time_taken, 3)} seconds")
#             with open(f'{save_dir}/r2_results_b{b}.pkl', 'wb') as handle:
#                 pickle.dump(explicand_results_b, handle, protocol=pickle.HIGHEST_PROTOCOL)
#         explicand_results[b] = explicand_results_b
#     with open(f'{save_dir}/r2_results.pkl', 'wb') as handle:
#         pickle.dump(explicand_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
#         #print()

   

# def main(TASK = 'drop', SEED = 0, DEVICE = 'cuda:0', 
#         MAX_B = 3, MAX_ORDER = 4, NUM_EXPLAIN = 100,
#         num_test_samples = 10000, batch_size = 128,
#         run_sampling = True, ALL_Bs = False):
   
#     print("Loading model and explicands")
#     explicands, model = get_model(TASK, NUM_EXPLAIN, DEVICE, SEED)
#     print("Finished loading model and explicands")
#     TASK, DEVICE, SEED, USE_CACHE = TASK, DEVICE, SEED, USE_CACHE
#     MAX_ORDER, NUM_EXPLAIN, MAX_B, num_test_samples = MAX_ORDER, NUM_EXPLAIN, MAX_B, num_test_samples
#     np.random.seed(SEED)
#     for explicand in explicands:
#         if run_sampling:
#             model.batch_size = batch_size
#             collect_data(explicand, model, TASK, num_test_samples, USE_CACHE, MAX_B)
#         else:
#             print(f"Computing R2 for explicand {explicand['id']}")
#             evaluate_r2(explicand, TASK, MAX_B, MAX_ORDER, ALL_Bs)

  

        

    
# choose TASK from parkinsons, cancer, sentiment,
# sentiment_mini, similarity, similarity_mini,
# comprehension, comprehension_mini, clinical
# context_cite (HotpotQA, DROP)

        
# def collect_data(explicand, model, TASK, num_test_samples, USE_CACHE, MAX_B):
#     sampler_set = set([SAMPLER_DICT[method] for method in METHODS])
#     print(explicand)
#     print(f'sequence length: {len(explicand["input"])}')
#     sample_id = explicand['id']
#     n = model.set_explicand(explicand)
#     sampling_function = lambda X: model.inference(X)
#     save_dir = f'/scratch/users/{os.getenv("USER")}/results/{TASK}/{sample_id}' #+ unix_time_seconds
#     os.makedirs(save_dir, exist_ok=True)

#     saved_samples_test = get_and_save_samples(sampling_function, type = 'test', num_test_samples = num_test_samples
#                                                 ,n = n, save_dir = save_dir, use_cache = USE_CACHE)
    


#     flush()

#     # Sample explanation function for choice of max b, caching automatically done
#     qsft_signal, num_samples = sampling_strategy(sampling_function, MAX_B, n, save_dir)
#     flush()

#     # Draws an equal number of uniform samples
#     active_sampler_dict = {"qsft": qsft_signal}
#     for sampler in tqdm(sampler_set):
#         if sampler == "faith_shapley":
#             continue
#         elif sampler != "qsft":
#             active_sampler_dict[sampler] = get_and_save_samples(sampling_function, type = sampler, n = n, save_dir = save_dir,
#                                                                 use_cache = USE_CACHE, qsft_signal = qsft_signal)
#             print(f'finished {sampler} sampling')
#             flush()
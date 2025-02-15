import numba
import numpy as np
from pyarrow import list_

from spectral_explain.dataloader import get_dataset
from spectral_explain.models.modelloader import get_model, HotPotQAModel, QAModel
from spectral_explain.support_recovery import sampling_strategy, support_recovery
from spectral_explain.qsft.qsft import fit_regression, transform_via_amp
from spectral_explain.baselines import neural_network
from experiment_utils import linear, lasso, qsft_hard, qsft_soft, get_and_evaluate_reconstruction
from spectral_explain.qsft.utils import qary_ints_low_order
import pickle
import time
import os
from math import prod
from spectral_explain.utils import estimate_r2
import shutil
import cProfile
import pstats
import argparse
from joblib import Parallel, delayed
SAVE_DIR = f'experiments/results/'
SUBTRACT_DIST = 7
SUBTRACTION_METHODS = {
    'linear_0': 'greedy',
    "linear_1": 'greedy',
    "linear_2": 'greedy',
    'linear_3': 'greedy',
    'linear_4': 'greedy',
    'lasso_0': 'greedy',
    "lasso_1": 'greedy',
    "lasso_2": 'greedy',
    "lasso_3": 'greedy',
    "lasso_4": 'greedy',
    "amp_1": 'greedy',
    "amp_2": 'greedy',
    "amp_3": 'greedy',
    "amp_4": 'greedy',
    "qsft_hard_0": 'greedy',
    "qsft_soft_0": 'greedy',
    "SV_1": 'greedy',
    "FSII_1": 'greedy',
    "FSII_2": 'greedy',
    "FSII_3": 'greedy',
    "FSII_4": 'greedy',
    "STII_1": 'greedy',
    "STII_2": 'greedy',
    "STII_3": 'greedy',
    "STII_4": 'greedy',
    "lime_1": 'greedy',
    'faith_banzhaf_1': 'greedy',
    'faith_banzhaf_2': 'greedy',
    'faith_banzhaf_3': 'greedy',
    'faith_banzhaf_4': 'greedy',
        }

def vec_to_index(vec):
    return np.nonzero(vec)[0]

def eval_function(x, list_of_interactions):
    return sum(v*prod((-1) ** x[i] for i in y) for y, v in list_of_interactions)

def make_dependancy_tree(list_of_interactions, n):
    tree = [[] for _ in range(n)]
    all_on_sum = 0
    for y, val in list_of_interactions:
        node = [val* (-1) ** len(y)]
        all_on_sum += node[0]
        for i in y:
            tree[i].append(node)
    return (tree, all_on_sum)

def flip_node(i, tree, all_on_sum):
    for coef in tree[i]:
        coef[0] *= -1
        all_on_sum += 2 * coef[0]
    return tree, all_on_sum

def compute_best_subtraction(transform, method, num_to_subtract=10):
    n = 0
    for elem in transform.keys():
        n = len(elem)
        break
    masks = [[1] * n]
    subtracted = []
    if n == 0:
        return masks
    list_of_interactions = []
    for interaction, val in transform.items():
        list_of_interactions.append((vec_to_index(interaction), val))
        if val > 1e9:
            raise ValueError("Value too large - consider normalizing the function")
    list_of_interactions.sort(key=lambda x: len(x[0]) + 1e-9 * abs(x[1]), reverse=True)  # DANGEROUS!
    if method == 'greedy': # Brute force
        direction = not (eval_function([1] * n, list_of_interactions) > 0)
        mask = [1] * n
        while num_to_subtract > 0:
            best = None
            best_val = eval_function(mask, list_of_interactions)
            for i in range(n):
                if mask[i] == 1:
                    mask[i] = 0
                    val = eval_function(mask, list_of_interactions)
                    if (val > best_val) != direction:
                        best = i
                        best_val = val
                    mask[i] = 1
            if best is None:
                break
            mask[best] = 0
            num_to_subtract -= 1
            subtracted.append(best)
            masks.append(mask.copy())
    elif method == 'smart-greedy':
        tree, best_val = make_dependancy_tree(list_of_interactions, n)
        direction = (best_val > 0)
        val = best_val
        while num_to_subtract > 0:
            best = None
            for i in range(n):
                if i in subtracted:
                    continue
                tree, val = flip_node(i, tree, val)
                if (val > best_val) != direction:
                    best = i
                    best_val = val
                tree, val = flip_node(i, tree, val)
            if best is None:
                break
            tree, val = flip_node(best, tree, val)
            subtracted.append(best)
            num_to_subtract -= 1
            masks.append([1 if i not in subtracted else 0 for i in range(n)])
    elif method == 'linear':
        full_sum = eval_function([1] * n, list_of_interactions)
        direction = not (full_sum > 0)
        list_of_interactions.sort(key=lambda x: len(x[0]) + 1e-9 * x[1], reverse=direction)  # DANGEROUS!
        num_to_subtract = min(num_to_subtract, len(list_of_interactions) - 1)
        i = 0
        while num_to_subtract > 0:
            interaction = list_of_interactions[i][0]
            if len(interaction) > 1:
                raise ValueError("Linear method can only subtract single interactions")
            if len(interaction) == 0:
                i += 1
                continue
            subtracted.append(interaction[0])
            masks.append([1 if i not in subtracted else 0 for i in range(n)])
            i += 1
            num_to_subtract -= 1
    else:
        raise NotImplementedError()
    return masks, subtracted

def subtraction_test(reconstruction, sampling_function, method, subtract_dist):
    sub_mask, subtracted = compute_best_subtraction(reconstruction, method, subtract_dist)
    f = sampling_function(sub_mask)
    res = abs(f[0] - f) / abs(f[0])
    if len(res) < subtract_dist + 1:
        res = np.pad(res, pad_width=(0, subtract_dist + 1 - len(res)), constant_values=np.nan)
    if len(subtracted) < subtract_dist:
        subtracted = np.pad(subtracted, pad_width=(0, subtract_dist - len(subtracted)), constant_values=-1).astype(int)
    return res, subtracted

def get_num_samples(task,save_dir):
    results_dir = f'{save_dir}/{task}'
    num_samples = 0
    for subdir in os.listdir(results_dir):
        if subdir == 'r2_results.pkl':
            continue
        subdir_path = os.path.join(results_dir, subdir)
        if 'lime_b8_order1.pickle' in os.listdir(subdir_path):
            num_samples += 1
    return num_samples


def process_explicand(results_dir, subdir, explicand, Bs, max_order, t):
    subdir_path = os.path.join(results_dir, subdir)
    # try:
    #     with open(os.path.join(subdir_path, 'reconstruction_dict.pickle'), 'rb') as handle:
    #         reconstruction_dict = pickle.load(handle)
    #     with open(os.path.join(subdir_path, 'r2_results.pickle'), 'rb') as handle:
    #         r2_results = pickle.load(handle)
    # except Exception as e:
    #     print(f'Explicand {explicand["id"]} not cached. Running reconstruction.')
    reconstruction_dict, r2_results = get_and_evaluate_reconstruction(explicand = explicand, Bs = Bs, save_dir = subdir_path, max_order = max_order, t = t)
    
    print(f'Finished reconstruction for explicand {explicand["id"]}')
    return reconstruction_dict, r2_results


def main(task = 'hotpotqa', max_order = 4, Bs = [4,6,8], t = 5, save_dir = 'experiments/results'):
    count = 0
    all_results = []
    reg_methods = [('linear', i) for i in range(1,max_order+1)] + [('lasso', i) for i in range(1,max_order+1)] + [('faith_banzhaf', i) for i in range(1,max_order+1)]
    qsft_methods = [('qsft_hard', 0), ('qsft_soft', 0)]
    shap_methods = [('SV', 1)] +  [('FSII', i) for i in range(1,max_order+1)] + [('STII', i) for i in range(1,max_order+1)]
    lime_methods = [('lime', 1)]
    ordered_methods = reg_methods + qsft_methods + shap_methods  + lime_methods
    results_dir = f'{save_dir}/{task}'
    explicand_list = []
    for subdir in os.listdir(results_dir):
        if subdir == 'r2_results.pkl':
            continue
        subdir_path = os.path.join(results_dir, subdir)
        if 'lime_b8_order1.pickle' in os.listdir(subdir_path):
            explicand = pickle.load(open(os.path.join(subdir_path, 'explicand_information.pickle'), 'rb'))
            explicand_list.append((subdir, explicand))
    

    # Process explicands in parallel
    
    reconstruction_results = []
    r2_results = []
    results_list = Parallel(n_jobs=45)(delayed(process_explicand)(results_dir, subdir, explicand, Bs, max_order, t) for subdir, explicand in explicand_list)
    
   
    for result in results_list:
        reconstruction_results.append(result[0])
        r2_results.append(result[1])

    if task == 'hotpotqa':
        model = HotPotQAModel(device = 'cuda:0')
    else:
        model = QAModel(device = 'cuda:0')
    sampling_function = lambda X: model.inference(X)
    
    subtract_results = {}
    explicand_list = [explicand[1] for explicand in explicand_list]
    for method, order in ordered_methods:
        subtract_results[f'{method}_{order}'] = np.zeros((len(explicand_list),SUBTRACT_DIST+1))
        subtract_results[f'{method}_{order}_subtracted_indices'] = np.zeros((len(explicand_list),SUBTRACT_DIST))
        subtract_results[f'{method}_{order}'][:,:] = np.nan
        #subtract_results['explicands'] = []

    for i,sample in enumerate(reconstruction_results):
        explicand = sample['explicand']
        n = model.set_explicand(explicand)
        reconstruction_explicand = reconstruction_results[i]
        for k in reconstruction_explicand.keys():
            if k == 'explicand':
                continue
            else:
                method, b = k[0], k[1]
                if b != 8:
                    continue
                subtract_list, subtracted = subtraction_test(reconstruction_explicand[k], sampling_function, SUBTRACTION_METHODS[method], SUBTRACT_DIST)
                subtract_results[f'{method}'][i,:] = subtract_list
                subtract_results[f'{method}_subtracted_indices'][i,:] = subtracted

    with open(f'experiments/new_results/{task}_subtract_results.pkl', 'wb') as handle:
        pickle.dump(subtract_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'experiments/new_results/{task}_explicand_list.pkl', 'wb') as handle:
        pickle.dump(explicand_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
     



if __name__ == "__main__":
    
    print("Starting main function")
    profiler = cProfile.Profile()
    profiler.enable()
    numba.set_num_threads(8)
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default='hotpotqa')
    parser.add_argument("--Bs", type=int, nargs='+', default=[4, 6, 8])
    parser.add_argument("--t", type=int, default=5)
    parser.add_argument("--MAX_ORDER", type=int, default=4)
    
    args = parser.parse_args()
    main(task = args.task, max_order = args.MAX_ORDER, Bs = [4,6,8], t = args.t, save_dir = SAVE_DIR)



# def main():
#     TASK = 'cancer'
#     DEVICE = 'cpu'
#     NUM_EXPLAIN = 3
#     METHODS = ['qsft_hard', 'qsft_soft', 'amp_first', 'lasso_first', 'linear_first']
#     MAX_B = 8
#     count_b = MAX_B - 2
#     SUBTRACT_DIST = 10
#     explicands, model = get_model(TASK, NUM_EXPLAIN, DEVICE)
#     results = {
#         "samples": np.zeros((NUM_EXPLAIN, count_b)),
#         "methods": {method: {'time': np.zeros((NUM_EXPLAIN, count_b)), 'test_r2': np.zeros((NUM_EXPLAIN, count_b,
#                                                                                             SUBTRACT_DIST+1))}
#                     for method in METHODS}
#     }
#     np.random.seed(0)
#     for i, explicand in enumerate(explicands):
#         n = model.set_explicand(explicand)
#         sampling_function = lambda X: model.inference(X)
#         unix_time_seconds = str(int(time.time()))
#         if not os.path.exists('samples/'):
#             os.makedirs('samples/')
#         save_dir = 'samples/' + unix_time_seconds

#         # Sample explanation function for choice of max b
#         signal, num_samples = sampling_strategy(sampling_function, MAX_B, n, save_dir)
#         results["samples"][i, :] = num_samples

#         for b in range(3, MAX_B+1):
#             print(f"b = {b}")
#             for method in METHODS:
#                 time_taken, subtract_list, subtracted = run_and_evaluate_method(method, signal, b, sampling_function)
#                 results["methods"][method]["time"][i, b-3] = time_taken
#                 results["methods"][method]["test_r2"][i, b-3, :] = subtract_list
#                 print(f"{method}: {np.round(subtract_list, 3)} in {np.round(time_taken, 3)} seconds, subtracted {subtracted}")
#             print()

#         shutil.rmtree(save_dir)

#     with open(f'{TASK}.pkl', 'wb') as handle:
#         pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

# if __name__ == "__main__":
#     profiler = cProfile.Profile()
#     profiler.enable()
#     numba.set_num_threads(8)
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#     main()
#     profiler.disable()
#     stats = pstats.Stats(profiler).sort_stats('tottime')
#     stats.print_stats(50)


# def run_and_evaluate_method(method, signal, b, sampling_function, t=5):
#     start_time = time.time()
#     if "first" in method:
#         order = 1
#     elif "second" in method:
#         order = 2
#     else:
#         order = None
#     reconstruction = {
#         "linear_first": linear,
#         "linear_second": linear,
#         "lasso_first": lasso,
#         "lasso_second": lasso,
#         "amp_first": amp,
#         "amp_second": amp,
#         "qsft_hard": qsft_hard,
#         "qsft_soft": qsft_soft,
#         }.get(method, NotImplementedError())(signal, b, order=order, t=t)
#     subtraction_method = {
#         "linear_first": 'linear',
#         "linear_second": 'smart-greedy',
#         "lasso_first": 'linear',
#         "lasso_second": 'smart-greedy',
#         "amp_first": 'linear',
#         "amp_second": 'smart-greedy',
#         "qsft_hard": 'smart-greedy',
#         "qsft_soft": 'smart-greedy',
#         }
#     end_time = time.time()
#     subtraction_list, subtracted = subtraction_test(reconstruction, sampling_function, subtraction_method[method])
#     return end_time - start_time, subtraction_list, subtracted

    # results = {
    #     "samples": np.zeros((num_samples, len(Bs))),
    #     "methods": {f'{method}_{order}': {'time': np.zeros((num_samples, len(Bs))),
    #                                       'test_r2': np.zeros((num_samples, len(Bs)))}
    #                 for method, order in ordered_methods}
    # }
    # i = 0
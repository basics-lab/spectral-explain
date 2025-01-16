import numba
import numpy as np

from spectral_explain.models.modelloader import get_model
from spectral_explain.support_recovery import sampling_strategy, support_recovery
from experiment_utils import linear, lasso, qsft_hard, qsft_soft, lime, faith_banzhaf, faith_shapley, Alternative_Sampler
import pickle
import time
import os
from math import prod
import shutil
import cProfile
import pstats

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
        direction = (eval_function([1] * n, list_of_interactions) > 0)
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
    return res, subtracted

def run_and_evaluate_method(method, samples, order, b, sampling_function, subtract_dist, t=5):
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
    if order == 1:
        subtraction_method = 'linear'
    else:
        subtraction_method = 'smart-greedy'
    subtraction_method = "greedy"
    end_time = time.time()
    subtraction_list, subtracted = subtraction_test(reconstruction, sampling_function, subtraction_method, subtract_dist)
    return end_time - start_time, subtraction_list, subtracted

SAMPLER_DICT = {
    "qsft_hard": "qsft",
    "qsft_soft": "qsft",
    "linear": "uniform",
    "lasso": "uniform",
    "lime": "lime",
    "faith_banzhaf": "uniform",
    "faith_shapley": "shapley"
}

def main():
    TASK = 'sentiment_mini'
    DEVICE = 'cuda'
    NUM_EXPLAIN = 10
    METHODS = ['linear', 'lasso', 'lime', 'qsft_hard', 'qsft_soft', 'faith_shapley']
    MAX_B = 8
    ALL_Bs = False
    MAX_ORDER = 4
    SUBTRACT_DIST = 8

    sampler_set = set([SAMPLER_DICT[method] for method in METHODS])

    ordered_methods = []
    for regression in ['linear', 'lasso', 'faith_banzhaf', 'faith_shapley']:
        if regression in METHODS:
            ordered_methods += [(regression, order) for order in range(1, MAX_ORDER + 1)]
            METHODS.remove(regression)
    ordered_methods += [(method, 0) for method in METHODS]

    count_b = MAX_B - 2 if ALL_Bs else 1

    explicands, model = get_model(TASK, NUM_EXPLAIN, DEVICE)

    results = {
        "samples": np.zeros((NUM_EXPLAIN, count_b)),
        "methods": {f'{method}_{order}': {'time': np.zeros((NUM_EXPLAIN, count_b)), 'test_r2': np.zeros((NUM_EXPLAIN, count_b,
                                                                                            SUBTRACT_DIST+1))}
                    for method, order in ordered_methods}
    }
    np.random.seed(0)
    for i, explicand in enumerate(explicands):
        n = model.set_explicand(explicand)
        sampling_function = lambda X: model.inference(X)
        unix_time_seconds = str(int(time.time()))
        if not os.path.exists('samples/'):
            os.makedirs('samples/')
        save_dir = 'samples/' + unix_time_seconds

        # Sample explanation function for choice of max b
        qsft_signal, num_samples = sampling_strategy(sampling_function, MAX_B, n, save_dir)
        results["samples"][i, :] = num_samples if ALL_Bs else num_samples[-1]

        # Draws an equal number of uniform samples
        active_sampler_dict = {"qsft": qsft_signal}
        for sampler in sampler_set:
            if sampler != "qsft":
                active_sampler_dict[sampler] = Alternative_Sampler(sampler, sampling_function, qsft_signal, n)

        for b in range(3 if ALL_Bs else MAX_B, MAX_B + 1):
            print(f"b = {b}")
            j = b - 3 if ALL_Bs else 0
            for method, order in ordered_methods:
                method_str = f'{method}_{order}'
                samples = active_sampler_dict[SAMPLER_DICT[method]]
                if (order >= 2 and n >= 128) or (order >= 3 and n >= 32) or (order >= 4 and n >= 16):
                    results["methods"][method_str]["time"][i, j] = np.nan
                    results["methods"][method_str]["test_r2"][i, j] = np.nan
                else:
                    time_taken, subtract_list, subtracted = run_and_evaluate_method(method, samples, order, b, sampling_function, SUBTRACT_DIST)
                    results["methods"][method_str]["time"][i, j] = time_taken
                    results["methods"][method_str]["test_r2"][i, j, :] = subtract_list
                    print(
                        f"{method_str}: {np.round(subtract_list, 3)[1:]} in {np.round(time_taken, 3)} seconds, subtracted {subtracted}")
                    print([explicand['input'][s] for s in subtracted])
            print()
        for s in active_sampler_dict.values():
            del s
        shutil.rmtree(save_dir)
    print('FINAL RESULTS')
    for method, order in ordered_methods:
        method_str = f'{method}_{order}'
        print(method_str)
        print(np.nanmean(results["methods"][method_str]["test_r2"][:,0,:], axis=0))

    with open(f'{TASK}_subtract.pkl', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    numba.set_num_threads(8)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.print_stats(50)

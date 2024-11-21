import numba
import numpy as np
from pyarrow import list_

from spectral_explain.dataloader import get_dataset
from spectral_explain.models.modelloader import get_model
from spectral_explain.support_recovery import sampling_strategy, support_recovery
from spectral_explain.qsft.qsft import fit_regression, transform_via_amp
from spectral_explain.baselines import neural_network
from experiment_utils import linear, lasso, amp, qsft_hard, qsft_soft
from spectral_explain.qsft.utils import qary_ints_low_order
import pickle
import time
import os
from math import prod
from spectral_explain.utils import estimate_r2
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

def subtraction_test(reconstruction, sampling_function, method):
    sub_mask, subtracted = compute_best_subtraction(reconstruction, method)
    f = sampling_function(sub_mask)
    res = abs(f[0] - f) / abs(f[0])
    if len(res) < 11:
        res = np.pad(res, pad_width=(0, 11 - len(res)), constant_values=-1)
    return res, subtracted


def run_and_evaluate_method(method, signal, b, sampling_function, t=5):
    start_time = time.time()
    if "first" in method:
        order = 1
    elif "second" in method:
        order = 2
    else:
        order = None
    reconstruction = {
        "linear_first": linear,
        "linear_second": linear,
        "lasso_first": lasso,
        "lasso_second": lasso,
        "amp_first": amp,
        "amp_second": amp,
        "qsft_hard": qsft_hard,
        "qsft_soft": qsft_soft,
        }.get(method, NotImplementedError())(signal, b, order=order, t=t)
    subtraction_method = {
        "linear_first": 'linear',
        "linear_second": 'smart-greedy',
        "lasso_first": 'linear',
        "lasso_second": 'smart-greedy',
        "amp_first": 'linear',
        "amp_second": 'smart-greedy',
        "qsft_hard": 'smart-greedy',
        "qsft_soft": 'smart-greedy',
        }
    end_time = time.time()
    subtraction_list, subtracted = subtraction_test(reconstruction, sampling_function, subtraction_method[method])
    return end_time - start_time, subtraction_list, subtracted

def main():
    TASK = 'cancer'
    DEVICE = 'cpu'
    NUM_EXPLAIN = 3
    METHODS = ['qsft_hard', 'qsft_soft', 'amp_first', 'lasso_first', 'linear_first']
    MAX_B = 8
    count_b = MAX_B - 2
    SUBTRACT_DIST = 10
    explicands, model = get_model(TASK, NUM_EXPLAIN, DEVICE)
    results = {
        "samples": np.zeros((NUM_EXPLAIN, count_b)),
        "methods": {method: {'time': np.zeros((NUM_EXPLAIN, count_b)), 'test_r2': np.zeros((NUM_EXPLAIN, count_b,
                                                                                            SUBTRACT_DIST+1))}
                    for method in METHODS}
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
        signal, num_samples = sampling_strategy(sampling_function, MAX_B, n, save_dir)
        results["samples"][i, :] = num_samples

        for b in range(3, MAX_B+1):
            print(f"b = {b}")
            for method in METHODS:
                time_taken, subtract_list, subtracted = run_and_evaluate_method(method, signal, b, sampling_function)
                results["methods"][method]["time"][i, b-3] = time_taken
                results["methods"][method]["test_r2"][i, b-3, :] = subtract_list
                print(f"{method}: {np.round(subtract_list, 3)} in {np.round(time_taken, 3)} seconds, subtracted {subtracted}")
            print()

        shutil.rmtree(save_dir)

    with open(f'{TASK}.pkl', 'wb') as handle:
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

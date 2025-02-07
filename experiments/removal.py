from math import prod
import numba
import pickle
import time
import os
import shutil
from spex.modelloader import get_model
from spex.support_recovery import sampling_strategy
from experiment_utils import *


def vec_to_index(vec):
    return np.nonzero(vec)[0]


def eval_function(x, list_of_interactions):
    return sum(v * prod((-1) ** x[i] for i in y) for y, v in list_of_interactions)


def make_dependancy_tree(list_of_interactions, n):
    tree = [[] for _ in range(n)]
    all_on_sum = 0
    for y, val in list_of_interactions:
        node = [val * (-1) ** len(y)]
        all_on_sum += node[0]
        for i in y:
            tree[i].append(node)
    return (tree, all_on_sum)


def flip_node(i, tree, all_on_sum):
    for coef in tree[i]:
        coef[0] *= -1
        all_on_sum += 2 * coef[0]
    return tree, all_on_sum


def compute_best_subtraction(transform, method, sampling_function, num_to_subtract=10, direction=None):
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
    if direction is None:
        direction = sampling_function([[1] * n]) > 0
    if method == 'greedy':  # Brute force
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
        list_of_interactions.sort(key=lambda x: len(x[0]) + 1e-9 * x[1], reverse=not direction)  # DANGEROUS!
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


def subtraction_test(reconstruction, sampling_function, method, subtract_dist, direction=None):
    sub_mask, subtracted = compute_best_subtraction(reconstruction, method, sampling_function, subtract_dist, direction)
    f = sampling_function(sub_mask)
    res = abs(f[0] - f) / abs(f[0])
    if len(res) < subtract_dist + 1:
        res = np.pad(res, pad_width=(0, subtract_dist + 1 - len(res)), constant_values=np.nan)
    if len(subtracted) < subtract_dist:
        subtracted = np.pad(subtracted, pad_width=(0, subtract_dist - len(subtracted)), constant_values=-1).astype(int)
    return res, subtracted


def run_and_evaluate_method(method, sampler, order, b, sampling_function, subtract_dist, direction):
    reconstruction = {
        "shapley": shapley,
        "banzhaf": banzhaf,
        "lime": LIME,
        "faith_shapley": faith_shapley,
        "faith_banzhaf": faith_banzhaf,
        "shapley_taylor": shapley_taylor,
        "spex_hard": spex_hard,
        "spex_soft": spex_soft,
    }.get(method, NotImplementedError())(sampler, b, order=order)
    if method in ["shapley", "banzhaf", "lime"]:
        subtraction_method = "linear"
    else:
        subtraction_method = "greedy"
    differences, subtracted_locs = subtraction_test(reconstruction, sampling_function,
                                                    subtraction_method, subtract_dist, direction)
    return differences, subtracted_locs


def removal(explicands, model, methods, max_b, all_bs, max_order, subtract_dist):
    sampler_set = set([SAMPLER_DICT[method] for method in methods])

    ordered_methods = get_ordered_methods(methods, max_order)

    count_b = max_b - 2 if all_bs else 1

    results = {
        "samples": np.zeros((len(explicands), count_b)),
        "methods": {f'{method}_{order}': {'positive_locs': np.zeros((len(explicands), count_b, subtract_dist)),
                                          'negative_locs': np.zeros((len(explicands), count_b, subtract_dist)),
                                          'positive_diffs': np.zeros(
                                              (len(explicands), count_b, subtract_dist + 1)),
                                          'negative_diffs': np.zeros(
                                              (len(explicands), count_b, subtract_dist + 1))}
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
        spex_signal, num_samples = sampling_strategy(sampling_function, max_b, n, save_dir)
        results["samples"][i, :] = num_samples if all_bs else num_samples[-1]

        # Draws an equal number of uniform samples
        active_sampler_dict = {"spex": spex_signal}
        for sampler in sampler_set:
            if sampler != "spex":
                active_sampler_dict[sampler] = AlternativeSampler(sampler, sampling_function, spex_signal, n)

        for b in range(3 if all_bs else max_b, max_b + 1):
            print(f"b = {b}")
            j = b - 3 if all_bs else 0
            for method, order in ordered_methods:
                method_str = f'{method}_{order}'
                print(method_str)
                samples = active_sampler_dict[SAMPLER_DICT[method]]
                if (order >= 2 and n >= 64) or (order >= 3 and n >= 32) or (order >= 4 and n >= 16):
                    results["methods"][method_str]["positive_locs"][i, j, :] = np.nan
                    results["methods"][method_str]["negative_locs"][i, j, :] = np.nan
                    results["methods"][method_str]["positive_diffs"][i, j, :] = np.nan
                    results["methods"][method_str]["negative_diffs"][i, j, :] = np.nan
                else:
                    pos_diffs, pos_subtracted_locs = run_and_evaluate_method(method, samples, order, b,
                                                                            sampling_function,
                                                                            subtract_dist, direction=True)

                    neg_diffs, neg_subtracted_locs = run_and_evaluate_method(method, samples, order, b,
                                                                            sampling_function,
                                                                            subtract_dist, direction=False)

                    results["methods"][method_str]["positive_locs"][i, j, :] = pos_subtracted_locs
                    results["methods"][method_str]["negative_locs"][i, j, :] = neg_subtracted_locs
                    results["methods"][method_str]["positive_diffs"][i, j, :] = pos_diffs
                    results["methods"][method_str]["negative_diffs"][i, j, :] = neg_diffs
            print()
        for s in active_sampler_dict.values():
            del s
        shutil.rmtree(save_dir)

    return results


if __name__ == "__main__":
    numba.set_num_threads(8)
    TASK = 'cancer'  # choose TASK from parkinsons, cancer, sentiment, puzzles, drop, hotpotqa, vision
    DEVICE = 'cpu'  # choose DEVICE from cpu, mps, or cuda
    NUM_EXPLAIN = 500  # the number of examples from TASK to be explained
    MAX_ORDER = 2  # the max order of baseline interaction methods
    MAX_B = 8  # the maximum sparsity parameter, which scales samples taken ~15 * 2^B * log(number of features)
    ALL_Bs = False  # uses different sparsity parameters b from 3 to MAX_B
    SUBTRACT_DIST = 10  # the maximum number of features to subtract

    # marginal attribution methods: shapley, banzhaf, lime
    # interaction attribution methods: faith_banzhaf, faith_shapley, shapley_taylor
    # spex attribution methods: spex_hard (faster decoding), spex_soft (slower decoding for better performance)
    METHODS = ['shapley', 'banzhaf', 'lime',
               'faith_banzhaf', 'faith_shapley', 'shapley_taylor',
               'spex_hard', 'spex_soft']

    if DEVICE == 'cuda':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    explicands, model = get_model(TASK, NUM_EXPLAIN, DEVICE)

    results = removal(explicands, model, METHODS, MAX_B, ALL_Bs, MAX_ORDER, SUBTRACT_DIST)

    if not os.path.exists('results/'):
        os.makedirs('results/')
    with open(f'results/{TASK}_removal.pkl', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

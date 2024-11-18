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

def compute_best_subtraction(transform, method):

    n = 0
    for elem in transform.keys():
        n = len(elem)
        break
    if n == 0:
        return 0
    num_to_subtract = 10
    if method == 'greedy': # Brute force
        list_of_interactions = []
        for interaction, val in transform.items():
            list_of_interactions.append((vec_to_index(interaction), val))
        list_of_interactions.sort(key=lambda x: len(x[0]) + 1e-9*abs(x[1]), reverse=True) # DANGEROUS!
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
        breakpoint()


def subtraction_test(reconstruction, saved_samples_test):
    sub = compute_best_subtraction(reconstruction, 'greedy')
    query_indices_test, samples_test = saved_samples_test
    return np.mean(np.abs(reconstruction - samples_test))


def run_and_evaluate_method(method, signal, b, saved_samples_test, t=5):
    start_time = time.time()
    if "first" in method:
        order = 1
    elif "second" in method:
        order = 2
    else:
        order = None
    if method == 'neural_network':
        nn = neural_network.NeuralNetwork(signal, b)
        end_time = time.time()
        return end_time - start_time, nn.evaluate(saved_samples_test)
    else:
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
        end_time = time.time()
        return end_time - start_time, subtraction_test(reconstruction, saved_samples_test)

def main():
    # choose TASK from parkinsons, cancer, sentiment,
    # sentiment_mini, similarity, similarity_mini,
    # comprehension, comprehension_mini, clinical
    TASK = 'parkinsons'
    DEVICE = 'cpu'
    NUM_EXPLAIN = 3

    METHODS = ['qsft_hard', 'qsft_soft', 'amp_first', 'lasso_first', 'linear_first', 'neural_network']

    MAX_B = 8
    count_b = MAX_B - 2

    explicands, model = get_model(TASK, NUM_EXPLAIN, DEVICE)

    results = {
        "samples": np.zeros((NUM_EXPLAIN, count_b)),
        "methods": {method: {'time': np.zeros((NUM_EXPLAIN, count_b)), 'test_r2': np.zeros((NUM_EXPLAIN, count_b))}
                    for method in METHODS}
    }

    np.random.seed(0)
    for i, explicand in enumerate(explicands):
        n = model.set_explicand(explicand)
        sampling_function = lambda X: model.inference(X)

        query_indices_test = np.random.choice(2, size=(10000, n))
        saved_samples_test = query_indices_test, sampling_function(query_indices_test)

        unix_time_seconds = str(int(time.time()))
        if not os.path.exists('samples/'):
            os.makedirs('samples/')
        save_dir = 'samples/' + unix_time_seconds

        # Sample explanation function for choice of max b
        signal, num_samples = sampling_strategy(sampling_function, MAX_B, n, save_dir)
        results["samples"][i, :] = num_samples

        for b in range(6, MAX_B+1):
            print(f"b = {b}")
            for method in METHODS:
                time_taken, test_r2 = run_and_evaluate_method(method, signal, b, saved_samples_test)
                results["methods"][method]["time"][i, b-3] = time_taken
                results["methods"][method]["test_r2"][i, b-3] = test_r2
                print(f"{method}: {np.round(test_r2, 3)} test r2 in {np.round(time_taken, 3)} seconds")
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

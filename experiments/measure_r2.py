import numba
import numpy as np
import pickle
import time
import os
import shutil
import cProfile
import pstats
from spectral_explain.models.modelloader import get_model
from spectral_explain.support_recovery import sampling_strategy
from spectral_explain.baselines import neural_network
from spectral_explain.utils import estimate_r2
from experiment_utils import linear, lasso, amp, qsft_hard, qsft_soft

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
        return end_time - start_time, estimate_r2(reconstruction, saved_samples_test)


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

        for b in range(3, MAX_B+1):
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

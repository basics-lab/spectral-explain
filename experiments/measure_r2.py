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
from spectral_explain.utils import estimate_r2, estimate_mobius_r2
from experiment_utils import linear, lasso, amp, qsft_hard, qsft_soft, banzhaf

def run_and_evaluate_method(method, signal_qsft, signal_uniform, order, b, saved_samples_test, t=5):
    start_time = time.time()
    reconstruction = {
        "linear": linear,
        "lasso": lasso,
        "qsft_hard": qsft_hard,
        "qsft_soft": qsft_soft,
    }.get(method, NotImplementedError())(signal_qsft if "qsft" in method else signal_uniform, b, order=order, t=t)
    end_time = time.time()
    return end_time - start_time, estimate_r2(reconstruction, saved_samples_test)

class Uniform_Sampler:
    def __init__(self, sampling_function, qsft_signal, n):
        self.n = n
        self.all_queries = []
        self.all_samples = []
        for m in range(len(qsft_signal.all_samples)):
            queries_subsample = []
            samples_subsample = []
            for d in range(len(qsft_signal.all_samples[0])):
                queries = np.random.choice(2, size=qsft_signal.all_queries[m][d].shape)
                queries_subsample.append(queries)
                samples_subsample.append(sampling_function(queries))
            self.all_queries.append(queries_subsample)
            self.all_samples.append(samples_subsample)


def main():
    # choose TASK from parkinsons, cancer, sentiment,
    # sentiment_mini, similarity, similarity_mini,
    # comprehension, comprehension_mini, clinical
    TASK = 'sentiment'
    DEVICE = 'cuda'
    NUM_EXPLAIN = 500
    MAX_ORDER = 3
    MAX_B = 9
    METHODS = ['linear', 'lasso', 'qsft_hard', 'qsft_soft']

    ordered_methods = []
    for regression in ['linear', 'lasso']:
        if regression in METHODS:
            ordered_methods += [(regression, order) for order in range(1, MAX_ORDER + 1)]
            METHODS.remove(regression)
    ordered_methods += [(method, 0) for method in METHODS]

    count_b = MAX_B - 2

    explicands, model = get_model(TASK, NUM_EXPLAIN, DEVICE)

    results = {
        "samples": np.zeros((len(explicands), count_b)),
        "methods": {f'{method}_{order}': {'time': np.zeros((len(explicands), count_b)), 'test_r2': np.zeros((len(explicands), count_b))}
                    for method, order in ordered_methods}
    }

    np.random.seed(0)
    for i, explicand in enumerate(explicands):
        print(explicand)
        n = model.set_explicand(explicand)
        sampling_function = lambda X: model.inference(X)

        query_indices_test = np.random.choice(2, size=(10000, n))
        saved_samples_test = query_indices_test, sampling_function(query_indices_test)

        unix_time_seconds = str(int(time.time()))
        if not os.path.exists('samples/'):
            os.makedirs('samples/')
        save_dir = 'samples/' + unix_time_seconds

        # Sample explanation function for choice of max b
        qsft_signal, num_samples = sampling_strategy(sampling_function, MAX_B, n, save_dir)
        results["samples"][i, :] = num_samples

        # Draws an equal number of uniform samples
        uniform_signal = Uniform_Sampler(sampling_function, qsft_signal, n)

        for b in range(3, MAX_B+1):
            print(f"b = {b}")
            for method, order in ordered_methods:
                method_str = f'{method}_{order}'
                if (order >= 2 and n >= 128) or (order >= 3 and n >= 32):
                    results["methods"][method_str]["time"][i, b - 3] = np.nan
                    results["methods"][method_str]["test_r2"][i, b - 3] = np.nan
                else:
                    time_taken, test_r2 = run_and_evaluate_method(method, qsft_signal, uniform_signal, order, b, saved_samples_test)
                    results["methods"][method_str]["time"][i, b-3] = time_taken
                    results["methods"][method_str]["test_r2"][i, b-3] = test_r2
                    print(f"{method_str}: {np.round(test_r2, 3)} test r2 in {np.round(time_taken, 3)} seconds")
            print()
        del uniform_signal
        shutil.rmtree(save_dir)

    with open(f'{TASK}_{unix_time_seconds}.pkl', 'wb') as handle:
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

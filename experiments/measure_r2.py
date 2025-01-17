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
from spectral_explain.utils import estimate_r2
from experiment_utils import *



def run_and_evaluate_method(method, samples, order, b, saved_samples_test, t=5):
    start_time = time.time()
    reconstruction = {
        "linear": linear,
        "lasso": lasso,
        "lime": LIME,
        "qsft_hard": qsft_hard,
        "qsft_soft": qsft_soft,
        "faith_banzhaf": faith_banzhaf,
        "faith_shapley": faith_shapley,
        "shapley": shapley,
        "banzhaf": banzhaf
    }.get(method, NotImplementedError())(samples, b, order=order, t=t)
    end_time = time.time()
    return end_time - start_time, estimate_r2(reconstruction, saved_samples_test)


def main():
    # choose TASK from parkinsons, cancer, sentiment,
    # sentiment_mini, similarity, similarity_mini,
    # comprehension, comprehension_mini, clinical
    TASK = 'sentiment'
    DEVICE = 'cuda'
    NUM_EXPLAIN = 500
    MAX_ORDER = 4
    MAX_B = 8
    ALL_Bs = False
    METHODS = ['shapley', 'banzhaf', 'linear', 'lasso', 'lime', 'qsft_hard', 'qsft_soft', 'faith_shapley']

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
        "samples": np.zeros((len(explicands), count_b)),
        "methods": {f'{method}_{order}': {'time': np.zeros((len(explicands), count_b)),
                                          'test_r2': np.zeros((len(explicands), count_b))}
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
        sampling_time_start = time.time()
        qsft_signal, num_samples = sampling_strategy(sampling_function, MAX_B, n, save_dir)
        sampling_time = time.time() - sampling_time_start
        results["samples"][i, :] = num_samples if ALL_Bs else num_samples[-1]

        # Draws an equal number of uniform samples
        active_sampler_dict = {"qsft": qsft_signal}
        for sampler in sampler_set:
            if sampler != "qsft":
                active_sampler_dict[sampler] = AlternativeSampler(sampler, sampling_function, qsft_signal, n)

        for b in range(3 if ALL_Bs else MAX_B, MAX_B + 1):
            print(f"b = {b}")
            j = b - 3 if ALL_Bs else 0
            for method, order in ordered_methods:
                method_str = f'{method}_{order}'
                samples = active_sampler_dict[SAMPLER_DICT[method]]
                if (order >= 2 and n >= 64) or (order >= 3 and n >= 32) or (order >= 4 and n >= 16):
                    results["methods"][method_str]["time"][i, j] = np.nan
                    results["methods"][method_str]["test_r2"][i, j] = np.nan
                else:
                    time_taken, test_r2 = run_and_evaluate_method(method, samples, order, b, saved_samples_test)
                    if method in ["lime", "shapley", "faith_shapley"]:
                        # SHAP-IQ / LIME do not specify sampling vs compute time, we estimate using our sampling time
                        time_taken -= sampling_time
                    results["methods"][method_str]["time"][i, j] = np.max(time_taken, 0)
                    results["methods"][method_str]["test_r2"][i, j] = test_r2
                    print(f"{method_str}: {np.round(test_r2, 3)} test r2 in {np.round(np.max(time_taken, 0), 3)} seconds")
            print()
        for s in active_sampler_dict.values():
            del s
        shutil.rmtree(save_dir)

    with open(f'{TASK}_faithfulness_{unix_time_seconds}.pkl', 'wb') as handle:
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

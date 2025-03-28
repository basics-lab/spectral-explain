import numba
import pickle
import time
import os
import shutil
from modelloader import get_model
from spex.support_recovery import sampling_strategy
from spex.utils import *
from experiment_utils import *


def run_and_evaluate_method(method, sampler, order, b, saved_samples_test):
    """
    Run and evaluate the specified method.

    Parameters:
    - method: The name of the method to run.
    - sampler: The sampler object that stores samples.
    - order: The order of the method.
    - b: The sparsity parameter.
    - saved_samples_test: The test samples to evaluate the method.

    Returns:
    - The time taken to run the method.
    - The test R2 of the reconstruction.
    - The Fourier reconstruction dictionary.
    """
    start_time = time.time()
    reconstruction = {
        "shapley": shapley,
        "banzhaf": banzhaf,
        "lime": LIME,
        "faith_shapley": faith_shapley,
        "faith_banzhaf": faith_banzhaf,
        "shapley_taylor": shapley_taylor,
        "spex_hard": spex_hard,
        "spex_soft": spex_soft,
        "proxy_spex": proxy_spex
    }.get(method, NotImplementedError())(sampler, b, order=order)
    end_time = time.time()
    return end_time - start_time, estimate_r2(reconstruction, saved_samples_test), reconstruction


def faithfulness(explicands, model, methods, bs, max_order, num_test_samples, sparsity_reconstruction):
    """
    Evaluate the faithfulness of different explanation methods.

    Parameters:
    - explicands: The explicands to explain.
    - model: The model to use for inference.
    - methods: The list of attribution methods to evaluate.
    - bs: The list of sparsity parameters to use.
    - max_order: The maximum order of baseline interaction methods.
    - num_test_samples: The number of test samples to use for evaluation.
    - sparsity_reconstruction: Whether or not to compute the reconstruction error by sparsity.

    Returns:
    - results. A python dictionary with removal results.
    """
    sampler_set = set([SAMPLER_DICT[method] for method in methods])

    ordered_methods = get_ordered_methods(methods, max_order)

    count_b = len(bs)

    results = {
        "samples": np.zeros((len(explicands), count_b)),
        "methods": {f'{method}_{order}': {'time': np.zeros((len(explicands), count_b)),
                                          'test_r2': np.zeros((len(explicands), count_b)),
                                          'reconstructions': [[None] * count_b for _ in range(len(explicands))],
                                          'sampler': None,
                                          'sparsity_reconstruction': [[None] * count_b for _ in range(len(explicands))]}
                    for method, order in ordered_methods}
    }

    np.random.seed(0)
    for i, explicand in enumerate(explicands):
        print(explicand)
        n = model.set_explicand(explicand)
        sampling_function = model.inference

        query_indices_test = np.random.choice(2, size=(num_test_samples, n))
        saved_samples_test = query_indices_test, sampling_function(query_indices_test)

        unix_time_seconds = str(int(time.time()))
        if not os.path.exists('samples/'):
            os.makedirs('samples/')
        save_dir = 'samples/' + unix_time_seconds

        # Sample explanation function for choice of max b
        sampling_time_start = time.time()
        spex_signal, num_samples = sampling_strategy(sampling_function, min(bs), max(bs), n, save_dir)
        sampling_time = time.time() - sampling_time_start
        results["samples"][i, :] = [num_samples[b] for b in bs]

        # Draws an equal number of uniform samples
        active_sampler_dict = {"spex": spex_signal}
        for sampler in sampler_set:
            if sampler != "spex":
                active_sampler_dict[sampler] = AlternativeSampler(sampler, sampling_function, spex_signal, n)

        for j, b in enumerate(bs):
            print(f"b = {b}")
            for method, order in ordered_methods:
                method_str = f'{method}_{order}'
                sampler = active_sampler_dict[SAMPLER_DICT[method]]
                results["methods"][method_str]["sampler"] = (sampler.all_queries, sampler.all_samples)
                if "spex" not in method and (
                        (order >= 2 and n >= 64) or (order >= 3 and n >= 32) or (order >= 4 and n >= 16)):
                    results["methods"][method_str]["time"][i, j] = np.nan
                    results["methods"][method_str]["test_r2"][i, j] = np.nan
                else:
                    time_taken, test_r2, recon = run_and_evaluate_method(method, sampler, order, b, saved_samples_test)
                    if method in ["lime", "shapley", "faith_shapley", "shapley_taylor"]:
                        # SHAP-IQ / LIME do not specify sampling vs compute time,
                        # we approximate using SPEX sampling time
                        time_taken -= sampling_time
                    results["methods"][method_str]["time"][i, j] = max(time_taken, 0)
                    results["methods"][method_str]["test_r2"][i, j] = test_r2
                    results["methods"][method_str]["reconstructions"][i][j] = recon
                    print(f"{method_str}: {np.round(test_r2, 3)} test r2 in {np.round(max(time_taken, 0), 3)} seconds")
                    if sparsity_reconstruction:
                        sorted_reconstruction = [(k, v) for k, v in
                                                 sorted(recon.items(), key=lambda item: -np.abs(item[1]))]
                        sparse_recons = np.zeros(len(sorted_reconstruction))
                        for sparsity in range(1, len(sorted_reconstruction) + 1):
                            sparse_recons[sparsity - 1] = estimate_r2({k: v for k, v in sorted_reconstruction[:sparsity]},
                                                               saved_samples_test)
                        results["methods"][method_str]["sparsity_reconstruction"][i][j] = sparse_recons
            print()
        for s in active_sampler_dict.values():
            del s
        shutil.rmtree(save_dir)

    return results


if __name__ == "__main__":
    setup_root()
    numba.set_num_threads(8)
    TASK = 'sentiment'  # choose TASK from parkinsons, cancer, sentiment, puzzles, drop, hotpotqa, visual-qa
    DEVICE = 'cuda'  # choose DEVICE from cpu, mps, or cuda
    NUM_EXPLAIN = 1000  # the number of examples from TASK to be explained
    MAX_ORDER = 1  # the max order of baseline interaction methods
    Bs = [4]  # (list) range of sparsity parameters B, samples ~15 * 2^B * log(number of features), rec. B = 8
    NUM_TEST_SAMPLES = 1000  # number of uniformly drawn test samples to measure faithfulness
    SPARSITY_RECONSTRUCTION = True  # whether to compute reconstruction by sparsity

    # marginal attribution methods: shapley, banzhaf, lime
    # interaction attribution methods: faith_banzhaf, faith_shapley, shapley_taylor
    # spex attribution methods: spex_hard (faster decoding), spex_soft (slower decoding for better performance)
    METHODS = ['shapley', 'banzhaf', 'lime',
               'faith_banzhaf', 'faith_shapley', 'shapley_taylor',
               'spex_hard', 'spex_soft', 'proxy_spex']

    if DEVICE == 'cuda':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    explicands, model = get_model(TASK, NUM_EXPLAIN, DEVICE)

    results = faithfulness(explicands, model, METHODS, Bs, MAX_ORDER, NUM_TEST_SAMPLES, SPARSITY_RECONSTRUCTION)

    if not os.path.exists('results/'):
        os.makedirs('results/')
    with open(f'results/{TASK}_faithfulness.pkl', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

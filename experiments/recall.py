import numba
import numpy as np
import pickle
import time
import os
import shutil
import cProfile
import pstats
from spex.models.modelloader import get_model
from spex.utils import estimate_r2, fourier_to_mobius
from spex.qsft.qsft import fit_regression
from spex.qsft.utils import qary_ints_low_order, qary_ints, gwht
from experiment_utils import qsft_soft, Alternative_Sampler, linear
from spex.support_recovery import sampling_strategy
import itertools
import matplotlib.pyplot as plt
from scipy import stats

def measure_mse(true_fbii, predicted_fbii):
    mse = 0
    for key in true_fbii.keys():
        if key in predicted_fbii:
            mse += (true_fbii[key] - predicted_fbii[key]) ** 2
        else:
            mse += (true_fbii[key]) ** 2
    return mse / len(true_fbii)

def measure_ranking_error(true_fbii, predicted_fbii, k=10):
    fbii_list = []
    fbii_pred_list = []
    for key in true_fbii.keys():
        fbii_list.append(true_fbii[key])
        if key in predicted_fbii:
            fbii_pred_list.append(predicted_fbii[key])
        else:
            fbii_pred_list.append(0)

    # get top k indices of true fbii
    top_k_indices = np.argsort(np.abs(fbii_list))[-k:]
    top_k_indices_pred = np.argsort(np.abs(fbii_pred_list))[-k:]
    print(top_k_indices)
    print(top_k_indices_pred)
    return len(set(top_k_indices).intersection(set(top_k_indices_pred))) / k



def main():
    # choose TASK from parkinsons, cancer, sentiment,
    # sentiment_mini, similarity, similarity_mini,
    # comprehension, comprehension_mini, clinical
    TASK = 'sentiment_mini'
    DEVICE = 'cuda'
    NUM_EXPLAIN = 100
    MAX_B = 8
    MAX_ORDER = 5
    Ks = [10,20,50,100]
    count_b = MAX_B - 2

    explicands, model = get_model(TASK, NUM_EXPLAIN, DEVICE)
    for e in explicands:
        assert len(e['input']) < 24, "script requires taking all 2^n masking patterns"

    np.random.seed(0)

    results = {
        "samples": np.zeros((len(explicands), count_b)),
        "methods": {f'{method}': {'recall_k': np.zeros((len(explicands), count_b,  len(Ks))),
                                  'errors': np.zeros((len(explicands), count_b))}
                    for method in ["SpectralExplain", "Regression"]}
    }

    for i, explicand in enumerate(explicands):
        n = model.set_explicand(explicand)
        sampling_function = lambda X: model.inference(X)

        # Compute ground truth MAX_ORDER Faith-Banzhaf Interaction Indices
        all_queries = np.array(list(itertools.product([0, 1], repeat=n)))
        all_values = sampling_function(all_queries)
        all_fourier = gwht(all_values, 2, n)
        fourier_low_order = {}
        for coef in range(all_queries.shape[0]):
            if np.sum(all_queries[coef]) <= MAX_ORDER:
                fourier_low_order[tuple(all_queries[coef, :].astype(int))] = all_fourier[coef]
        fbii_gt = fourier_to_mobius(fourier_low_order)

        # Compute SpectralExplain predicted MAX_ORDER Faith-Banzhaf Interaction Indices
        unix_time_seconds = str(int(time.time()))
        if not os.path.exists('samples/'):
            os.makedirs('samples/')
        save_dir = 'samples/' + unix_time_seconds
        qsft_signal, num_samples = sampling_strategy(sampling_function, MAX_B, n, save_dir, t=MAX_ORDER)
        print(num_samples)
        uniform_sampler = Alternative_Sampler("uniform", sampling_function, qsft_signal, n)
        for j, b in enumerate(range(3, MAX_B + 1)):
            qsft_result = qsft_soft(qsft_signal, b, t=MAX_ORDER)
            fbii_qsft = fourier_to_mobius(qsft_result)

            regression_result = linear(uniform_sampler, b, MAX_ORDER)
            fbii_regression = fourier_to_mobius(regression_result)

            results["methods"]["SpectralExplain"]["errors"][i, j] = measure_mse(fbii_gt, fbii_qsft)
            results["methods"]["Regression"]["errors"][i, j] = measure_mse(fbii_gt, fbii_regression)
            for k in range(len(Ks)):
                results["methods"]["SpectralExplain"]["recall_k"][i, j, k] = measure_ranking_error(fbii_gt, fbii_qsft, Ks[k])
                results["methods"]["Regression"]["recall_k"][i, j, k] = measure_ranking_error(fbii_gt, fbii_regression, Ks[k])

            print(results["methods"]["SpectralExplain"]["recall_k"][i, j, :])
            print(results["methods"]["Regression"]["recall_k"][i, j, :])
            print()


    # Save to pickle file
    with open(f'{TASK}_interactions_{MAX_ORDER}.pkl', 'wb') as handle:
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
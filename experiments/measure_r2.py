import numpy as np
from data.dataloader import get_dataset
from models.modelloader import get_model
import itertools
from support_recovery import sampling_strategy, support_recovery
from qsft.qsft import fit_regression, transform_via_amp
from qsft.utils import qary_ints_low_order
import pickle
import time
import os
from utils import estimate_r2

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def linear(signal, b, order):
    assert order in [1, 2]
    if order == 1:
        return fit_regression('linear', {'locations': qary_ints_low_order(signal.n, 2, 1).T}, signal, signal.n, b)[0]
    else:
        return fit_regression('linear', {'locations': qary_ints_low_order(signal.n, 2, 2).T}, signal, signal.n, b)[0]


def lasso(signal, b, order):
    assert order in [1, 2]

    if order == 1:
        return fit_regression('lasso', {'locations': qary_ints_low_order(signal.n, 2, 1).T}, signal, signal.n, b)[0]
    else:
        return fit_regression('lasso', {'locations': qary_ints_low_order(signal.n, 2, 2).T}, signal, signal.n, b)[0]


def amp(signal, b, order):
    assert order in [1,2]
    return transform_via_amp(signal, b, order=order)["transform"]


def qsft_hard(signal, b, order):
    return support_recovery("hard", signal, b)["transform"]


def qsft_soft(signal, b, order):
    return support_recovery("soft", signal, b)["transform"]


def run_and_evaluate_method(method, signal, b, saved_samples_test):
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
    }.get(method, NotImplementedError())(signal, b, order)
    end_time = time.time()
    return end_time - start_time, estimate_r2(reconstruction, saved_samples_test)


if __name__ == "__main__":
    # choose TASK from parkings, cancer, sentiment,
    # sentiment_mini, similarity, similarity_mini,
    # comprehension, comprehension_mini, clinical
    TASK = 'cancer'
    DEVICE = 'cpu'
    NUM_EXPLAIN = 100

    METHODS = ['linear_first', 'linear_second', 'lasso_first', 'lasso_second', 'amp_first', 'amp_second', 'qsft_hard', 'qsft_soft']
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
        signal = sampling_strategy(sampling_function, MAX_B, n, save_dir)

        for b in range(3, MAX_B+1):
            print(f"b = {b}")
            for method in METHODS:
                time_taken, test_r2 = run_and_evaluate_method(method, signal, b, saved_samples_test)
                results["methods"][method]["time"] = time_taken
                results["methods"][method]["test_r2"] = test_r2
                print(f"{method}: {np.round(test_r2, 3)} test r2 in {np.round(time_taken, 3)} seconds")
            print()

    with open(f'{TASK}.pkl', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

import numpy as np
from spectral_explain.dataloader import get_dataset
from spectral_explain.models.modelloader import get_model
from spectral_explain.support_recovery import sampling_strategy, support_recovery
from spectral_explain.qsft.qsft import fit_regression, transform_via_amp
from spectral_explain.baselines import neural_network
from spectral_explain.qsft.utils import qary_ints_low_order
import pickle
import time
import os
from spectral_explain.utils import estimate_r2
import shutil

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
        }.get(method, NotImplementedError())(signal, b, order)
        end_time = time.time()
        return end_time - start_time, estimate_r2(reconstruction, saved_samples_test)


if __name__ == "__main__":
    # choose TASK from parkinsons, cancer, sentiment,
    # sentiment_mini, similarity, similarity_mini,
    # comprehension, comprehension_mini, clinical
    TASK = 'parkinsons'
    DEVICE = 'cpu'
    NUM_EXPLAIN = 3

    METHODS = ['linear_first', 'lasso_first',
               'amp_first', 'neural_network', 'qsft_hard', 'qsft_soft']
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

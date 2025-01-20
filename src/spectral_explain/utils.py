import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from itertools import chain, combinations
from sklearn.metrics import ndcg_score
import math
from tqdm import tqdm

def estimate_r2(recon_function, saved_samples):
    query_indices, y_true = saved_samples

    if len(recon_function) == 0:
        y_hat = np.zeros(y_true.shape)
    else:
        beta_keys = list(recon_function.keys())
        beta_values = list(recon_function.values())
        freqs = np.array(query_indices) @ np.array(beta_keys).T
        H = np.exp(2j * np.pi * freqs / 2)
        y_hat = np.real(H @ np.array(beta_values))
    return 1 - (np.linalg.norm(y_true - y_hat) ** 2 / np.linalg.norm(y_true - np.mean(y_true)) ** 2)

def fourier_to_shapley(qsft_results, n):
    if len(qsft_results) == 0:
        return np.zeros(n)
    else:
        fourier_svs = np.zeros(n)
        for loc, coef in qsft_results.items():
            hw = np.sum(loc)
            if hw % 2 == 1:
                fourier_svs += -2 * np.real(coef) * np.array(loc) / hw
        return fourier_svs

def powerset(loc_tuple, max_order=None):
    nonzero_locs = [i for i, val in enumerate(loc_tuple) if val == 1]

    if max_order is None:
        max_order = len(nonzero_locs)
    nonzero_locs_powerset = chain.from_iterable(combinations(nonzero_locs, r) for r in range(max_order + 1))
    tuples = []
    for nzl in nonzero_locs_powerset:
        entry = np.zeros(len(loc_tuple)).astype(int)
        entry[list(nzl)] = 1
        tuples.append(tuple(entry))
    return tuples

def fourier_to_mobius(qsft_results):
    if len(qsft_results) == 0:
        return {}
    else:
        unscaled_mobius_dict = {}
        for loc, coef in qsft_results.items():
            real_coef = np.real(coef)
            for subset in powerset(loc):
                if subset in unscaled_mobius_dict:
                    unscaled_mobius_dict[subset] += real_coef
                else:
                    unscaled_mobius_dict[subset] = real_coef

        # multiply each entry by (-2)^(cardinality)
        return {loc: val * np.power(-2.0, np.sum(loc)) for loc, val in unscaled_mobius_dict.items() if np.abs(val) > 1e-12}

def mobius_to_fourier(qsft_results):
    if len(qsft_results) == 0:
        return {}
    else:
        unscaled_fourier_dict = {}
        for loc, coef in qsft_results.items():
            real_coef = np.real(coef) / (2 ** sum(loc))
            for subset in powerset(loc):
                if subset in unscaled_fourier_dict:
                    unscaled_fourier_dict[subset] += real_coef
                else:
                    unscaled_fourier_dict[subset] = real_coef

        # multiply each entry by (-1)^(cardinality)
        return {loc: val * np.power(-1.0, np.sum(loc)) for loc, val in unscaled_fourier_dict.items() if np.abs(val) > 1e-12}

def fourier_to_banzhaf(qsft_results, n):
    if len(qsft_results) == 0:
        return np.zeros(n)
    else:
        # this is simply -2 * the first order coefficients
        fourier_bvs = np.zeros(n)
        for loc, coef in qsft_results.items():
            if np.sum(loc) == 1:
                fourier_bvs += np.array(loc) * np.real(coef)
        return -2 * fourier_bvs

def mobius_to_shapley_ii(mobius_dict):
    # Equation (7) of https://ikojadin.perso.univ-pau.fr/kappalab/pub/GraMarRouMOR2000.pdf

    sii_dict = {}
    for loc, coef in mobius_dict.items():
        real_coef = np.real(coef)
        for subset in powerset(loc):
            contribution = real_coef / (1 + sum(loc) - sum(subset))
            if subset in sii_dict:
                sii_dict[subset] += contribution
            else:
                sii_dict[subset] = contribution
    return sii_dict


def mobius_to_banzhaf_ii(mobius_dict):
    # Equation (6) of https://ikojadin.perso.univ-pau.fr/kappalab/pub/GraMarRouMOR2000.pdf

    bii_dict = {}
    for loc, coef in mobius_dict.items():
        real_coef = np.real(coef)
        for subset in powerset(loc):
            contribution = real_coef / np.power(2.0, sum(loc) - sum(subset))
            if subset in bii_dict:
                bii_dict[subset] += contribution
            else:
                bii_dict[subset] = contribution
    return bii_dict

def mobius_to_faith_shapley_ii(mobius_dict, order):
    # Equation (16) of https://arxiv.org/pdf/2203.00870

    lower_order_mobius, higher_order_mobius = {}, {}
    for loc in mobius_dict.keys():
        if sum(loc) <= order:
            lower_order_mobius[loc] = np.real(mobius_dict[loc])
        else:
            higher_order_mobius[loc] = np.real(mobius_dict[loc])

    # find all projections to lower order terms from higher order terms
    fsii_dict = {}
    for loc, coef in tqdm(higher_order_mobius.items()):
        card = sum(loc)
        for subset in powerset(loc, order):
            card_subset = sum(subset)
            scaling = math.comb(card-1, order) / math.comb(card + order - 1, order + card_subset)
            if subset in fsii_dict:
                fsii_dict[subset] += scaling * coef
            else:
                fsii_dict[subset] = scaling * coef

    # apply weighting of these terms
    for loc, coef in fsii_dict.items():
        card = sum(loc)
        fsii_dict[loc] = coef * np.power(-1.0, order - card) * (card / (card+order)) * math.comb(order, card)

    # add in lower order_terms:
    for loc, coef in lower_order_mobius.items():
        if loc in fsii_dict:
            fsii_dict[loc] += coef
        else:
            fsii_dict[loc] = coef

    return fsii_dict

def mobius_to_faith_banzhaf_ii(mobius_dict, order):
    # Equation (13) of https://arxiv.org/pdf/2203.00870

    lower_order_mobius, higher_order_mobius = {}, {}
    for loc in mobius_dict.keys():
        if sum(loc) <= order:
            lower_order_mobius[loc] = np.real(mobius_dict[loc])
        else:
            higher_order_mobius[loc] = np.real(mobius_dict[loc])

    # find all projections to lower order terms from higher order terms
    fsii_dict = {}
    for loc, coef in tqdm(higher_order_mobius.items()):
        card = sum(loc)
        for subset in powerset(loc, order):
            card_subset = sum(subset)
            scaling = (1 / np.power(2.0, card - card_subset)) * math.comb(card - card_subset - 1, order - card_subset)
            if subset in fsii_dict:
                fsii_dict[subset] += scaling * coef
            else:
                fsii_dict[subset] = scaling * coef

    # apply weighting of these terms
    for loc, coef in fsii_dict.items():
        card = sum(loc)
        fsii_dict[loc] = coef * np.power(-1.0, order - card)

    # add in lower order_terms:
    for loc, coef in lower_order_mobius.items():
        if loc in fsii_dict:
            fsii_dict[loc] += coef
        else:
            fsii_dict[loc] = coef

    return fsii_dict

def mobius_to_shapley_taylor_ii(mobius_dict, order):
    # Equations (18-19) of https://arxiv.org/pdf/2402.02631

    stii_dict, higher_order_mobius = {}, {}
    for loc in mobius_dict.keys():
        if sum(loc) < order:
            stii_dict[loc] = np.real(mobius_dict[loc])
        else:
            higher_order_mobius[loc] = np.real(mobius_dict[loc])

    # find all projections to order terms from higher order terms
    for loc, coef in tqdm(higher_order_mobius.items()):
        contribution = coef / math.comb(sum(loc), order)
        nonzero_locs = [i for i, val in enumerate(loc) if val == 1]
        for subset in combinations(nonzero_locs, order):
            entry = np.zeros(len(loc)).astype(int)
            entry[list(subset)] = 1
            entry = tuple(entry)
            if entry in stii_dict:
                stii_dict[entry] += contribution
            else:
                stii_dict[entry] = contribution

    return stii_dict

def get_top_interactions(interaction_index_dict, inputs, order=None, top_k=5):

    if order is not None:
        order_ii_dict = {}
        for loc in interaction_index_dict.keys():
            if sum(loc) == order:
                order_ii_dict[loc] = interaction_index_dict[loc]
    else:
        order_ii_dict = interaction_index_dict

    significant_interactions = []
    for coord, coef in sorted(order_ii_dict.items(), key=lambda item: -np.abs(item[1]))[:top_k]:
        interaction = []
        for j in range(len(coord)):
            if coord[j] == 1:
                interaction.append(inputs[j])
        significant_interactions.append(tuple(interaction))
    return tuple(significant_interactions)


def plot_error_samples(baseline_results, fourier_results, dir_name):
    plt.clf()
    # baselines
    n_samples = baseline_results['n_samples']
    for method in baseline_results['methods']:
        mean = baseline_results['methods'][method]['sv_mse_mean']
        std = baseline_results['methods'][method]['sv_mse_std']
        plt.fill_between(n_samples, mean - std, mean + std, alpha=0.3)
        plt.loglog(n_samples, mean, label=f'{method}')

    for fourier_result in fourier_results:
        n_samples = fourier_result['n_samples']
        algo_name = next(filter(lambda x: 'FourierShap' in x, list(fourier_result.keys())))
        mean = fourier_result[algo_name]['sv_mse_mean']
        std = fourier_result[algo_name]['sv_mse_std']
        plt.fill_between(n_samples, mean - std, mean + std, alpha=0.3)
        plt.loglog(n_samples, mean, label=f'{algo_name}')

    DATASET = dir_name.split('/')[2].title()
    plt.xlabel('Samples')
    plt.ylabel('NMSE of Shapley Value')
    plt.title(f'Shapley Value Error vs Model Samples for {DATASET} Dataset')
    plt.legend()
    plt.tight_layout()
    plt.savefig(dir_name + '/error_samples.png', dpi=300)


def plot_error_time(baseline_results, fourier_results, dir_name):
    plt.clf()
    for method in baseline_results['methods']:
        errors = baseline_results['methods'][method]['sv_mse_mean']
        times = baseline_results['methods'][method]['avg_compute_times']
        plt.scatter(times.flatten(), errors.flatten(), alpha=0.3, label=method)

    for fourier_result in fourier_results:
        algo_name = next(filter(lambda x: 'FourierShap' in x, list(fourier_result.keys())))
        errors = fourier_result[algo_name]['sv_mse_mean']
        times = fourier_result[algo_name]['avg_compute_times']
        plt.scatter(times.flatten(), errors.flatten(), alpha=0.3, label=algo_name)

    DATASET = dir_name.split('/')[2].title()
    plt.xlabel('Clock Time (seconds)')
    plt.ylabel('NMSE of Shapley Value')
    plt.title(f'Shapley Value Error vs Total Compute Time for {DATASET} Dataset')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig(dir_name + '/error_time.png', dpi=300)


def plot_ranking_samples(baseline_results, fourier_results, dir_name):
    plt.clf()
    n_samples = baseline_results['n_samples']
    for method in baseline_results['methods']:
        mean = baseline_results['methods'][method]['sv_src_mean']
        std = baseline_results['methods'][method]['sv_src_std']
        plt.fill_between(n_samples, mean - std, mean + std, alpha=0.3)
        plt.loglog(n_samples, mean, label=f'{method}')

    for fourier_result in fourier_results:
        n_samples = fourier_result['n_samples']
        algo_name = next(filter(lambda x: 'FourierShap' in x, list(fourier_result.keys())))
        mean = fourier_result[algo_name]['sv_src_mean']
        std = fourier_result[algo_name]['sv_src_std']
        plt.fill_between(n_samples, mean - std, mean + std, alpha=0.3)
        plt.loglog(n_samples, mean, label=f'{algo_name}')

    DATASET = dir_name.split('/')[2].title()
    plt.xlabel('Samples')
    plt.ylabel('Spearman Ranking Correlation')
    plt.title(f'Spearman Ranking Correlation vs Model Samples for {DATASET} Dataset')
    plt.legend()
    plt.tight_layout()
    plt.savefig(dir_name + '/ranking_samples.png', dpi=300)


def plot_ranking_time(baseline_results, fourier_results, dir_name):
    plt.clf()
    for method in baseline_results['methods']:
        corrs = baseline_results['methods'][method]['sv_src_mean']
        times = baseline_results['methods'][method]['avg_compute_times']
        plt.scatter(times.flatten(), corrs.flatten(), alpha=0.3, label=method)

    for fourier_result in fourier_results:
        algo_name = next(filter(lambda x: 'FourierShap' in x, list(fourier_result.keys())))
        corrs = fourier_result[algo_name]['sv_src_mean']
        times = fourier_result[algo_name]['avg_compute_times']
        plt.scatter(times.flatten(), corrs.flatten(), alpha=0.3, label=algo_name)

    DATASET = dir_name.split('/')[2].title()
    plt.xlabel('Clock Time (seconds)')
    plt.ylabel('Spearman Ranking Correlation')
    plt.title(f'Spearman Ranking Correlation vs Total Compute Time for {DATASET} Dataset')
    plt.xscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig(dir_name + '/ranking_time.png', dpi=300)

def plot_spectrum(signal, sparse_recovery_results, fourier_spectrum_dir, n):
    print('Plotting Spectrum!')
    ### Assumes perfect recovery of sparse coefficients, except for faithfulness plots
    DATASET = fourier_spectrum_dir.split('/')[2].title()

    ### Plot 1: Avg magnitude by degree
    print('Plot 1')
    plt.clf()
    avg_magnitude_degrees = np.zeros(10 + 1)
    for loc, coef in sparse_recovery_results['transform'].items():
        hw = int(np.sum(loc))
        avg_magnitude_degrees[hw] += np.abs(np.real(coef))

    for i in range(10 + 1):
        if avg_magnitude_degrees[i] > 0.0:
            avg_magnitude_degrees[i] = avg_magnitude_degrees[i] / math.comb(n, i)

    plt.bar(range(1, 10 + 1), avg_magnitude_degrees[1:])
    plt.xlabel('Degree')
    plt.ylabel('Avg. Coefficient Magnitudes')
    plt.title(f'Avg. Coefficient Magnitudes vs Degree for {DATASET} Dataset')
    plt.tight_layout()
    plt.savefig(fourier_spectrum_dir + '/degree_magnitudes.png', dpi=300)
    plt.clf()

    ### Plot 2: Avg peeled magnitude by degree
    print('Plot 2')
    plt.clf()
    avg_magnitude_degrees = np.zeros(10 + 1)
    count_magnitude_degrees = np.zeros(10 + 1)
    for loc, coef in sparse_recovery_results['transform'].items():
        hw = int(np.sum(loc))
        avg_magnitude_degrees[hw] += np.abs(np.real(coef))
        count_magnitude_degrees[hw] += 1

    plt.bar(range(1, 10 + 1), avg_magnitude_degrees[1:] / count_magnitude_degrees[1:])
    plt.xlabel('Degree')
    plt.ylabel('Avg. Peeled Coefficient Magnitudes')
    plt.title(f'Avg. Peeled Coefficient Magnitudes vs Degree for {DATASET} Dataset')
    plt.tight_layout()
    plt.savefig(fourier_spectrum_dir + '/degree_peeled_magnitudes.png', dpi=300)
    plt.clf()

    plt.bar(range(10 + 1), count_magnitude_degrees)
    plt.xlabel('Degree')
    plt.ylabel('Count of Peeled Coefficients')
    plt.title(f'Count of Peeled Coefficients vs Degree for {DATASET} Dataset')
    plt.tight_layout()
    plt.savefig(fourier_spectrum_dir + '/degree_peeled_counts.png', dpi=300)
    plt.clf()

    ### Plot 3: Faithfulness by degree
    print('Plot 3')
    degree_dict = {}
    faithfulness_degree = np.zeros(10 + 1)

    for i in range(10 + 1):
        # add in coefficents of degree i
        for loc, coef in sparse_recovery_results['transform'].items():
            if int(np.sum(loc)) == i:
                degree_dict[loc] = coef
        faithfulness_degree[i] = estimate_r2(signal, degree_dict)
        if len(degree_dict) == len(sparse_recovery_results['transform']):
            faithfulness_degree[i:] = faithfulness_degree[i]
            break

    plt.semilogy(range(10 + 1), faithfulness_degree)
    plt.xlabel('Degree')
    plt.ylabel('Faithfulness')
    plt.title(f'Faithfulness vs. Degree for {DATASET} Dataset')
    plt.tight_layout()
    plt.savefig(fourier_spectrum_dir + '/degree_faithfulness.png', dpi=300)
    plt.clf()

    ### Plot 4: Faithfulness by top magnitude coordinates
    print('Plot 4')
    highest_log_10 = math.floor(20 * np.log10(2))
    num_coefs = np.round(np.logspace(0, highest_log_10, num=highest_log_10 * 4 + 1)).astype(int)

    faithfulness_coefs = np.zeros(len(num_coefs))
    sortable_dict = {k: np.abs(np.real(v)) for k, v in sparse_recovery_results['transform'].items()}
    sorted_keys = sorted(sparse_recovery_results['transform'], key=sortable_dict.get, reverse=True)
    for i, nc in enumerate(num_coefs):
        top_coefs_dict = {}
        for key in sorted_keys[:nc]:
            top_coefs_dict[key] = sparse_recovery_results['transform'][key]
        faithfulness_coefs[i] = estimate_r2(signal, top_coefs_dict)
        if len(top_coefs_dict) == len(sparse_recovery_results['transform']):
            faithfulness_coefs[i:] = faithfulness_coefs[i]
            break

    plt.loglog(num_coefs, faithfulness_coefs)
    plt.xlabel('Sparsity')
    plt.ylabel('Faithfulness')
    plt.title(f'Faithfulness vs. Sparsity for {DATASET} Dataset')
    plt.tight_layout()
    plt.savefig(fourier_spectrum_dir + '/sparsity_faithfulness.png', dpi=300)
    plt.clf()



class L0CV:
    def __init__(self, CV = 5):
        self.CV = CV

    def fit(self, X, y):
        pass

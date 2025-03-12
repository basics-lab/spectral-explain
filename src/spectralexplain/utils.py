from itertools import chain, combinations
import math
from tqdm import tqdm
import numpy as np
import itertools
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV


def estimate_r2(recon_function, saved_samples):
    """
    Estimate the R2 score of the reconstruction function.

    Parameters:
    - recon_function: The reconstruction function.
    - saved_samples: A tuple containing query indices and true values.

    Returns:
    - The R2 score of the reconstruction.
    """
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


def powerset(loc_tuple, max_order=None):
    """
    Generate the powerset of a location tuple up to a specified maximum order.

    Parameters:
    - loc_tuple: The location tuple.
    - max_order: The maximum order of the powerset (default is None).

    Returns:
    - A list of tuples representing the powerset.
    """
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


def fourier_to_mobius(fourier_dict):
    """
    Convert Fourier coefficients to Mobius coefficients.

    Parameters:
    - fourier_dict: A dictionary of Fourier coefficients.

    Returns:
    - A dictionary of Mobius coefficients.
    """
    if len(fourier_dict) == 0:
        return {}
    else:
        unscaled_mobius_dict = {}
        for loc, coef in fourier_dict.items():
            real_coef = np.real(coef)
            for subset in powerset(loc):
                if subset in unscaled_mobius_dict:
                    unscaled_mobius_dict[subset] += real_coef
                else:
                    unscaled_mobius_dict[subset] = real_coef

        # multiply each entry by (-2)^(cardinality)
        return {loc: val * np.power(-2.0, np.sum(loc)) for loc, val in unscaled_mobius_dict.items() if
                np.abs(val) > 1e-12}


def mobius_to_fourier(mobius_dict):
    """
    Convert Mobius coefficients to Fourier coefficients.

    Parameters:
    - mobius_dict: A dictionary of Mobius coefficients.

    Returns:
    - A dictionary of Fourier coefficients.
    """
    if len(mobius_dict) == 0:
        return {}
    else:
        unscaled_fourier_dict = {}
        for loc, coef in mobius_dict.items():
            real_coef = np.real(coef) / (2 ** sum(loc))
            for subset in powerset(loc):
                if subset in unscaled_fourier_dict:
                    unscaled_fourier_dict[subset] += real_coef
                else:
                    unscaled_fourier_dict[subset] = real_coef

        # multiply each entry by (-1)^(cardinality)
        return {loc: val * np.power(-1.0, np.sum(loc)) for loc, val in unscaled_fourier_dict.items() if
                np.abs(val) > 1e-12}


def mobius_to_shapley_ii(mobius_dict, **kwargs):
    """
    Convert Mobius coefficients to Shapley interaction indices.
    Equation (7) of https://ikojadin.perso.univ-pau.fr/kappalab/pub/GraMarRouMOR2000.pdf

    Parameters:
    - mobius_dict: A dictionary of Mobius coefficients.

    Returns:
    - A dictionary of Shapley interaction indices.
    """

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


def mobius_to_banzhaf_ii(mobius_dict, **kwargs):
    """
    Convert Mobius coefficients to Banzhaf interaction indices.
    Equation (6) of https://ikojadin.perso.univ-pau.fr/kappalab/pub/GraMarRouMOR2000.pdf

    Parameters:
    - mobius_dict: A dictionary of Mobius coefficients.

    Returns:
    - A dictionary of Banzhaf interaction indices.
    """

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
    """
    Convert Mobius coefficients to Faith-Shapley interaction indices.
    Equation (16) of https://arxiv.org/pdf/2203.00870

    Parameters:
    - mobius_dict: A dictionary of Mobius coefficients.
    - order: The max order of the FSII.

    Returns:
    - A dictionary of Faith-Shapley interaction indices.
    """
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
            scaling = math.comb(card - 1, order) / math.comb(card + order - 1, order + card_subset)
            if subset in fsii_dict:
                fsii_dict[subset] += scaling * coef
            else:
                fsii_dict[subset] = scaling * coef

    # apply weighting of these terms
    for loc, coef in fsii_dict.items():
        card = sum(loc)
        fsii_dict[loc] = coef * np.power(-1.0, order - card) * (card / (card + order)) * math.comb(order, card)

    # add in lower order_terms:
    for loc, coef in lower_order_mobius.items():
        if loc in fsii_dict:
            fsii_dict[loc] += coef
        else:
            fsii_dict[loc] = coef

    return fsii_dict


def mobius_to_faith_banzhaf_ii(mobius_dict, order):
    """
    Convert Mobius coefficients to Faith-Banzhaf interaction indices.
    Equation (13) of https://arxiv.org/pdf/2203.00870

    Parameters:
    - mobius_dict: A dictionary of Mobius coefficients.
    - order: The max order of the FBII.

    Returns:
    - A dictionary of Faith-Banzhaf interaction indices.
    """
    #

    lower_order_mobius, higher_order_mobius = {}, {}
    for loc in mobius_dict.keys():
        if sum(loc) <= order:
            lower_order_mobius[loc] = np.real(mobius_dict[loc])
        else:
            higher_order_mobius[loc] = np.real(mobius_dict[loc])

    # find all projections to lower order terms from higher order terms
    fbii_dict = {}
    for loc, coef in tqdm(higher_order_mobius.items()):
        card = sum(loc)
        for subset in powerset(loc, order):
            card_subset = sum(subset)
            scaling = (1 / np.power(2.0, card - card_subset)) * math.comb(card - card_subset - 1, order - card_subset)
            if subset in fbii_dict:
                fbii_dict[subset] += scaling * coef
            else:
                fbii_dict[subset] = scaling * coef

    # apply weighting of these terms
    for loc, coef in fbii_dict.items():
        card = sum(loc)
        fbii_dict[loc] = coef * np.power(-1.0, order - card)

    # add in lower order_terms:
    for loc, coef in lower_order_mobius.items():
        if loc in fbii_dict:
            fbii_dict[loc] += coef
        else:
            fbii_dict[loc] = coef
    return fbii_dict


def mobius_to_shapley_taylor_ii(mobius_dict, order):
    """
    Convert Mobius coefficients to Shapley-Taylor interaction indices.
    Equations (18-19) of https://arxiv.org/pdf/2402.02631

    Parameters:
    - mobius_dict: A dictionary of Mobius coefficients.
    - order: The order of the interaction.

    Returns:
    - A dictionary of Shapley-Taylor interaction indices.
    """

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
    """
    Get the top interactions from the interaction index dictionary.

    Parameters:
    - interaction_index_dict: A dictionary of interaction indices.
    - inputs: The input features.
    - order: The order of interactions to consider (default is None).
    - top_k: The number of top interactions to return (default is 5).

    Returns:
    - A tuple of the top interactions and their coefficients.
    """
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
        significant_interactions.append((tuple(interaction), np.round(coef, 3)))
    return tuple(significant_interactions)


def bin_vecs_low_order(m, order):
    """
    Generate binary vectors of length `m` with a maximum number of `order` ones.

    Parameters:
    - m: The length of the binary vectors.
    - order: The maximum number of ones in the binary vectors.

    Returns:
    - A numpy array of shape (m, num_vectors) containing the binary vectors.
    """
    num_of_ks = np.sum([math.comb(m, o) for o in range(order + 1)])
    K = np.zeros((num_of_ks, m))
    counter = 0
    for o in range(order + 1):
        positions = itertools.combinations(np.arange(m), o)
        for pos in positions:
            K[counter:counter+1, pos] = np.array(list(itertools.product(1 + np.arange(1), repeat=o)))
            counter += 1
    return K.T

def fit_regression(type, results, signal, n, b, fourier_basis=True):
    """
    Fit a regression model to the given signal data.

    Parameters:
    - type: The type of regression model to use ('linear', 'ridge', 'lasso').
    - results: A dictionary containing the locations of the support.
    - signal: The signal data to fit the model to.
    - n: The number of features in the signal.
    - b: The sparsity parameter used in the fit.
    - fourier_basis: Whether to use the Fourier basis (default is True).

    Returns:
    - A tuple containing the Fourier regression coefficients and the support.
    """
    assert type in ['linear', 'ridge', 'lasso']
    coordinates = []
    values = []
    for m in range(len(signal.all_samples)):
        for d in range(len(signal.all_samples[0])):
            for z in range(2 ** b):
                coordinates.append(signal.all_queries[m][d][z])
                values.append(np.real(signal.all_samples[m][d][z]))

    coordinates = np.array(coordinates)
    values = np.array(values)

    if len(results['locations']) == 0:
        support = np.zeros((1, n))
    else:
        support = results['locations']

    # add null and linear coefficients if not contained
    support = np.vstack([support, np.zeros(n), np.eye(n)])
    support = np.unique(support, axis=0)
    if fourier_basis:
        X = np.real(np.exp(coordinates @ (1j * np.pi * support.T)))
    else:
        X = ((coordinates @ support.T) >= np.sum(support, axis=1)).astype(int)
        X[:, 0] = 1

    if type == 'linear':
        reg = LinearRegression(fit_intercept=False).fit(X, values)
    elif type == 'lasso':
        reg = LassoCV(fit_intercept=False).fit(X, values)
    else:
        reg = RidgeCV(fit_intercept=False).fit(X, values)
    coefs = reg.coef_

    regression_coefs = {}
    for coef in range(support.shape[0]):
        regression_coefs[tuple(support[coef, :].astype(int))] = coefs[coef]

    if not fourier_basis:
        # solved in Mobius basis ({0,1}), transforming back to Fourier
        regression_coefs = mobius_to_fourier(regression_coefs)

    return regression_coefs, support
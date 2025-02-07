import time

import numpy as np
from .reconstruct import singleton_detection
from .input_signal_subsampled import SubsampledSignal
from .utils import qary_vec_to_dec, sort_qary_vecs, calc_hamming_weight, dec_to_qary_vec, qary_ints_low_order
import logging
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from functools import partial
import random
import copy
from spex.utils import mobius_to_fourier, fourier_to_mobius

logger = logging.getLogger(__name__)


def transform(signal: SubsampledSignal,
              reconstruct_method_source,
              reconstruct_method_channel,
              b,
              num_subsample,
              num_repeat,
              source_decoder=None,
              verbosity=0,
              report=False,
              timing_verbose=False,
              sort=False,
              noise_sd=0.0,
              peeling_method="multi-detect",
              refined=False,
              regress=None,
              peel_average=False,
              probabalistic_peel=False,
              trap_exit=False,
              res_energy_cutoff=0.9,
              ) -> dict:
    """
        Computes the q-ary Fourier transform of a signal object.

        Parameters:
        signal (SubsampledSignal): The signal object to be transformed.
        reconstruct_method_source (str): Method for source reconstruction.
        reconstruct_method_channel (str): Method for channel reconstruction.
        b (int): Parameter b used in the transformation.
        num_subsample (int): Number of subsamples.
        num_repeat (int): Number of repetitions.
        source_decoder (optional): Decoder for the source.
        verbosity (int): Level of verbosity for logging. Larger numbers lead to more detailed logs.
        report (bool): If True, outputs detailed information about the time taken for each transform step.
        timing_verbose (bool): If True, returns a dictionary with additional metrics.
        sort (bool): If True, returns the locations sorted in lexicographical order.
        noise_sd (float): Standard deviation of the noise.
        peeling_method (str): Method used for peeling. Default is "multi-detect".
        refined (bool): If True, refines the values from the data.
        regress (optional): Method for regression. Default is None.
        peel_average (bool): If True, averages the values when peeling the same index multiple times.
        probabalistic_peel (bool): If True, uses probabilistic peeling.
        trap_exit (bool): If True, exits the peeling loop if no progress is made for a certain number of iterations.
        res_energy_cutoff (float): Cutoff for residual energy. Default is 0.9.

        Returns:
        dict: A dictionary containing the results of the transformation.
        """

    logger.info(f"Peeling Method:{peeling_method}")
    q = signal.q
    n = signal.n

    omega = np.exp(2j * np.pi / q)
    result = []
    transform_dict = {}
    qsft_counts = {}

    # import data
    if isinstance(signal, SubsampledSignal):
        Ms, Ds, Us, Ts = signal.get_MDU(num_subsample, num_repeat, b, trans_times=True)
    else:
        raise NotImplementedError("QSFT currently only supports signals that inherit from SubsampledSignal")
    for i in range(len(Ds)):
        Us[i] = np.vstack(Us[i])
        Ds[i] = np.vstack(Ds[i])
    if regress in ["freq_domain", "freq_domain_lasso"]:
        initial_Us = copy.deepcopy(Us)
    transform_time = np.sum(Ts)
    if timing_verbose:
        logger.info(f"Transform Time:{transform_time}")
    Us = np.array(Us)
    if refined:
        samples = np.array(signal.all_samples).flatten()
        indicies = np.reshape(signal.all_queries, (len(samples), n))

        def refine_from_data(k_vec):
            nonlocal samples
            nonlocal indicies
            active = [i for i in range(len(k_vec)) if k_vec[i] == 1]
            signs = 1 - 2 * (np.sum(indicies[:, active], 1) % 2)
            inner_prod = sum(signs * samples)
            return inner_prod / len(samples)

    # Peeling Parameters
    peeling_max = q ** n
    peeled = set([])
    gamma = 0.5
    max_iter = 100
    peeled_at_iter = [0] * max_iter
    cutoff = 1e-9 + (1 + gamma) * (noise_sd ** 2) / (q ** b)  # noise threshold
    iter_step = 0
    cont_peeling = True
    num_peeling = 0
    multiton_count = 0
    zeroton_count = 0
    c = len(Ms)
    B = q ** b
    to_check = [[True] * B for _ in range(c)]
    logging.info(to_check)
    peeling_start = time.time()
    logging.info(f"cutoff = {cutoff}")
    logging.info(f"res_energy_cutoff = {res_energy_cutoff}")
    check_disable = False
    history_list = [dict() for _ in range(max_iter)]

    ####################################################################################################################
    # Peeling Loop
    ####################################################################################################################
    while cont_peeling and num_peeling < peeling_max and iter_step < max_iter:
        if verbosity >= 2:
            logger.info('-----')
            logger.info(f"iter {iter_step}")
        singletons = {}  # dictionary from (i, j) values to the true index of the singleton, k (peelable).
        multitons = []  # list of (i, j) values indicating where multitons are (too hard to peel).
        ################################################################################################################
        # Step 1: Identify potential coefficients to peel by going through every bin
        ################################################################################################################
        for i, (U, M, D) in enumerate(zip(Us, Ms, Ds)):
            for j, col in enumerate(U.T):
                j_qary = dec_to_qary_vec([j], q, b).T[0]
                valid_k_partial = partial(valid_k,
                                          col=col,
                                          j_qary=j_qary,
                                          M=M,
                                          D=D,
                                          q=q,
                                          res_energy_cutoff=res_energy_cutoff,
                                          cutoff=cutoff,
                                          peeling_method=peeling_method,
                                          )
                if np.linalg.norm(col) ** 2 > cutoff * len(col) and (to_check[i][j] or check_disable):
                    k = singleton_detection(
                        col,
                        method_channel=reconstruct_method_channel,
                        method_source=reconstruct_method_source,
                        q=q,
                        source_parity=signal.get_source_parity(),
                        nso_subtype="nso1",
                        source_decoder=source_decoder,
                        valid_k=valid_k_partial,
                    )
                    # This exists only because we currently can't peel multiple k from the same bin in the same iter
                    if type(k) == list:
                        any_bin_matching = [np.all((M.T @ k_el) % q == j_qary) for k_el in k]
                        matching_count = 0
                        cw = -1
                        for k_i in range(len(k)):
                            if any_bin_matching[k_i]:
                                matching_count += 1
                                cw = k[k_i]
                        if type(cw) == int:
                            bin_matching = False
                            k = k[0]
                        else:
                            k = cw
                            bin_matching = True
                        if matching_count > 1:
                            print("Warning: list-decoding is working, but is not being exploited")
                        logger.info(f"Multi-detection output more than 1 Codeword!")
                        print(f"Multi-detection output more than 1 Codeword!")
                    is_singleton, rho, residual = valid_k_partial(k)
                    if verbosity >= 5:
                        logger.info(f"({i}, {j}), res: {np.linalg.norm(residual) ** 2}, thresh: {cutoff * len(col)}")
                        logger.info(
                            f"frac. energy left: {(np.linalg.norm(residual) ** 2) / (np.linalg.norm(col) ** 2)}")
                    if (not is_singleton) or (probabalistic_peel and (random.random() < 0.2)):
                        multitons.append((i, j))
                        to_check[i][j] = is_singleton
                        if verbosity >= 6:
                            logger.info("We have a Multiton (Un-peelable)")
                            history_list[iter_step][(i, j)] = 2
                    else:  # declare as singleton
                        if refined:
                            refined_rho = refine_from_data(k)
                            if verbosity >= 3:
                                logger.info(f"Value refined from {rho}->{refined_rho}")
                            rho = refined_rho
                        singletons[(i, j)] = (k, rho)
                        if verbosity >= 3:
                            logger.info(f"We have a Singleton at " + "[" + " ".join(map(str, k)) + "]")
                            history_list[iter_step][(i, j)] = k
                else:
                    if verbosity >= 6:
                        if to_check[i][j]:
                            logger.info(f" ({i}, {j}) We have a Zeroton (nothing here)")
                        else:
                            logger.info(f" ({i}, {j}) skipping, nothing new.")
                        history_list[iter_step][(i, j)] = 0
                    to_check[i][j] = False
        # all singletons and multitons are discovered
        if verbosity >= 5:
            logger.info('singletons:')
            for ston in singletons.items():
                logger.info(f"\t {ston[0]} {qary_vec_to_dec(ston[1][0], q)}")
            logger.info("Multitons : {0}\n".format(multitons))
        multiton_count = len(multitons)
        zeroton_count = len(Us) * (q ** b) - len(singletons) - len(multitons)
        logger.info(f"{iter_step, zeroton_count, multiton_count}")

        ################################################################################################################
        # Step 2: Peel those peelable terms
        ################################################################################################################
        # if there were no singletons, terminate
        if not check_disable:
            cont_peeling = any(any(to_check[i]) for i in range(len(to_check)))
        elif len(singletons) == 0:
            cont_peeling = False
        if trap_exit:
            if iter_step >= 15 and peeled_at_iter[iter_step - 1] == peeled_at_iter[iter_step - 15]:
                cont_peeling = False
        # balls to peel
        balls_to_peel = set()
        ball_values = {}
        peeled_counter = {}
        for (i, j) in singletons:
            k, rho = singletons[(i, j)]
            ball = tuple(k)  # Must be a hashable type
            if peel_average:
                if ball in peeled_counter:
                    rho = (ball_values[ball] * peeled_counter[ball] + rho) / (peeled_counter[ball] + 1)
                    peeled_counter[ball] += 1
                else:
                    peeled_counter[ball] = 1
            balls_to_peel.add(ball)
            ball_values[ball] = rho
        if verbosity >= 5:
            logger.info('these balls will be peeled')
            logger.info(balls_to_peel)
        # peel
        for ball in balls_to_peel:
            num_peeling += 1
            k = np.array(ball)[..., np.newaxis]
            potential_peels = [(l, qary_vec_to_dec(M.T.dot(k) % q, q)[0]) for l, M in enumerate(Ms)]
            result.append((k.T[:][0], ball_values[ball], iter_step))
            if verbosity >= 6:
                k_dec = qary_vec_to_dec(k, q)
                peeled.add(int(k_dec))
                logger.info("Processing Singleton {0}".format(k_dec.T))
                logger.info(k.T)
                for (l, j) in potential_peels:
                    logger.info("The singleton appears in M({0}), U({1})".format(l, j))
                    history_list[iter_step][ball] = potential_peels

            for peel in potential_peels:
                signature_in_stage = omega ** (Ds[peel[0]] @ k)
                to_subtract = ball_values[ball] * signature_in_stage.reshape(-1, 1)
                if verbosity >= 6:
                    logger.info("Peeled ball {0} off bin {1}".format(qary_vec_to_dec(k, q), peel))
                Us[peel[0]][:, peel[1]] -= np.array(to_subtract)[:, 0]
                to_check[peel[0]][peel[1]] = True
            if verbosity >= 5:
                logger.info("Iteration Complete: The peeled indicies are:")
                logger.info(np.sort(list(peeled)))
        peeled_at_iter[iter_step] = len(peeled)
        iter_step += 1
    ####################################################################################################################
    # Step 3: Do some summing if we peeled the same index multiple times
    ####################################################################################################################
    loc = set()
    for k, value, iter_step in result:
        loc.add(tuple(k))
        transform_dict[tuple(k)] = transform_dict.get(tuple(k), 0) + value

    peeling_time = time.time() - peeling_start
    if timing_verbose:
        logger.info(f"Peeling Time:{peeling_time}")

    ####################################################################################################################
    # Step 4: Format the output as required
    ####################################################################################################################
    if not report:
        return transform_dict
    else:
        n_samples = np.prod(np.shape(np.array(Us)))
        if len(loc) > 0:
            loc = list(loc)
            if sort:
                loc = sort_qary_vecs(loc)
            avg_hamming_weight = np.mean(calc_hamming_weight(loc))
            max_hamming_weight = np.max(calc_hamming_weight(loc))
        else:
            loc, avg_hamming_weight, max_hamming_weight = [], 0, 0
        result = {
            "transform": transform_dict,
            "runtime": transform_time + peeling_time,
            "n_samples": n_samples,
            "locations": loc,
            "avg_hamming_weight": avg_hamming_weight,
            "max_hamming_weight": max_hamming_weight,
            "fail_low_noise": len(loc) == 0 and zeroton_count <= 10,
            "fail_high_noise": len(loc) == 0 and multiton_count <= 10,
            "history": history_list,
        }
        regress_start = time.time()
        if regress is not None:
            new_transform_dict, _ = fit_regression(regress, result, signal, n, b)
            result['transform'] = new_transform_dict
            logger.info(f"Regression Time:{time.time() - regress_start}")
        return result


def valid_k(k, col, j_qary, M, D, q, res_energy_cutoff, cutoff, peeling_method):
    """
    Validates if a given vector k is a singleton based on the provided parameters.

    Parameters:
    k (np.ndarray): The vector to be validated.
    col (np.ndarray): The column vector from the subsampled signal.
    j_qary (np.ndarray): The q-ary representation of the index j.
    M (np.ndarray): The matrix used in the QSFT algorithm.
    D (np.ndarray): The matrix used in the QSFT algorithm.
    q (int): The base of the q-ary Fourier transform.
    res_energy_cutoff (float): The cutoff for the residual energy.
    cutoff (float): The noise threshold.
    peeling_method (str): The method used for peeling ('multi-detect' or other).

    Returns:
    tuple: A tuple containing a boolean indicating if k is a valid singleton, the estimated coefficient rho, and the residual vector.
    """
    omega = np.exp(2j * np.pi / q)
    signature = omega ** (D @ k)
    rho = np.dot(np.conjugate(signature), col) / D.shape[0]
    residual = col - rho * signature
    bin_matching = np.all((M.T @ k) % q == j_qary)
    if peeling_method == "multi-detect":
        # Check if the residual has less than x% of the remaining energy
        peel_condition = (np.linalg.norm(residual) ** 2) / (np.linalg.norm(col) ** 2) < res_energy_cutoff
    else:
        # Check if the residual is small
        peel_condition = np.linalg.norm(residual) ** 2 < cutoff * len(col)
    return (peel_condition and bin_matching), rho, residual


def fit_regression(type, results, signal, n, b, fourier_basis=True, coordinates=None, values=None):
    """
    Fits a regression model to the given signal data.

    Parameters:
    type (str): The type of regression model to use ('linear', 'lasso', or 'ridge').
    results (dict): A dictionary containing the results of the transformation, including the locations of non-zero indices.
    signal (SubsampledSignal): The signal object containing the data to be used for regression.
    n (int): The length of the signal.
    b (int): The number of bits used in the transformation.

    Returns:
    tuple: A tuple containing the regression coefficients and the support locations.
    """
    assert type in ['linear', 'ridge', 'lasso']
    if coordinates is None and values is None:
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

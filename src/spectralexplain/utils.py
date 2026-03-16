from itertools import chain, combinations
try:
    from gurobipy import Model, GRB, LinExpr, Env
except ImportError:
    pass
try:
    import pandas as pd
    import lightgbm as lgb
    from sklearn.model_selection import GridSearchCV
except ImportError:
    pass
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

def lgboost_tree_to_fourier(tree_info):
    """
    Strips the Fourier coefficients from an LGBoost tree
    Code adapted from:
        Gorji, Ali, Andisheh Amrollahi, and Andreas Krause.
        "Amortized SHAP values via sparse Fourier function approximation."
        arXiv preprint arXiv:2410.06300 (2024).
    """

    def fourier_tree_sum(left_fourier, right_fourier, feature):
        final_fourier = {}
        all_freqs_tuples = set(left_fourier.keys()).union(right_fourier.keys())
        for freq_tuple in all_freqs_tuples:
            final_fourier[freq_tuple] = (left_fourier.get(freq_tuple, 0) + right_fourier.get(freq_tuple, 0)) / 2
            current_freq_set = set(freq_tuple)
            feature_set = {feature}
            united_set = current_freq_set.union(feature_set)
            final_fourier[tuple(sorted(united_set))] = (0.5 * left_fourier.get(freq_tuple, 0)
                                                        - 0.5 * right_fourier.get(freq_tuple, 0))
        return final_fourier

    def dfs(node):
        if 'leaf_value' in node:  # Leaf node in LightGBM JSON
            return {tuple(): node['leaf_value']}
        else:  # Split node
            left_fourier = dfs(node['left_child'])
            right_fourier = dfs(node['right_child'])
            feature_index = node['split_feature']  # Feature index for LightGBM
            return fourier_tree_sum(left_fourier, right_fourier, feature_index)

    return dfs(tree_info['tree_structure'])


def lgboost_to_fourier(model):
    final_fourier = []
    dumped_model = model.booster_.dump_model()
    for tree_info in dumped_model['tree_info']:
        final_fourier.append(lgboost_tree_to_fourier(tree_info))

    combined_fourier = {}
    for fourier in final_fourier:
        for k, v in fourier.items():
            tuple_k = [0] * model.n_features_
            for feature in k:
                tuple_k[feature] = 1
            tuple_k = tuple(tuple_k)
            if tuple_k in combined_fourier:
                combined_fourier[tuple_k] += v
            else:
                combined_fourier[tuple_k] = v
    return combined_fourier


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


def proxy_spex(samples, num_train_samples=None, **kwargs):
    if num_train_samples is None:
        num_train_samples = len(samples[0])
    
    max_order = kwargs.get('max_order', 5)
    max_depth_choices = [max(1, max_order - 2), max_order]
    param_grid = {
        'max_depth': sorted(list(set(max_depth_choices))),
        'n_estimators': [500, 1000, 5000],
        'learning_rate': [0.01, 0.1],
    }

    # 2. Create a base model with fixed parameters
    cols = [f"f{i}" for i in range(samples[0].shape[1])]

    train_X = pd.DataFrame(
        samples[0][:num_train_samples],
        columns=cols
    )
    train_y = samples[1][:num_train_samples]
    
    base_model = lgb.LGBMRegressor(
        verbose=-1,
        n_jobs=-1,
        random_state=0
    )
    
    # 3. Set up GridSearchCV with cross-validation
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring='r2',
        cv=5,
        verbose=0,
        n_jobs=1
    )

    # 4. Fit the model on the training data
    grid_search.fit(train_X, train_y)

    best_model = grid_search.best_estimator_
    four_dict = lgboost_to_fourier(best_model)
    
    # find index of null Fourier coefficient
    list_keys = list(four_dict.keys())
    nfc_idx = None
    if tuple([0] * samples[0].shape[1]) in list_keys:
        nfc_idx = list_keys.index(tuple([0] * samples[0].shape[1]))
    four_coefs = np.array(list(four_dict.values()))
    if nfc_idx is not None:
        four_coefs[nfc_idx] = 0
    four_coefs_sq = four_coefs ** 2
    tot_energy = np.sum(four_coefs_sq)
    sorted_four_coefs = np.sort(four_coefs_sq)[::-1]
    thresh_idx_95 = np.argmin(np.cumsum(sorted_four_coefs / tot_energy) < .95)
    thresh = np.sqrt(sorted_four_coefs[thresh_idx_95])
    four_dict_trunc = {k: v for k,v in four_dict.items() if abs(v) > thresh}
    support = np.array(list(four_dict_trunc.keys()))
    
    X = np.real(np.exp(train_X @ (1j * np.pi * support.T)))
    reg = RidgeCV(alphas=np.logspace(-6, 6, 100), fit_intercept=False).fit(X, train_y)

    regression_coefs = {}
    for coef in range(support.shape[0]):
        regression_coefs[tuple(support[coef, :].astype(int))] = reg.coef_[coef]
    return regression_coefs


class ExactSolver:
    def __init__(self, fourier_dictionary, maximize=True, max_solution_order=None, exact_solution_order=None):
        if "Env" not in globals():
            raise ImportError("The 'gurobipy' library is required to use ExactSolver. Please install it with 'pip install gurobipy', and ensure you have a valid Gurobi license.")
        
        self.maximize = maximize
        self.max_solution_order = max_solution_order
        self.exact_solution_order = exact_solution_order
        
        self.fourier_dictionary = fourier_dictionary
        assert len(self.fourier_dictionary) > 0, "Empty Dictionary"
        self.n = len(list(self.fourier_dictionary.keys())[0])
        self.mobius_dictionary = self.fourier_to_mobius(self.fourier_dictionary)

        self.baseline_value = self.mobius_dictionary[tuple([0] * self.n)] if tuple(
            [0] * self.n) in self.mobius_dictionary else 0

        self.initialize_model()


    def fourier_to_mobius(self, fourier_dict):
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
                for subset in self.all_subsets(np.nonzero(loc)[0]):
                    one_hot_subset = tuple([1 if i in subset else 0 for i in range(self.n)])
                    if one_hot_subset in unscaled_mobius_dict:
                        unscaled_mobius_dict[one_hot_subset] += real_coef
                    else:
                        unscaled_mobius_dict[one_hot_subset] = real_coef

            # multiply each entry by (-2)^(cardinality)
            return {loc: val * np.power(-2.0, np.sum(loc)) for loc, val in unscaled_mobius_dict.items()}

    def all_subsets(self, iterable, order=None):
        """
        Returns all subset tuples of the given iterable.
        """
        if not order:
            return list(chain.from_iterable(combinations(iterable, r) for r in range(len(iterable) + 1)))
        else:
            return list(chain.from_iterable(combinations(iterable, r) for r in range(order, order + 1)))

    def initialize_model(self):
        import os
        try:
            self.env = Env(empty=True)
            self.env.setParam("OutputFlag",   0)   # suppress solver progress log
            self.env.setParam("LogToConsole", 0)   # suppress licence/banner/parameter echo
        
            # Support WLS/Cloud credentials via environment variables
            wls_access_id = os.environ.get("GRB_WLSACCESSID")
            wls_secret = os.environ.get("GRB_WLSSECRET")
            license_id = os.environ.get("GRB_LICENSEID")

            if wls_access_id and wls_secret and license_id:
                self.env.setParam("WLSAccessID", wls_access_id)
                self.env.setParam("WLSSecret", wls_secret)
                self.env.setParam("LicenseID", int(license_id))

            self.env.start()                         # now start the quiet environment
            self.model = Model("Mobius Maximization Problem", env=self.env)
        except Exception as e:
            raise RuntimeError(
                f"Gurobi failed to initialize: {e}. "
                "Please ensure you have a valid local 'gurobi.lic' or set the following environment variables: "
                "GRB_WLSACCESSID, GRB_WLSSECRET, and GRB_LICENSEID."
            ) from e
        vars = [(tuple(np.nonzero(key)[0]), val) for key, val in self.mobius_dictionary.items() if sum(key) > 0]
        self.locs, self.coefs = [i[0] for i in vars], [i[1] for i in vars]
        # add in all first order terms
        self.missing_first_order = []
        for i in range(self.n):
            if (i,) not in self.locs:
                self.missing_first_order.append(i)
        
        locs_set = set(self.locs)

        # Define the variables and objective function
        self.y = self.model.addVars(len(self.locs), vtype=GRB.BINARY, name="y")
        self.model.setObjective(
            sum(self.coefs[i] * self.y[i] for i in range(len(self.locs))),
            GRB.MAXIMIZE if self.maximize else GRB.MINIMIZE
        )

        # Constraint 1: y_S <= y_R \forall R \subset S, \forall S
        count_constraint_1, count_constraint_2 = 0, 0
        for i, loc in enumerate(self.locs):
            for loc_subset in self.all_subsets(loc, order=len(loc) - 1):
                if loc_subset in locs_set and loc_subset != loc:
                    j = self.locs.index(loc_subset)
                    self.model.addConstr(self.y[i] <= self.y[j])
                    count_constraint_1 += 1

        # Constraint 2: \sum_{i \in S} y_{i} <= |S| + y_S - 1, \forall S
        for i, loc in enumerate(self.locs):
            if len(loc) > 1:
                expr = LinExpr()
                for idx in loc:
                    if (idx,) in locs_set:
                        expr.add(self.y[self.locs.index((idx,))])
                self.model.addConstr(expr <= len(loc) + self.y[i] - 1)
                count_constraint_2 += 1

        # (Optional) Constraint 3: \sum_{i \in n} y_{i} <= n
        if self.exact_solution_order is not None:
            expr = LinExpr()
            for idx in range(self.n):
                if (idx,) in locs_set:
                    expr.add(self.y[self.locs.index((idx,))])
            self.model.addConstr(expr <= self.exact_solution_order)
        else:
            if self.max_solution_order is not None:
                expr = LinExpr()
                for idx in range(self.n):
                    if (idx,) in locs_set:
                        expr.add(self.y[self.locs.index((idx,))])
                self.model.addConstr(expr <= self.max_solution_order)

    def solve(self):
        self.model.optimize()
        
        if self.model.SolCount == 0:
            status = self.model.Status
            raise RuntimeError(f"Gurobi failed to find a solution. Status code: {status}. Check if the problem is infeasible or hit a limit.")

        # Print the optimal values
        argmax = [0] * self.n
        for i in range(len(self.locs)):
            if len(self.locs[i]) == 1 and self.y[i].X > 0.5:
                argmax[self.locs[i][0]] = 1

        if self.exact_solution_order is not None:
            old_sum = sum(argmax)
            if old_sum < self.exact_solution_order:
                for i in self.missing_first_order[:self.exact_solution_order-old_sum]:
                    argmax[i] = 1

        return argmax


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
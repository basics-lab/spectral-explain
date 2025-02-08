from spex.support_recovery import sampling_strategy, support_recovery, get_num_samples
from spex.utils import *
import shutil
class Explainer:
    def __init__(self, value_function, num_features, index="FBII", budget=None, max_order=5):
        self.value_function = value_function
        self.num_features = num_features
        self.max_order = max_order

        if budget is None:
            self.sparsity_parameter = 8
        else:
            self.sparsity_parameter = self.compute_sparsity_parameter(self.num_features, budget, self.max_order)
        self.budget = budget

        self.fourier_transform = self.compute_interaction_values()

    def compute_sparsity_parameter(self, num_features, budget, max_order):
        signal, budget_thresholds = sampling_strategy(lambda X: [0] * len(X), 12, self.num_features, sample_save_dir=None, t=self.max_order)
        self.budget_thresholds = budget_thresholds
        assert budget > budget_thresholds[0], \
            f"budget must be greater than {budget_thresholds[0]}, recommended at least {budget_thresholds[1]}"
        return 2 + next(i for i, samples in enumerate(budget_thresholds) if budget < samples)


    def compute_interaction_values(self):
        signal, _ = sampling_strategy(self.value_function, self.sparsity_parameter,
                                   self.num_features, sample_save_dir=None, t=self.max_order)
        fourier_transform = support_recovery("soft", signal, self.sparsity_parameter, self.max_order)["transform"]
        return fourier_transform

    def get_interaction_values(self, index):
        assert index.lower() in ["fourier", "mobius", "fsii", "fbii", "stii", "sii", "bii"], \
            "index must belong to set {Fourier, Mobius, FSII, FBII, STII, SII, BII}"
        if index.lower() == "fourier":
            return self.fourier_transform
        else:
            mobius_transform = fourier_to_mobius(self.fourier_transform)
            if index.lower() == "mobius":
                return mobius_transform
            elif index.lower() == "fsii":
                return mobius_to_faith_shapley_ii(mobius_transform, self.max_order)
            elif index.lower() == "fbii":
                return mobius_to_faith_banzhaf_ii(mobius_transform, self.max_order)
            elif index.lower() == "stii":
                return mobius_to_shapley_taylor_ii(mobius_transform, self.max_order)
            elif index.lower() == "sii":
                return mobius_to_shapley_ii(mobius_transform)
            else:
                return mobius_to_banzhaf_ii(mobius_transform)
from spex.support_recovery import sampling_strategy, support_recovery, get_num_samples

from .Interactions import Interactions

class Explainer:
    def __init__(self, value_function, num_features, index="FBII", budget=None, max_order=5):
        self.value_function = value_function
        self.num_features = num_features
        self.max_order = max_order

        if budget is None:
            self.sparsity_parameter = 8
        else:
            self.sparsity_parameter = self.compute_sparsity_parameter(budget)
        self.budget = budget

        self.fourier_transform = self.compute_interaction_values()
        self.interactions = Interactions(self.fourier_transform, index)

    def compute_sparsity_parameter(self, budget):
        signal, budget_thresholds = sampling_strategy(lambda X: [0] * len(X), 3, 12, self.num_features,
                                                      sample_save_dir=None, t=self.max_order)
        self.budget_thresholds = budget_thresholds
        print(self.budget_thresholds)
        print(budget_thresholds[3])
        assert budget > budget_thresholds[3], \
            f"budget must be greater than {budget_thresholds[0]}, recommended at least {budget_thresholds[1]}"
        return next(b for b, samples in budget_thresholds.items() if budget < samples)


    def compute_interaction_values(self):
        signal, _ = sampling_strategy(self.value_function, self.sparsity_parameter,
                                   self.num_features, sample_save_dir=None, t=self.max_order)
        fourier_transform = support_recovery("soft", signal, self.sparsity_parameter, self.max_order)["transform"]
        return fourier_transform

    def interactions(self):
        return self.interactions

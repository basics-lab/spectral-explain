from spex.utils import *


class Interactions:
    def __init__(self, fourier_transform, features, index, sample_budget):
        assert len(fourier_transform) > 0, "Fourier transform is empty"
        self.fourier_transform = fourier_transform
        self.features = features
        self.index = index
        self.num_features = len(list(fourier_transform.keys())[0])
        self.sample_budget = sample_budget

        self.interactions = self.sort_interactions(self.nonzero_keys(self.convert_fourier_interactions()))
        self.max_order = max([len(key) for key in self.interactions.keys()])
        self.is_baseline_value = tuple() in self.interactions
        self.baseline_value = self.interactions[tuple()] if self.is_baseline_value else 0

        self.num_interactions = len(self.interactions)

    def convert_fourier_interactions(self):
        if self.index.lower() == "fourier":
            return self.fourier_transform
        else:
            mobius_transform = fourier_to_mobius(self.fourier_transform)
            if self.index.lower() == "mobius":
                return mobius_transform
            else:
                max_order = self.get_max_order(mobius_transform)
                return {"fsii": mobius_to_faith_shapley_ii,
                        "fbii": mobius_to_faith_banzhaf_ii,
                        "stii": mobius_to_shapley_taylor_ii,
                        "sii": mobius_to_shapley_ii,
                        "bii": mobius_to_banzhaf_ii,
                        }.get(self.index.lower(), NotImplementedError())(mobius_transform, order=max_order)

    def sort_interactions(self, interactions):
        return dict(sorted(interactions.items(), key=lambda item: -abs(item[1])))

    def get_max_order(self, dictionary):
        return max([sum(locs) for locs in dictionary.keys()])

    def get_interactions(self):
        return self.interactions

    def nonzero_keys(self, dictionary):
        returned_dict = {}
        for key, value in dictionary.items():
            features_tuple = tuple([self.features[j] for j in tuple(np.nonzero(key)[0])])
            returned_dict[features_tuple] = value
        return returned_dict

    def __str__(self):
        header = (
            f"Interactions(\n"
            f"    index={self.index.upper()}, max_order={self.max_order}, baseline_value={np.round(self.baseline_value,3)}\n"
            f"    sample_budget={self.sample_budget}, num_features={self.num_features},\n"
            f"    Top Interactions:"
        )
        top_5_interactions = []
        for key, value in self.interactions.items():
            if len(key) > 0:
                top_5_interactions.append((key, value))
            if len(top_5_interactions) == 5:
                break

        interactions_str = "\n".join([
            f"\t\t{indices}: {np.round(value, 3)}"
            for indices, value in top_5_interactions]
        )

        return f"{header}\n{interactions_str}\n)"

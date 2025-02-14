from spex.utils import *

class Interactions:

    def __init__(self, fourier_interactions, interaction_index):
        assert len(fourier_interactions) > 0, "Fourier interactions must be non-empty"
        self.fourier_interactions = fourier_interactions
        self.interactions = self.convert_fourier_to_interaction(interaction_index).sort_interactions()
        self.interaction_index = interaction_index
        self.num_features = len(list(self.interactions.keys())[0])
        self.max_order = max([sum(interaction) for interaction in self.interactions.keys()])


    def convert_fourier_to_interaction(self, interaction_index):
        assert interaction_index.lower() in ["fourier", "mobius", "fsii", "fbii", "stii", "sii", "bii"], \
            "index must belong to set {Fourier, Mobius, FSII, FBII, STII, SII, BII}"
        if interaction_index.lower() == "fourier":
            return self.fourier_interactions
        else:
            mobius_transform = fourier_to_mobius(self.fourier_interactions)
            if interaction_index.lower() == "mobius":
                return mobius_transform
            elif interaction_index.lower() == "fsii":
                return mobius_to_faith_shapley_ii(mobius_transform, self.max_order)
            elif interaction_index.lower() == "fbii":
                return mobius_to_faith_banzhaf_ii(mobius_transform, self.max_order)
            elif interaction_index.lower() == "stii":
                return mobius_to_shapley_taylor_ii(mobius_transform, self.max_order)
            elif interaction_index.lower() == "sii":
                return mobius_to_shapley_ii(mobius_transform)
            else:
                return mobius_to_banzhaf_ii(mobius_transform)

    def __str__(self):
        interactions_str = ""
        for loc, value in self.interactions.keys():
            interactions_str += f"{np.nonzeros(loc)[0]}: {np.round(value,3)}\n"

        return (
            f"Interaction(\n"
            f"\tindex={self.interaction_index}\n"
            f"\tnum features={self.num_features}\n"
            f"\tmax order={self.max_order}\n"
            f"\ttop interactions:\n"
            f"\t{interactions_str}"
            f")"
        )

    def sort_interactions(self):
        return sorted(self.interactions.items(), key=lambda x: x[1])

    def get_interactions_by_order(self, order):
        return {interaction: self.interactions[interaction] for interaction in self.interactions if sum(interaction) == order}